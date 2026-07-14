#include "search.h"
#include "../eval/eval.h"
#include "../moveutils.h"
#include "../fathom/tbprobe.h"
#include <cmath>
#include <iostream>
#include <string>

using namespace chess;

// This file is organized in dependency order (later sections may call earlier ones). 
// Reading top to bottom:
//
//   search              - time management (public API, used by uci.cpp)
//   search::tb          - Syzygy/Fathom tablebase probing (root + mid-search)
//   search::see         - static exchange evaluation (capture ordering)
//   search::ordering    - move scoring/sorting + killer/history bookkeeping
//   search::qsearch     - quiescence search (capture-only leaf resolution)
//   search::pruning     - the shallow-depth pruning/reduction techniques
//                         used by alphaBeta: RFP, razoring, IIR, futility,
//                         and the LMR reduction formula
//   search::alphabeta   - the main search: draw/mate-distance pruning, TB
//                         probe, TT probe, check extension, null-move
//                         pruning, the PVS move loop, TT store
//   search              - iterativeDeepening: root TB probe, aspiration
//                         windows, UCI "info"/"bestmove" output
//
// Nothing outside this translation unit calls into any of the sub-
// namespaces below directly — search.h only declares SearchContext,
// Limits and iterativeDeepening()

namespace search
{

  // Max legal moves in any chess position is 218 (a well-known bound);
  // used to size fixed arrays for move scoring and quiet-move tracking
  // without heap allocation in the hot path.
  constexpr size_t MAX_MOVES = 218;

  // Converts a score computed AT ply `ply` into a ply-independent form
  // safe to store in the TT, and back again when read at a (possibly
  // different) ply on a later transposition hit. A mate score only
  // means "mate in K plies from here" if K is extracted correctly, and K
  // is ply-independent while the raw returned score is not.
  constexpr int valueToTT(int v, int ply)
  {
    if (v >= MATE_SCORE - MAX_PLY)
      return v + ply;
    if (v <= -(MATE_SCORE - MAX_PLY))
      return v - ply;
    return v;
  }

  constexpr int valueFromTT(int v, int ply)
  {
    if (v >= MATE_SCORE - MAX_PLY)
      return v - ply;
    if (v <= -(MATE_SCORE - MAX_PLY))
      return v + ply;
    return v;
  }

  bool SearchContext::timeUp() const
  {
    if (limits.infinite)
      return false;
    if (limits.nodes && nodes >= limits.nodes)
      return true;
    if (allocatedMs <= 0)
      return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startTime)
                       .count();
    return elapsed >= allocatedMs;
  }

  int estimateMovesToGo(const Limits &limits, int moveNumber)
  {
    if (limits.movestogo > 0)
      return limits.movestogo;
    if (moveNumber < 0)
      return 30;
    int mtg = 45 - moveNumber / 2;
    return std::clamp(mtg, 15, 45);
  }

  // A move's time allowance, split into two numbers instead of one:
  //   - optimalMs ("soft" budget): how long we'd LIKE to spend.
  //     iterativeDeepening checks this between completed depths to decide
  //     whether it's worth starting another (~2-3x-the-nodes) iteration at
  //     all, rather than only finding out we've overspent via a hard cutoff
  //     mid-search.
  //   - maximumMs ("hard" budget): the absolute ceiling, enforced by
  //     timeUp(). Deliberately larger than optimalMs so a root that's unstable 
  //     right now (best move just changed, or the score just fell sharply) can
  //     keep searching past the soft plan instead of being cut off mid-thought
  //     — the two numbers only ever diverge in exactly that situation;
  //     see iterativeDeepening.
  //
  // Both numbers respect the safety rails: never plan to use more than 85% of
  // what's left, and never eat into the final 50ms reserved so "bestmove" 
  // can still be sent on time.
  struct TimeBudget
  {
    int64_t optimalMs;
    int64_t maximumMs;
  };

  TimeBudget computeTimeBudget(const Limits &limits, Color us, int moveNumber)
  {
    if (limits.movetime > 0)
    {
      // Clamped to at least 1ms: for movetime <= 50, `movetime - 50`
      // used to go to zero or negative, and timeUp() treats an
      // allocatedMs <= 0 as "no time limit at all" (see timeUp() above) —
      // so a short movetime command would silently make the engine ignore
      // its own deadline and search until depth/node limits took over
      // instead. Clamping keeps the 50ms safety buffer's intent (leave
      // headroom for UCI I/O) without that fall-through.
      int64_t t = std::max<int64_t>(limits.movetime - 50, 1);
      return {t, t};
    }

    int64_t myTime = (us == Color::WHITE) ? limits.wtime : limits.btime;
    int64_t myInc = (us == Color::WHITE) ? limits.winc : limits.binc;

    if (myTime <= 0)
      return {0, 0};

    int movesToGo = estimateMovesToGo(limits, moveNumber);

    int64_t optimal = myTime / movesToGo + myInc;
    int64_t hardCap = (myTime * 85) / 100;
    optimal = std::min(optimal, hardCap);
    optimal = std::max<int64_t>(optimal, 20);
    optimal = std::min<int64_t>(optimal, myTime - 50 > 0 ? myTime - 50 : optimal);

    // Maximum: up to 3x the soft plan, still inside the same 85%/50ms
    // rails optimal itself respects.
    int64_t maximum = std::min<int64_t>(optimal * 3, hardCap);
    maximum = std::max(maximum, optimal);
    maximum = std::min<int64_t>(maximum, myTime - 50 > 0 ? myTime - 50 : maximum);

    return {optimal, maximum};
  }

} // namespace search

// search::tb — Syzygy tablebase probing via Fathom.
//
// Two call sites share the same preconditions (no castling rights left for
// either side, since Syzygy WDL/DTZ tables don't model castling at all) but
// differ in what they do with the result: the root probe (called once, before
// any search) can just play the perfect move outright; the mid-search probe
// (called at every node during alphaBeta) has to fold the result into the
// alpha/beta window like a bound, because unlike the root we still need a
// score to propagate back up through the tree.

namespace search::tb
{

  // Builds the TB_CASTLING_* bitmask Fathom expects from the engine's own
  // castling-rights representation.
  unsigned castlingMask(const Board &board)
  {
    unsigned c = 0;
    auto cr = board.castlingRights();
    if (cr.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE))
      c |= TB_CASTLING_K;
    if (cr.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE))
      c |= TB_CASTLING_Q;
    if (cr.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE))
      c |= TB_CASTLING_k;
    if (cr.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE))
      c |= TB_CASTLING_q;
    return c;
  }

  // Fathom wants 0 for "no en-passant square" rather than a sentinel —
  // translate from the engine's own Square::NO_SQ.
  unsigned epSquare(const Board &board)
  {
    Square ep = board.enpassantSq();
    return (ep == Square::NO_SQ) ? 0 : static_cast<unsigned>(ep.index());
  }

  // Root-only probe: if the position is already within the tablebases (and
  // has no castling rights, which TBs don't cover), fetch the exact result
  // and play it immediately — no need to spend any search time on a
  // position where the outcome is already known perfectly.
  //
  // Returns true if it already printed "info"/"bestmove" and the caller
  // (iterativeDeepening) should return without running any search at all.
  // Returns false if there was no usable TB hit, in which case the normal
  // iterative-deepening loop proceeds exactly as if this function didn't
  // exist.
  bool tryRootProbe(search::SearchContext &ctx)
  {
    if (!ctx.tbLoaded)
      return false;
    if (ctx.board.occ().count() > static_cast<int>(TB_LARGEST))
      return false;
    if (castlingMask(ctx.board) != 0)
      return false;

    unsigned result = tb_probe_root(
        ctx.board.us(Color::WHITE).getBits(),
        ctx.board.us(Color::BLACK).getBits(),
        ctx.board.pieces(PieceType::KING).getBits(),
        ctx.board.pieces(PieceType::QUEEN).getBits(),
        ctx.board.pieces(PieceType::ROOK).getBits(),
        ctx.board.pieces(PieceType::BISHOP).getBits(),
        ctx.board.pieces(PieceType::KNIGHT).getBits(),
        ctx.board.pieces(PieceType::PAWN).getBits(),
        static_cast<unsigned>(ctx.board.halfMoveClock()),
        0, // castling — 0 enforced above
        epSquare(ctx.board),
        ctx.board.sideToMove() == Color::WHITE,
        nullptr // nullptr = return single best move, no ranked list
    );

    if (result == TB_RESULT_FAILED)
      return false;

    unsigned wdl = TB_GET_WDL(result);
    unsigned from = TB_GET_FROM(result);
    unsigned to = TB_GET_TO(result);
    unsigned promo = TB_GET_PROMOTES(result);

    PieceType pt = PieceType::QUEEN;
    if (promo == TB_PROMOTES_ROOK)
      pt = PieceType::ROOK;
    else if (promo == TB_PROMOTES_BISHOP)
      pt = PieceType::BISHOP;
    else if (promo == TB_PROMOTES_KNIGHT)
      pt = PieceType::KNIGHT;

    // Fathom gives us from/to/promo, not the engine's own Move encoding —
    // resolve it against the legal move list so castling/promotion end up
    // represented exactly the way the rest of the engine expects.
    Movelist legal;
    movegen::legalmoves(legal, ctx.board);
    Move tbMove = Move::NO_MOVE;

    for (const auto &m : legal)
    {
      if (m.from().index() != static_cast<int>(from))
        continue;
      if (m.to().index() != static_cast<int>(to))
        continue;

      if (promo == TB_PROMOTES_NONE)
      {
        tbMove = m;
        break;
      }
      if (m.typeOf() != Move::PROMOTION)
        continue;

      if (m.promotionType() == pt)
      {
        tbMove = m;
        break;
      }
    }

    if (tbMove == Move::NO_MOVE)
      return false; // shouldn't happen, but fall back to a real search rather than send nothing

    std::string scoreStr;
    if (wdl == TB_WIN)
      scoreStr = "cp 20000";
    else if (wdl == TB_LOSS)
      scoreStr = "cp -20000";
    else
      scoreStr = "cp 0";

    std::cout << "info depth 0 score " << scoreStr
              << " nodes 0 time 0 tbhits 1"
              << " pv " << moveutils::moveToUci(tbMove)
              << std::endl;
    std::cout << "bestmove " << moveutils::moveToUci(tbMove)
              << std::endl;
    return true;
  }

  // Mid-search probe, called from every non-root alphaBeta node once TBs
  // are loaded. Conditions for even attempting a probe:
  //   - piece count within the largest table we actually have loaded
  //   - no castling rights for either side (again, TBs don't model this)
  //   - the 50-move clock reads exactly 0 — Fathom's WDL tables assume a
  //     fresh clock; a non-zero clock could flip a "win" into a drawn-by-
  //     fifty-move-rule result that WDL alone can't express, so we simply
  //     don't trust the probe in that case and fall through to a normal
  //     search instead.
  //
  // On a successful probe: always records the result in the TT (so
  // sibling/later nodes reaching this same position skip re-probing), then
  // folds the WDL result into the alpha/beta window the same way a TT hit
  // would — EXACT resolves the node outright, LOWERBOUND/UPPERBOUND only
  // cut off if they already satisfy beta/alpha, otherwise a LOWERBOUND
  // still gets to tighten alpha before the normal move loop continues.
  //
  // Returns true if the caller should return `outScore` immediately.
  // Returns false if the probe didn't apply or wasn't decisive enough to
  // stop the search — `alpha` may have been raised in place either way,
  // and the caller should keep using it.
  bool probeMidSearch(search::SearchContext &ctx, int depth, int ply, int &alpha, int &beta, int &outScore)
  {
    int pieces = ctx.board.occ().count();
    if (pieces > static_cast<int>(TB_LARGEST))
      return false;
    if (castlingMask(ctx.board) != 0)
      return false;
    if (ctx.board.halfMoveClock() != 0)
      return false;

    unsigned result = tb_probe_wdl(
        ctx.board.us(Color::WHITE).getBits(),
        ctx.board.us(Color::BLACK).getBits(),
        ctx.board.pieces(PieceType::KING).getBits(),
        ctx.board.pieces(PieceType::QUEEN).getBits(),
        ctx.board.pieces(PieceType::ROOK).getBits(),
        ctx.board.pieces(PieceType::BISHOP).getBits(),
        ctx.board.pieces(PieceType::KNIGHT).getBits(),
        ctx.board.pieces(PieceType::PAWN).getBits(),
        0, // fifty-move clock — 0 enforced above
        0, // castling         — 0 enforced above
        epSquare(ctx.board),
        ctx.board.sideToMove() == Color::WHITE);

    if (result == TB_RESULT_FAILED)
      return false;

    ++ctx.tbHits;

    int tbScore;
    TTFlag flag;

    switch (result)
    {
    case TB_WIN:
      tbScore = search::MATE_SCORE - search::MAX_PLY - ply;
      flag = TTFlag::LOWERBOUND;
      break;
    case TB_LOSS:
      tbScore = -search::MATE_SCORE + search::MAX_PLY + ply;
      flag = TTFlag::UPPERBOUND;
      break;
    default: // TB_DRAW / TB_CURSED_WIN / TB_BLESSED_LOSS
      tbScore = 0;
      flag = TTFlag::EXACT;
      break;
    }

    ctx.tt->store(ctx.board.hash(), depth, search::valueToTT(tbScore, ply), flag, 0);

    if (flag == TTFlag::EXACT)
    {
      outScore = tbScore;
      return true;
    }
    if (flag == TTFlag::LOWERBOUND)
    {
      if (tbScore >= beta)
      {
        outScore = tbScore;
        return true;
      }
      if (tbScore > alpha)
        alpha = tbScore;
      return false;
    }
    if (tbScore <= alpha)
    {
      outScore = tbScore;
      return true;
    }
    if (tbScore < beta)
      beta = tbScore;
    return false;
  }

} // namespace search::tb

// search::see — static exchange evaluation.
//
// Used only for move ordering (scoring captures) and for the qsearch
// see<0 capture filter. Not a full/exact SEE across every edge case
// (there's no need for one here — used only to *order* captures, not
// to decide legality) but implements the real minimax-over-a-capture-
// sequence algorithm rather than a material-difference approximation.

namespace search::see
{

  // Finds the cheapest piece of `side` that attacks `to`, restricted to
  // the (already partially exchanged) occupancy `occ`. Sliding piece
  // attacks are recomputed against `occ` each call, so previously-removed
  // attackers correctly stop blocking behind them (e.g. a rook behind a
  // captured pawn becomes an attacker once the pawn is gone).
  bool leastValuableAttacker(const Board &board, Color side, Square to,
                             const Bitboard &occ, Square &outSq, PieceType &outType)
  {
    static constexpr PieceType::underlying order[] = {
        PieceType::underlying::PAWN, PieceType::underlying::KNIGHT,
        PieceType::underlying::BISHOP, PieceType::underlying::ROOK,
        PieceType::underlying::QUEEN, PieceType::underlying::KING};

    for (auto u : order)
    {
      Bitboard attackers;
      switch (u)
      {
      case PieceType::underlying::PAWN:
        attackers = attacks::pawn(~side, to) & board.pieces(PieceType::PAWN, side) & occ;
        break;
      case PieceType::underlying::KNIGHT:
        attackers = attacks::knight(to) & board.pieces(PieceType::KNIGHT, side) & occ;
        break;
      case PieceType::underlying::BISHOP:
        attackers = attacks::bishop(to, occ) & board.pieces(PieceType::BISHOP, side) & occ;
        break;
      case PieceType::underlying::ROOK:
        attackers = attacks::rook(to, occ) & board.pieces(PieceType::ROOK, side) & occ;
        break;
      case PieceType::underlying::QUEEN:
        attackers = attacks::queen(to, occ) & board.pieces(PieceType::QUEEN, side) & occ;
        break;
      case PieceType::underlying::KING:
        attackers = attacks::king(to) & board.pieces(PieceType::KING, side) & occ;
        break;
      }

      if (attackers.count() > 0)
      {
        outSq = Square(attackers.lsb());
        outType = PieceType(u);
        return true;
      }
    }
    return false;
  }

  // Plays out the full capture sequence on `to` (both sides always
  // recapturing with their cheapest attacker) and folds the resulting
  // gain list back into a single signed value: positive means the side
  // making the initial capture comes out ahead in material.
  int see(const Board &board, const Move &m)
  {
    // Castling is encoded as "king moves to its own rook's square" in
    // this move representation — never an actual capture, so it must
    // never enter the exchange logic below (which would otherwise treat
    // the rook as captured material).
    if (m.typeOf() == Move::CASTLING)
      return 0;

    Square to = m.to();
    Square from = m.from();
    Color mover = board.sideToMove();

    Piece capturedPiece = board.at(to);
    if (m.typeOf() == Move::ENPASSANT)
      capturedPiece = Piece(PieceType::PAWN, ~mover);

    if (capturedPiece == Piece::NONE)
      return 0; // not a capture, SEE not applicable

    Bitboard occ = board.occ();
    occ.clear(from.index());

    if (m.typeOf() == Move::ENPASSANT)
    {
      // The captured pawn is NOT on `to` for en passant — it sits one
      // rank behind, on the capturing pawn's starting rank. Must clear
      // it explicitly or the exchange loop below would still count it
      // as a defender of `to`.
      int epPawnIdx = to.index() + (mover == Color::WHITE ? -8 : 8);
      occ.clear(epPawnIdx);
    }

    // Value of the piece that ends up sitting on `to` after THIS move —
    // for a promoting capture, that's the promoted piece, not the pawn
    // that made the move.
    PieceType movingResultType = (m.typeOf() == Move::PROMOTION)
                                     ? m.promotionType()
                                     : board.at(from).type();

    constexpr int MAX_EXCHANGE = 32;
    int gain[MAX_EXCHANGE];
    PieceType onSquare[MAX_EXCHANGE]; // piece type currently sitting on `to`
    int d = 0;

    gain[0] = eval::materialValue(capturedPiece.type());
    onSquare[0] = movingResultType;

    Color side = ~mover; // opponent recaptures first

    while (d + 1 < MAX_EXCHANGE)
    {
      Square lvaSq;
      PieceType lvaType;
      if (!leastValuableAttacker(board, side, to, occ, lvaSq, lvaType))
        break;

      ++d;
      gain[d] = eval::materialValue(onSquare[d - 1]) - gain[d - 1];
      onSquare[d] = lvaType;

      occ.clear(lvaSq.index());
      side = ~side;
    }

    // Fold the gain list back-to-front: at each step, a side stops the
    // exchange early if continuing would be bad for them — the standard
    // minimax-over-a-list fold for SEE.
    while (d > 0)
    {
      gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
      --d;
    }

    return gain[0];
  }

} // namespace search::see

// search::ordering — move scoring/sorting, plus the killer/history/
// countermove bookkeeping that feeds it.
//
// Priority ladder (highest first): TT move -> winning/equal captures
// (by SEE) -> killers -> countermove -> history -> losing captures.
// Losing captures are ordered *below* quiet heuristics (not skipped
// outright — only quiescence does that) since a quiet move with a
// proven track record at this ply is more likely to be useful than a
// capture SEE already flags as materially bad.

namespace search::ordering
{

  int scoreMove(const Board &board, const Move &m, const Move &ttMove,
               const Move killers[2], const Move &counterMove, const int history[64][64])
  {
    if (m == ttMove)
      return 1'000'000;

    bool isCapture = (m.typeOf() == Move::ENPASSANT) ||
                     (m.typeOf() != Move::CASTLING && board.at(m.to()) != Piece::NONE);

    if (isCapture)
    {
      int s = see::see(board, m);
      return (s >= 0) ? (200'000 + s) : (1'000 + s);
    }

    if (m == killers[0])
      return 90'000;
    if (m == killers[1])
      return 80'000;

    // Countermove: the quiet move that most recently refuted the
    // opponent's last move, tracked globally rather than per-position —
    // cheaper signal than killers, but complements them well since it
    // still captures "moves that punish this specific opponent move"
    // rather than just "moves that are generally good here".
    if (counterMove != Move::NO_MOVE && m == counterMove)
      return 70'000;

    return history[m.from().index()][m.to().index()];
  }

  // Insertion sort rather than std::sort: move lists are short (rarely
  // above ~40, capped at MAX_MOVES) and already close to sorted once
  // killers/history stabilize during a game, so the O(n^2) worst case
  // never materializes in practice and the constant-factor win over
  // std::sort's overhead is real at this size.
  void orderMoves(const Board &board, Movelist &moves, const Move &ttMove,
                  const Move killers[2], const Move &counterMove, const int history[64][64])
  {
    std::array<int, search::MAX_MOVES> scores;
    size_t n = moves.size();

    for (size_t i = 0; i < n; ++i)
      scores[i] = scoreMove(board, moves[i], ttMove, killers, counterMove, history);

    for (size_t i = 1; i < n; ++i)
    {
      int j = static_cast<int>(i);
      while (j > 0 && scores[j - 1] < scores[j])
      {
        std::swap(scores[j - 1], scores[j]);
        std::swap(moves[j - 1], moves[j]);
        --j;
      }
    }
  }

  // Called on a beta cutoff caused by quiet move `m`. Three things happen:
  //   1. `m` becomes this ply's newest killer (bumping the old primary
  //      killer down to secondary).
  //   2. `m` is recorded as the countermove that refutes `prevMove`.
  //   3. `m`'s history score goes up by depth^2 (deeper cutoffs are
  //      stronger evidence the move is good, so they're weighted more).
  //
  // Every *other* quiet move tried earlier at this node — which did NOT
  // cause a cutoff, meaning it was tried and passed over in favor of `m`
  // — gets a matching history *penalty* ("history malus"). Without this,
  // history only ever goes up, so a move that looks tempting by
  // heuristic but keeps failing to actually cut off would never sink in
  // priority; the malus keeps the table honest in both directions.
  void updateOrderingHeuristics(search::SearchContext &ctx, int ply, Color us,
                                const Move &m, const Move &prevMove, int depth,
                                const std::array<Move, search::MAX_MOVES> &quietsTried,
                                int quietsTriedCount)
  {
    int safeKillerPly = std::min(ply, search::MAX_PLY - 1);
    ctx.killers[safeKillerPly][1] = ctx.killers[safeKillerPly][0];
    ctx.killers[safeKillerPly][0] = m;

    if (prevMove != Move::NO_MOVE)
      ctx.counterMoves[static_cast<int>(us)][prevMove.from().index()][prevMove.to().index()] = m;

    int &h = ctx.history[static_cast<int>(us)][m.from().index()][m.to().index()];
    h += depth * depth;
    if (h > 30'000)
      h = 30'000; // prevent overflow in long games

    for (int qi = 0; qi < quietsTriedCount; ++qi)
    {
      const Move &qm = quietsTried[qi];
      int &hq = ctx.history[static_cast<int>(us)][qm.from().index()][qm.to().index()];
      hq -= depth * depth / 2;
      if (hq < -30'000)
        hq = -30'000;
    }
  }

} // namespace search::ordering

// search::qsearch — quiescence search.
//
// Resolves tactical sequences (captures, and full check evasions when in
// check) at the leaves of the main search, so alphaBeta never has to
// evaluate a position where a piece is hanging mid-exchange. No TB probing
// here — qsearch is meant to be cheap and deep, and every node it visits
// will have already been covered by a TB probe higher up the tree if one
// applied.

namespace search::qsearch
{

  int quiescence(search::SearchContext &ctx, int alpha, int beta, int ply)
  {
    ++ctx.nodes;
    if ((ctx.nodes & 2047) == 0 && ctx.timeUp())
      ctx.stop = true;
    if (ctx.stop)
      return 0;

    // Draw detection — qsearch can run many plies deep during long forcing
    // sequences (perpetual checks, repeated captures), so without this it
    // could return a stale eval instead of correctly recognizing a draw.
    if (ctx.board.isRepetition(2) || ctx.board.isHalfMoveDraw() || ctx.board.isInsufficientMaterial())
      return 0;

    // Hard safety cap — full-evasion search on checks can recurse deeper
    // than capture-only qsearch ever could; don't run past array bounds
    // (killers/history are indexed by ply up to MAX_PLY elsewhere).
    if (ply >= search::MAX_PLY - 1)
      return eval::evaluate(ctx.board);

    alpha = std::max(alpha, -search::MATE_SCORE + ply);
    beta = std::min(beta, search::MATE_SCORE - ply);
    if (alpha >= beta)
      return alpha;

    const bool inCheck = ctx.board.inCheck();
    int standPat = 0;

    if (!inCheck)
    {
      standPat = eval::evaluate(ctx.board);
      if (standPat >= beta)
        return beta;
      if (standPat > alpha)
        alpha = standPat;

      // Delta pruning: if even winning a free queen couldn't close the
      // gap to alpha, no capture sequence from here will either — bail
      // without generating captures at all.
      if (standPat + eval::QUEEN_VALUE + 200 < alpha)
        return alpha;
    }

    // While in check we cannot trust standPat (the position might be
    // mate or force a losing line), so no stand-pat cutoff above, and we
    // must search every legal evasion rather than captures only —
    // there's no such thing as "no quiet way to respond to check" being
    // safe to ignore.
    Movelist moves;
    if (inCheck)
      movegen::legalmoves(moves, ctx.board); // full evasions, not just captures
    else
      movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, ctx.board);

    if (inCheck && moves.empty())
      return -search::MATE_SCORE + ply; // checkmate found inside qsearch

    int safeKillerPly = std::min(ply, search::MAX_PLY - 1);
    ordering::orderMoves(ctx.board, moves, Move::NO_MOVE,
                        ctx.killers[safeKillerPly], Move::NO_MOVE,
                        ctx.history[static_cast<int>(ctx.board.sideToMove())]);

    for (const auto &m : moves)
    {
      // Skip losing captures outright (unlike alphaBeta's move loop,
      // which still searches them — quiescence exists specifically to
      // avoid spending nodes on lines that a cheap material check
      // already tells us won't help either side).
      if (!inCheck && see::see(ctx.board, m) < 0)
        continue;

      ctx.board.makeMove(m);
      int score = -quiescence(ctx, -beta, -alpha, ply + 1);
      ctx.board.unmakeMove(m);

      if (ctx.stop)
        return alpha;
      if (score >= beta)
        return beta;
      if (score > alpha)
        alpha = score;
    }

    // Not in check and no capture improved on standPat: the stand-pat
    // alpha set above (or the original alpha, if standPat never beat it)
    // is the final answer for this node.
    return alpha;
  }

} // namespace search::qsearch

// search::pruning — the shallow-depth pruning and reduction techniques
// used by alphaBeta. Each function here is a pure yes/no (or a pure
// depth/reduction calculation) taking exactly the position facts it
// needs, so alphaBeta's own body can read as a checklist: "try RFP, try
// razoring, apply IIR, ..." instead of a wall of nested ifs.
//
// Null-move pruning is the one shallow-depth technique that stays inline
// in alphaBeta itself rather than living here

namespace search::pruning
{

  // Reverse futility pruning (a.k.a. static null-move pruning): if the
  // static eval already sits so far above beta that a real search would
  // almost certainly also fail high, assume that outcome and cut
  // immediately rather than paying for the search that would confirm it.
  //
  // Guarded to:
  //   - not in check (a check means tactics are in play; static eval is
  //     not trustworthy enough to prune on)
  //   - not a PV node (too risky for the line the search cares most about)
  //   - not root (root always needs a real move, never just a cutoff score)
  //   - shallow depth only (RFP_MAX_DEPTH) — the eval's margin of error
  //     relative to what a real search would find grows with depth, so
  //     this is only safe to trust a few plies out
  //   - beta not already near a mate score — mate-score arithmetic isn't
  //     meaningful to compare against a flat centipawn margin
  //
  // Margin has a base component plus a per-depth component, and widens
  // further when `improving` is false — i.e. when this node's static
  // eval did NOT improve on the eval 2 plies ago (same side to move).
  // A non-improving eval is exactly the case where a rich, high-variance
  // eval (king safety's quadratic term, initiative()) is most likely to
  // be a noisy read on the position rather than a real trend, so it gets
  // less benefit of the doubt before a no-search cutoff is allowed.
  bool tryReverseFutility(bool inCheck, bool isPV, bool isRoot, int depth,
                          int staticEval, int beta, bool improving, int &outScore)
  {
    constexpr int RFP_MARGIN_BASE = 70;
    constexpr int RFP_MARGIN_PER_DEPTH = 90;
    constexpr int RFP_NOT_IMPROVING_PENALTY = 70;
    constexpr int RFP_MAX_DEPTH = 7;

    if (inCheck || isPV || isRoot || depth > RFP_MAX_DEPTH)
      return false;
    if (std::abs(beta) >= search::MATE_SCORE - search::MAX_PLY)
      return false;

    int margin = RFP_MARGIN_BASE + RFP_MARGIN_PER_DEPTH * depth +
                 (improving ? 0 : RFP_NOT_IMPROVING_PENALTY);
    if (staticEval - margin >= beta)
    {
      outScore = staticEval - margin;
      return true;
    }
    return false;
  }

  // Razoring: at very shallow depth, if static eval is far below alpha,
  // the position is probably too far gone for a quiet move to fix —
  // drop straight into quiescence rather than spending a full node
  // confirming that. Verified against the qsearch result before
  // committing to the cutoff (rather than trusting the static margin
  // alone), which guards against the margin being wrong in sharp
  // tactical positions where a capture sequence would actually recover.
  // This is already the safest of the three shallow-depth techniques
  // (it never returns a cutoff a real qsearch call didn't confirm), so
  // its margin only needed a small widening for consistency with RFP/
  // futility below, not a structural change.
  bool tryRazoring(search::SearchContext &ctx, bool inCheck, bool isPV, bool isRoot,
                   int depth, int staticEval, int alpha, int beta, int ply, int &outScore)
  {
    constexpr int RAZOR_MARGIN_PER_DEPTH = 300;
    constexpr int RAZOR_MAX_DEPTH = 3;

    if (inCheck || isPV || isRoot || depth > RAZOR_MAX_DEPTH)
      return false;

    // Mirrors RFP's guard against pruning near mate scores, on the other
    // side of the window: alpha near a mate bound means this node is
    // already being asked "can you avoid getting mated this fast", and a
    // flat centipawn margin comparison isn't a meaningful test against
    // that — so skip razoring rather than risk a wrong answer on the
    // question that matters most at that point in the tree.
    if (std::abs(alpha) >= search::MATE_SCORE - search::MAX_PLY)
      return false;

    int margin = RAZOR_MARGIN_PER_DEPTH * depth;
    if (staticEval + margin > alpha)
      return false;

    int razorScore = qsearch::quiescence(ctx, alpha, beta, ply);
    if (razorScore <= alpha)
    {
      outScore = razorScore;
      return true;
    }
    return false;
  }

  // Internal iterative reduction: with no TT move to anchor move
  // ordering at a depth deep enough to matter, the first move tried is
  // likely to be a poor guess — shave one ply off rather than spend a
  // full-depth search verifying an unranked first move. Self-correcting:
  // once this node's result lands in the TT, later visits get a real TT
  // move and this no longer applies.
  int applyIIR(bool ttHit, bool inCheck, int depth)
  {
    if (!ttHit && !inCheck && depth >= 4)
      return depth - 1;
    return depth;
  }

  // Futility-pruning eligibility check for the move loop: if static eval
  // plus a per-depth margin still can't reach alpha, quiet moves at this
  // node are assumed hopeless and get skipped (except the first move,
  // and except a move with independent evidence it's actually good — see
  // the killer/countermove/history override at the call site in
  // alphaBeta, which this function doesn't know about by design: this
  // stays a pure "does the margin clear alpha" check). Capped to very
  // shallow depth since, like RFP, the eval's error margin against a
  // real search grows with depth. Margins widened and improving-aware
  // for the same reason as RFP above.
  bool isFutile(bool inCheck, bool isPV, int depth, int staticEval, int alpha, bool improving)
  {
    constexpr int FUTILITY_MARGIN[4] = {0, 220, 400, 650};
    constexpr int FUTILITY_NOT_IMPROVING_PENALTY = 100;
    if (inCheck || isPV || depth > 3)
      return false;
    int margin = FUTILITY_MARGIN[depth] + (improving ? 0 : FUTILITY_NOT_IMPROVING_PENALTY);
    return staticEval + margin <= alpha;
  }

  // Late Move Reduction: how many plies to shave off the search of a
  // move purely because of its position in the (already SEE/history-
  // ordered) move list. Grows smoothly with both depth and move index
  // via a log-log formula (the shape most modern engines converge on)
  // rather than a flat step, so the reduction scales up gracefully for
  // deep searches with long move lists instead of either under- or
  // over-reducing at the extremes.
  //
  // Never applied to: captures or promotions (too tactically loaded to
  // reduce blindly), the first move (already the presumed-best move by
  // ordering, always searched at full depth), moves made while already
  // in check, or moves that themselves give check (both indicate forcing
  // lines that a shallow search could easily misjudge).
  //
  // PV nodes get one ply less reduction than the raw formula gives,
  // since they're more likely to matter for the actual principal
  // variation and are worth the extra node cost.
  //
  // Also reduces one ply less for a move that's a killer at this ply
  // or the recorded countermove for the opponent's last move (both are
  // moves with independent evidence of being good here, not just move-
  // list position), and one ply more for a move with a clearly negative
  // history score (independent evidence it's been bad here before).
  int lateMoveReduction(int depth, int moveIndex, bool isPV, bool isCapture,
                        bool isPromotion, bool inCheck, bool givesCheck,
                        bool isKillerOrCounter, int historyScore)
  {
    if (depth < 2 || moveIndex < 1 || isCapture || isPromotion || inCheck || givesCheck)
      return 0;

    double lr = 0.4 + std::log(static_cast<double>(depth)) *
                           std::log(static_cast<double>(moveIndex + 1)) / 2.0;
    int reduction = static_cast<int>(lr);

    if (isPV && reduction > 0)
      --reduction;

    if (isKillerOrCounter)
      --reduction;
    else if (historyScore < -4000)
      ++reduction;

    int maxReduction = std::max(0, depth - 2);
    return std::max(0, std::min(reduction, maxReduction));
  }

} // namespace search::pruning

// search::alphabeta — the main search.
//
// Node order: time check -> draw detection -> mate-distance pruning ->
// drop to qsearch at depth 0 -> TB probe -> TT probe -> check extension
// -> static eval -> RFP -> razoring -> IIR -> null-move pruning ->
// futility-eligibility check -> move loop (PVS + LMR) -> TT store.

namespace search::alphabeta
{

  int search(search::SearchContext &ctx, int depth, int alpha, int beta, int ply,
            bool nullMoveAllowed = true, int extensions = 0, Move prevMove = Move::NO_MOVE)
  {
    // Invalidate this ply's improving-heuristic slot the instant the node
    // is entered, before any possible early return.
    ctx.staticEvalStack[ply] = NO_STATIC_EVAL;

    ++ctx.nodes;
    if ((ctx.nodes & 2047) == 0 && ctx.timeUp())
      ctx.stop = true;
    if (ctx.stop)
      return 0;

    const bool isRoot = (ply == 0);
    const bool isPV = (beta - alpha > 1);

    // Draw detection.
    if (!isRoot && (ctx.board.isRepetition(2) || ctx.board.isHalfMoveDraw() || ctx.board.isInsufficientMaterial()))
      return 0;

    // Mate-distance pruning: no line through this node can be better
    // than "mate in (ply)" for the side to move, or worse than "mated in
    // (ply)" — clamp the window to that so a deeper, farther-away mate
    // score can never be preferred over a shorter one already available
    // higher in the tree.
    alpha = std::max(alpha, -search::MATE_SCORE + ply);
    beta = std::min(beta, search::MATE_SCORE - ply);
    if (alpha >= beta)
      return alpha;

    if (depth <= 0)
      return qsearch::quiescence(ctx, alpha, beta, ply);

    // Syzygy WDL probe — root has its own probe (search::tb::tryRootProbe,
    // called once from iterativeDeepening), so this only runs for
    // non-root nodes reached during the normal search.
    if (ctx.tbLoaded && !isRoot)
    {
      int tbScore;
      if (tb::probeMidSearch(ctx, depth, ply, alpha, beta, tbScore))
        return tbScore;
    }

    // Transposition table probe.
    uint64_t key = ctx.board.hash();
    bool ttHit = false;
    TTEntry *tte = ctx.tt->probe(key, ttHit);
    Move ttMove = Move::NO_MOVE;

    if (ttHit)
    {
      ttMove = Move(tte->bestMove);
      // Root always needs an actual move played and reported, so it
      // never resolves purely from a stored score even if the depth is
      // sufficient — only non-root nodes can short-circuit here.
      if (!isRoot && tte->depth >= depth)
      {
        // valueFromTT: the raw stored score is ply-independent (see the
        // valueToTT comment near the top of this file); convert it back
        // to a score meaningful AT THIS ply BEFORE comparing it against
        // alpha/beta (which are ply-relative) — comparing the raw value
        // first and converting only the returned result would still be
        // wrong near mate scores, just in a subtler way, since the
        // comparison itself needs the ply-correct number, not just the
        // eventual return value.
        int ttScore = valueFromTT(tte->score, ply);
        if (tte->flag == TTFlag::EXACT)
          return ttScore;
        if (tte->flag == TTFlag::LOWERBOUND && ttScore >= beta)
          return ttScore;
        if (tte->flag == TTFlag::UPPERBOUND && ttScore <= alpha)
          return ttScore;
      }
    }

    const bool inCheck = ctx.board.inCheck();

    // Check extension: search one ply deeper when in check, since a
    // check forces a narrow, forcing response that's worth resolving
    // fully rather than cutting off at the "normal" depth. Capped at 16
    // to prevent a long forcing sequence of checks from extending the
    // search indefinitely.
    if (inCheck && extensions < 16)
    {
      ++depth;
      ++extensions;
    }

    // Static eval, computed once here and reused by every pruning check
    // below (RFP / razoring / futility) plus IIR's ttHit check. Only
    // meaningful when not in check — a check means the position is
    // inherently tactical, so the static eval is skipped (left at 0) and
    // every technique that depends on it is gated off by inCheck instead.
    int staticEval = inCheck ? 0 : eval::evaluate(ctx.board);

    // Improving: is this node's static eval at least as good as the
    // static eval 2 plies ago (same side to move)? Only meaningful (and
    // only ever consulted) when !inCheck, since every technique below
    // that reads it is already gated off during check. Deliberately
    // defaults to false (the more conservative, harder-to-prune setting)
    // when there isn't yet 2 plies of history on this path — see
    // NO_STATIC_EVAL's comment in search.h. This is the opposite default
    // from Stockfish's own (which defaults to true via a -infinity
    // sentinel); the choice here is deliberate given the explicit
    // priority on not losing good moves over raw node-count efficiency.
    bool improving = false;
    if (!inCheck)
    {
      ctx.staticEvalStack[ply] = staticEval;
      if (ply >= 2 && ctx.staticEvalStack[ply - 2] != NO_STATIC_EVAL)
        improving = staticEval >= ctx.staticEvalStack[ply - 2];
    }

    {
      int rfpScore;
      if (pruning::tryReverseFutility(inCheck, isPV, isRoot, depth, staticEval, beta, improving, rfpScore))
        return rfpScore;
    }

    {
      int razorScore;
      if (pruning::tryRazoring(ctx, inCheck, isPV, isRoot, depth, staticEval, alpha, beta, ply, razorScore))
        return razorScore;
    }

    depth = pruning::applyIIR(ttHit, inCheck, depth);

    // Null-move pruning: skip our move entirely (pass) and see if the
    // opponent still can't beat beta even with a free tempo — if so, a
    // real move would do at least as well, so cut immediately. Kept
    // inline (rather than in search::pruning) because it recursively
    // calls this very function.
    //
    // Skipped when:
    //   - in check (a "null move" while in check isn't a legal chess
    //     position and would misrepresent it to the recursive call)
    //   - at a PV node (too risky for the line the search cares most about)
    //   - nullMoveAllowed is false (guards against two null moves in a row,
    //     which would search the position against itself)
    //   - depth < 3 (not enough depth left for the reduced search below
    //     to mean anything)
    //   - root (root must always produce a real move)
    //   - staticEval < beta: the position doesn't even statically look
    //     good enough to deserve the assumption "passing the move is
    //     still fine" — trying NMP anyway from here is exactly the case
    //     most likely to produce a false cutoff, so it's not attempted at
    //     all rather than being attempted and only caught by verification.
    //   - we have no non-pawn, non-king material: this is the classic
    //     zugzwang guard — in pure pawn (or bare-king) endings, "passing"
    //     can look artificially strong precisely because every real move
    //     might be forced to weaken the position, so null-move's whole
    //     assumption (a free tempo can only help) breaks down exactly here
    if (!inCheck && !isPV && nullMoveAllowed && depth >= 3 && !isRoot && staticEval >= beta)
    {
      Color us = ctx.board.sideToMove();
      Bitboard nonPawnPieces = ctx.board.us(us) & ~ctx.board.pieces(PieceType::PAWN, us) & ~ctx.board.pieces(PieceType::KING, us);

      if (nonPawnPieces.count() >= 1)
      {
        // Deeper searches, and positions where static eval clears beta by
        // a wide margin (i.e. the safest cases to trust), can afford a
        // bigger reduction.
        int R = (depth >= 6 ? 3 : 2) + (staticEval - beta >= 200 ? 1 : 0);
        ctx.board.makeNullMove();
        int nullScore = -search(ctx, depth - 1 - R, -beta, -beta + 1, ply + 1, false, extensions);
        ctx.board.unmakeNullMove();
        if (ctx.stop)
          return 0;

        if (nullScore >= beta)
        {
          // Verification search: the material-based zugzwang guard above
          // only catches the extreme case (no non-pawn material at all).
          // It says nothing about positions that still have plenty of
          // pieces but happen to be quiet-zugzwang-like right now — a
          // mate threat the opponent can't get to on the tempo we handed
          // them, a fortress-like setup, etc. A false null-move cutoff
          // near the root of a deep search discards an entire subtree
          // that a shallow search would never have trusted enough to
          // discard outright, so the deeper the remaining search, the
          // more that mistake costs. Below this depth threshold, trust
          // the cutoff outright — the cost of an occasional wrong ~2-3
          // ply cut is small. At or above it, spend one shallow real
          // search (real move, null move disabled to prevent a repeated
          // false pass) to confirm the position truly holds up to a real
          // reply before committing to the cutoff.
          constexpr int NMP_VERIFICATION_MIN_DEPTH = 8;
          if (depth < NMP_VERIFICATION_MIN_DEPTH)
            return beta;

          int verifyScore = search(ctx, depth - R, alpha, beta, ply, false, extensions, prevMove);
          if (ctx.stop)
            return 0;
          if (verifyScore >= beta)
            return beta;
          // Verification disagreed — fall through to the normal move
          // loop below rather than trusting the null-move result.
        }
      }
    }

    bool canFutilityPrune = pruning::isFutile(inCheck, isPV, depth, staticEval, alpha, improving);

    // Generate and order moves.
    Movelist moves;
    movegen::legalmoves(moves, ctx.board);

    if (moves.empty())
      return inCheck ? (-search::MATE_SCORE + ply) : 0; // checkmate or stalemate

    int safeKillerPly = std::min(ply, search::MAX_PLY - 1);
    Color us = ctx.board.sideToMove(); // capture BEFORE any makeMove

    // Countermove lookup: indexed by the side about to move (us) and the
    // from/to of the opponent's move that led to this node.
    Move counterMove = Move::NO_MOVE;
    if (prevMove != Move::NO_MOVE)
      counterMove = ctx.counterMoves[static_cast<int>(us)][prevMove.from().index()][prevMove.to().index()];

    ordering::orderMoves(ctx.board, moves, ttMove,
                        ctx.killers[safeKillerPly], counterMove,
                        ctx.history[static_cast<int>(us)]);

    int origAlpha = alpha;
    Move bestMove = moves[0];
    int bestScore = -search::INF;

    // Tracks quiet moves tried at this node so far that did NOT cause a
    // cutoff, so that if a later move DOES cut off, updateOrderingHeuristics
    // can apply the history malus described above to all of them.
    std::array<Move, search::MAX_MOVES> quietsTried;
    int quietsTriedCount = 0;

    for (int i = 0; i < static_cast<int>(moves.size()); ++i)
    {
      const Move &m = moves[i];

      bool isCapture = (ctx.board.at(m.to()) != Piece::NONE && m.typeOf() != Move::CASTLING) || (m.typeOf() == Move::ENPASSANT);
      bool isPromotion = (m.typeOf() == Move::PROMOTION);

      // Independent evidence this quiet move is good here, separate from
      // (and not accounted for by) the static-eval-based futility margin:
      // a killer at this ply, the recorded countermove for the opponent's
      // last move, or a history score with a strong positive track
      // record. Used below both to exempt the move from the futility
      // skip and to reduce it less under LMR.
      bool isKillerOrCounter = (m == ctx.killers[safeKillerPly][0]) ||
                               (m == ctx.killers[safeKillerPly][1]) ||
                               (counterMove != Move::NO_MOVE && m == counterMove);
      int historyScore = ctx.history[static_cast<int>(us)][m.from().index()][m.to().index()];
      constexpr int FUTILITY_HISTORY_IMMUNITY = 4000;
      bool historyOverride = historyScore > FUTILITY_HISTORY_IMMUNITY;

      // The move must actually be made before we can know whether it
      // gives check — inCheck() reads the position AFTER the move.
      ctx.board.makeMove(m);
      bool givesCheck = ctx.board.inCheck();

      // Futility pruning: skip hopeless-looking quiet moves. Never
      // applied to move index 0 (the presumed-best move from ordering,
      // always fully searched), to a move that gives check — a checking
      // move is forcing, and static eval has no way to know it might be
      // leading into a mating attack the material-based margin never
      // priced in (mirrors the same givesCheck exclusion
      // lateMoveReduction() already applies, for the same reason) — or,
      // as of this revision, to a killer/countermove/strong-history move.
      // Those three signals are exactly the case a flat centipawn margin
      // is most likely to misjudge: a positional idea the static eval
      // doesn't price in but that has already proven itself at this node
      // or against this exact reply. Skipping those anyway is the single
      // most direct way a real good move could "slip through" a shallow
      // node's pruning, so they now always get a real (if possibly still
      // LMR-reduced) search instead.
      if (canFutilityPrune && i > 0 && !isCapture && !isPromotion && !givesCheck &&
          !isKillerOrCounter && !historyOverride)
      {
        ctx.board.unmakeMove(m);
        continue;
      }

      int score;
      if (i == 0)
      {
        // First (and presumed best) move: full-window search. Both
        // `nullMoveAllowed=true` and `extensions` are passed explicitly
        // here — an earlier version of this code passed `extensions`
        // positionally into the nullMoveAllowed slot, silently resetting
        // the check-extension counter for the rest of the subtree. Kept
        // as an explicit named pair of arguments so that mistake can't
        // silently reoccur.
        score = -search(ctx, depth - 1, -beta, -alpha, ply + 1, true, extensions, m);
      }
      else
      {
        // Zero-window (PVS) search, at a depth possibly reduced by LMR.
        int reduction = pruning::lateMoveReduction(depth, static_cast<int>(i), isPV, isCapture,
                                                   isPromotion, inCheck, givesCheck,
                                                   isKillerOrCounter, historyScore);
        score = -search(ctx, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, true, extensions, m);

        // Re-search at full depth and full window if the reduced/narrow
        // search beat alpha — either LMR under-reduced a move that's
        // actually good, or (for PV nodes) the zero-window result
        // surprised us and needs a real window to get an exact score.
        if (score > alpha && (reduction > 0 || isPV))
          score = -search(ctx, depth - 1, -beta, -alpha, ply + 1, true, extensions, m);
      }

      ctx.board.unmakeMove(m);

      if (ctx.stop)
        return bestScore == -search::INF ? alpha : bestScore;

      if (score > bestScore)
      {
        bestScore = score;
        bestMove = m;
      }
      if (score > alpha)
        alpha = score;

      if (alpha >= beta)
      {
        if (!isCapture && !isPromotion)
          ordering::updateOrderingHeuristics(ctx, ply, us, m, prevMove, depth, quietsTried, quietsTriedCount);
        break;
      }

      // No cutoff this move: if it was quiet, remember it for the malus
      // pass above in case a later move causes a cutoff.
      if (!isCapture && !isPromotion && quietsTriedCount < static_cast<int>(search::MAX_MOVES))
        quietsTried[quietsTriedCount++] = m;
    }

    TTFlag flag = (bestScore <= origAlpha) ? TTFlag::UPPERBOUND
                  : (bestScore >= beta)    ? TTFlag::LOWERBOUND
                                           : TTFlag::EXACT;
    ctx.tt->store(key, depth, valueToTT(bestScore, ply), flag, bestMove.move());

    return bestScore;
  }

} // namespace search::alphabeta

// search::iterativeDeepening — the public entry point called by uci.cpp's
// "go" handler. Tries a root TB probe first (an instant, exact answer
// when available); otherwise runs iterative deepening with aspiration
// windows, printing a UCI "info" line per completed depth and a final
// "bestmove" line once the loop ends.

namespace search
{

  void iterativeDeepening(SearchContext &ctx)
  {
    ctx.startTime = std::chrono::steady_clock::now();

    const TimeBudget timeBudget = computeTimeBudget(
        ctx.limits, ctx.board.sideToMove(), static_cast<int>(ctx.board.fullMoveNumber()));
    ctx.allocatedMs = timeBudget.maximumMs; // hard ceiling; timeUp() enforces this
    const int64_t optimalMs = timeBudget.optimalMs;
    const int64_t maximumMs = timeBudget.maximumMs;

    ctx.tt->newSearch();

    // SearchContext persists across multiple "go" commands within a game
    // (uci.cpp constructs it once), so the improving heuristic's per-ply
    // static-eval stack needs an explicit reset here every time — the
    // constructor's initialization in search.h only ever runs once.
    std::fill(std::begin(ctx.staticEvalStack), std::end(ctx.staticEvalStack), NO_STATIC_EVAL);

    // Syzygy root probe: if the position is already exactly known, play
    // it immediately and skip searching altogether.
    if (ctx.tbLoaded && tb::tryRootProbe(ctx))
      return;

    Move bestMove = Move::NO_MOVE;

    // Safety net: always have a legal move ready, even if depth 1 never
    // finishes (e.g. movetime expires mid-search). Prevents "bestmove
    // 0000" being sent, which GUIs treat as an illegal/forfeit move.
    {
      Movelist rootMoves;
      movegen::legalmoves(rootMoves, ctx.board);
      if (!rootMoves.empty())
        bestMove = rootMoves[0];
    }

    int prevScore = 0;
    Move stableBestMove = bestMove; // last-settled root move, for instability detection below
    int maxDepth = ctx.limits.depth > 0 ? ctx.limits.depth : MAX_PLY - 1;

    for (int depth = 1; depth <= maxDepth; ++depth)
    {
      int score;

      if (depth <= 4)
      {
        // Full window for early depths — too little information yet
        // from previous iterations to aspire a narrow window around.
        score = alphabeta::search(ctx, depth, -INF, INF, 0);
      }
      else
      {
        // Aspiration window: start narrow around the previous
        // iteration's score, and double the window on either side each
        // time the search fails outside it, until a search lands inside
        // the window (and is therefore an exact score for this depth).
        
        int delta = 40;
        int alpha = prevScore - delta;
        int beta = prevScore + delta;

        while (true)
        {
          score = alphabeta::search(ctx, depth, alpha, beta, 0);
          if (ctx.stop)
            break;
          if (score <= alpha)
          {
            alpha -= delta;
            delta *= 2;
          }
          else if (score >= beta)
          {
            beta += delta;
            delta *= 2;
          }
          else
            break;
        }
      }

      if (ctx.stop && depth > 1)
        break;

      // Retrieve the best move from the TT rather than tracking it
      // locally — more reliable, since it reflects whatever the deepest
      // completed search actually settled on for the root position.
      bool found = false;
      TTEntry *tte = ctx.tt->probe(ctx.board.hash(), found);
      if (found && Move(tte->bestMove) != Move::NO_MOVE)
        bestMove = Move(tte->bestMove);

      // Instability signals, compared against the PREVIOUS iteration's
      // settled values (captured before either gets overwritten below).
      // Only trusted from depth > 4 onward — the same point aspiration
      // windows kick in above, since prevScore isn't a meaningful
      // reference before that (depths 1-4 always use a full window).
      bool bestMoveChanged = depth > 4 && bestMove != stableBestMove;
      bool scoreDropped = depth > 4 && score < prevScore - 50;
      stableBestMove = bestMove;
      prevScore = score;

      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - ctx.startTime)
                         .count();

      std::string scoreStr;
      if (score >= MATE_SCORE - MAX_PLY)
        scoreStr = "mate " + std::to_string((MATE_SCORE - score + 1) / 2);
      else if (score <= -MATE_SCORE + MAX_PLY)
        scoreStr = "mate -" + std::to_string((MATE_SCORE + score + 1) / 2);
      else
        scoreStr = "cp " + std::to_string(score);

      std::cout << "info depth " << depth
                << " score " << scoreStr
                << " nodes " << ctx.nodes
                << " tbhits " << ctx.tbHits
                << " time " << elapsed
                << " pv " << moveutils::moveToUci(bestMove)
                << std::endl;

      if (ctx.timeUp())
        break;

      // Soft stop: even with time left before the hard ceiling above,
      // don't start another iteration (roughly 2-3x the nodes of the one
      // that just finished) once we've already spent our planned share
      // for this move — UNLESS the root looks unstable (best move just
      // changed, or the score just fell sharply), in which case we lean
      // on the extra headroom in maximumMs instead of cutting off a
      // search that's still actively changing its mind. maximumMs == 0
      // means no time control was actually given (fixed depth / nodes /
      // infinite search), so this check is skipped entirely in that case
      // — timeUp() above already handles those correctly on its own.
      if (!ctx.limits.infinite && maximumMs > 0)
      {
        int64_t softLimit = optimalMs;
        if (bestMoveChanged || scoreDropped)
          softLimit = std::min<int64_t>(maximumMs, optimalMs * 2);

        if (elapsed >= softLimit)
          break;
      }
    }

    std::cout << "bestmove " << moveutils::moveToUci(bestMove) << std::endl;
  }

} // namespace search