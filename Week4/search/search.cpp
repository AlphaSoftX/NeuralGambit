#include "search.h"
#include "../eval/eval.h"
#include "../moveutils.h"
#include "../fathom/tbprobe.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace chess;

namespace search
{

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

  int64_t allocateTime(const Limits &limits, Color us)
  {
    if (limits.movetime > 0)
      return limits.movetime - 250;

    int64_t myTime = (us == Color::WHITE) ? limits.wtime : limits.btime;
    int64_t myInc = (us == Color::WHITE) ? limits.winc : limits.binc;

    if (myTime <= 0)
      return 0;

    int movesToGo = limits.movestogo > 0 ? limits.movestogo : 30;

    int64_t budget = myTime / movesToGo + myInc;
    int64_t maxUse = (myTime * 85) / 100;
    budget = std::min(budget, maxUse);
    budget = std::max<int64_t>(budget, 20);
    budget = std::min<int64_t>(budget, myTime - 50 > 0 ? myTime - 50 : budget);

    return budget;
  }

  namespace
  {

    constexpr size_t MAX_MOVES = 218;

    // Fathom helper: build castling / EP arguments from the current board.
    // Extracted so both root probe and mid-search probe share the same logic.

    unsigned tbCastling(const Board &board)
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

    unsigned tbEp(const Board &board)
    {
      Square ep = board.enpassantSq();
      return (ep == Square::NO_SQ) ? 0 : static_cast<unsigned>(ep.index());
    }

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
        default: // KING
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

    int see(const Board &board, const Move &m)
    {
      // Castling is encoded as "king moves to its own rook's square" — never
      // a real capture, must not enter the exchange logic.
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
        // The captured pawn is NOT on `to` for en passant — it's one rank
        // behind, on the capturing pawn's starting rank.
        int epPawnIdx = to.index() + (mover == Color::WHITE ? -8 : 8);
        occ.clear(epPawnIdx);
      }

      // Value of the piece that ends up sitting on `to` after THIS move —
      // for a promoting capture, that's the promoted piece, not the pawn.
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

      // Fold the gain list back-to-front: each side stops the exchange early
      // if continuing would be bad for them (standard minimax-over-list fold).
      while (d > 0)
      {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
      }

      return gain[0];
    }

    // Move scoring for ordering.
    // Priority: TT move → winning captures (SEE) → killers → countermove → history.

    int scoreMove(const Board &board, const Move &m, const Move &ttMove,
                  const Move killers[2], const Move &counterMove, const int history[64][64])
    {
      if (m == ttMove)
        return 1'000'000;

      bool isCapture = (m.typeOf() == Move::ENPASSANT) ||
                       (m.typeOf() != Move::CASTLING && board.at(m.to()) != Piece::NONE);

      if (isCapture)
      {
        int s = see(board, m);
        // Good/equal captures: ordered above killers. Losing captures: ordered
        // below killers/history so quiet moves with proven track records get
        // tried first — but still searched eventually, never skipped here
        // (only quiescence skips them outright).
        return (s >= 0) ? (200'000 + s) : (1'000 + s);
      }

      if (m == killers[0])
        return 90'000;
      if (m == killers[1])
        return 80'000;

      // Countermove: the quiet move that most recently refuted the
      // opponent's last move at any node, regardless of position. Cheaper
      // signal than killers (not position-specific) but complements them
      // well — slotted just below killers, above plain history.
      if (counterMove != Move::NO_MOVE && m == counterMove)
        return 70'000;

      return history[m.from().index()][m.to().index()];
    }

    void orderMoves(const Board &board, Movelist &moves, const Move &ttMove,
                    const Move killers[2], const Move &counterMove, const int history[64][64])
    {
      std::array<int, MAX_MOVES> scores;
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

    // Quiescence search

    int quiescence(SearchContext &ctx, int alpha, int beta, int ply)
    {
      ++ctx.nodes;
      if ((ctx.nodes & 2047) == 0 && ctx.timeUp())
        ctx.stop = true;
      if (ctx.stop)
        return 0;

      // Draw detection — qsearch can run many plies deep during long forcing
      // sequences (perpetual checks, repeated captures), so without this it
      // can return a stale eval instead of correctly recognizing a draw.
      if (ctx.board.isRepetition(2) || ctx.board.isHalfMoveDraw() || ctx.board.isInsufficientMaterial())
        return 0;

      // Hard safety cap — full-evasion search on checks can recurse deeper
      // than capture-only qsearch ever could; don't run past array bounds.
      if (ply >= MAX_PLY - 1)
        return eval::evaluate(ctx.board);

      alpha = std::max(alpha, -MATE_SCORE + ply);
      beta = std::min(beta, MATE_SCORE - ply);
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

        // Delta pruning: if even a free queen can't help, bail early.
        if (standPat + eval::QUEEN_VALUE + 200 < alpha)
          return alpha;
      }

      // If in check, no stand-pat: we MUST find a way out, so we search every
      // legal evasion rather than assuming the static eval is trustworthy.

      Movelist moves;
      if (inCheck)
        movegen::legalmoves(moves, ctx.board); // full evasions, not just captures
      else
        movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, ctx.board);

      if (inCheck && moves.empty())
        return -MATE_SCORE + ply; // checkmate found inside qsearch

      int safeKillerPly = std::min(ply, MAX_PLY - 1);
      orderMoves(ctx.board, moves, Move::NO_MOVE,
                 ctx.killers[safeKillerPly], Move::NO_MOVE,
                 ctx.history[static_cast<int>(ctx.board.sideToMove())]);

      for (const auto &m : moves)
      {

        if (!inCheck && see(ctx.board, m) < 0)
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

      // Not in check and no captures available: stand-pat alpha already set above.
      return alpha;
    }

    // Alpha-beta with:
    //   - Draw detection (repetition + 50-move)
    //   - Mate-distance pruning
    //   - Syzygy WDL probe (mid-search)
    //   - TT probe / store
    //   - Check extension
    //   - Null-move pruning
    //   - Futility pruning
    //   - PVS (zero-window re-search)
    //   - LMR (late move reductions)

    int alphaBeta(SearchContext &ctx, int depth, int alpha, int beta, int ply,
                  bool nullMoveAllowed = true, int extensions = 0, chess::Move prevMove = chess::Move::NO_MOVE)
    {
      ++ctx.nodes;
      if ((ctx.nodes & 2047) == 0 && ctx.timeUp())
        ctx.stop = true;
      if (ctx.stop)
        return 0;

      const bool isRoot = (ply == 0);
      const bool isPV = (beta - alpha > 1);

      // Draw detection
      if (!isRoot && (ctx.board.isRepetition(2) || ctx.board.isHalfMoveDraw() || ctx.board.isInsufficientMaterial()))
        return 0;

      // Mate-distance pruning
      alpha = std::max(alpha, -MATE_SCORE + ply);
      beta = std::min(beta, MATE_SCORE - ply);
      if (alpha >= beta)
        return alpha;

      if (depth <= 0)
        return quiescence(ctx, alpha, beta, ply);

      // Syzygy WDL probe (mid-search)
      // Only probe when:
      //   - TBs are loaded
      //   - not root (root has its own probe in iterativeDeepening)
      //   - piece count within the largest TB we have
      //   - no castling rights (TBs don't model them)
      //   - 50-move clock is 0 (avoids wrong WDL when clock is non-zero)
      if (ctx.tbLoaded && !isRoot)
      {
        int pieces = ctx.board.occ().count();
        if (pieces <= static_cast<int>(TB_LARGEST) && tbCastling(ctx.board) == 0 && ctx.board.halfMoveClock() == 0)
        {
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
              0, // castling   — 0 enforced above
              tbEp(ctx.board),
              ctx.board.sideToMove() == Color::WHITE);

          if (result != TB_RESULT_FAILED)
          {
            ++ctx.tbHits;

            int tbScore;
            TTFlag flag;

            switch (result)
            {
            case TB_WIN:
              tbScore = MATE_SCORE - MAX_PLY - ply;
              flag = TTFlag::LOWERBOUND;
              break;
            case TB_LOSS:
              tbScore = -MATE_SCORE + MAX_PLY + ply;
              flag = TTFlag::UPPERBOUND;
              break;
            default: // TB_DRAW / TB_CURSED_WIN / TB_BLESSED_LOSS
              tbScore = 0;
              flag = TTFlag::EXACT;
              break;
            }

            ctx.tt->store(ctx.board.hash(), depth, tbScore, flag, 0);

            if (flag == TTFlag::EXACT)
              return tbScore;
            if (flag == TTFlag::LOWERBOUND && tbScore >= beta)
              return tbScore;
            if (flag == TTFlag::UPPERBOUND && tbScore <= alpha)
              return tbScore;
            if (flag == TTFlag::LOWERBOUND && tbScore > alpha)
              alpha = tbScore;
          }
        }
      }

      // Transposition table probe
      uint64_t key = ctx.board.hash();
      bool ttHit = false;
      TTEntry *tte = ctx.tt->probe(key, ttHit);
      Move ttMove = Move::NO_MOVE;

      if (ttHit)
      {
        ttMove = Move(tte->bestMove);
        if (!isRoot && tte->depth >= depth)
        {
          if (tte->flag == TTFlag::EXACT)
            return tte->score;
          if (tte->flag == TTFlag::LOWERBOUND && tte->score >= beta)
            return tte->score;
          if (tte->flag == TTFlag::UPPERBOUND && tte->score <= alpha)
            return tte->score;
        }
      }

      const bool inCheck = ctx.board.inCheck();

      // Check extension
      if (inCheck && extensions < 16)
      {
        ++depth;
        ++extensions;
      }

      // Static eval, computed once and reused by RFP / razoring / futility /
      // IIR below. Only meaningful when not in check.
      int staticEval = inCheck ? 0 : eval::evaluate(ctx.board);

      // Reverse futility pruning (a.k.a. static null move pruning).
      // If our static eval is already far above beta, assume a real search
      // would also fail high and cut immediately — cheap and one of the
      // highest-value low-depth pruning techniques.
      constexpr int RFP_MARGIN_PER_DEPTH = 80;
      constexpr int RFP_MAX_DEPTH = 7;
      if (!inCheck && !isPV && !isRoot && depth <= RFP_MAX_DEPTH &&
          std::abs(beta) < MATE_SCORE - MAX_PLY)
      {
        int margin = RFP_MARGIN_PER_DEPTH * depth;
        if (staticEval - margin >= beta)
          return staticEval - margin;
      }

      // Razoring: at shallow depth, if static eval is far below alpha, the
      // position is probably lost for quiet improvement — drop straight
      // into quiescence rather than spending a full node on it.
      constexpr int RAZOR_MARGIN_PER_DEPTH = 250;
      constexpr int RAZOR_MAX_DEPTH = 3;
      if (!inCheck && !isPV && !isRoot && depth <= RAZOR_MAX_DEPTH)
      {
        int margin = RAZOR_MARGIN_PER_DEPTH * depth;
        if (staticEval + margin <= alpha)
        {
          int razorScore = quiescence(ctx, alpha, beta, ply);
          if (razorScore <= alpha)
            return razorScore;
        }
      }

      // Internal iterative reduction: with no TT move to guide ordering at
      // a meaningful depth, the first move searched is likely to be poorly
      // chosen. Shave a ply off rather than spending a full-depth search on
      // an unverified first move — cheap and self-correcting since the TT
      // fills in on subsequent visits.
      if (!ttHit && !inCheck && depth >= 4)
        --depth;

      // Null-move pruning
      if (!inCheck && !isPV && nullMoveAllowed && depth >= 3 && !isRoot)
      {
        Color us = ctx.board.sideToMove();
        Bitboard nonPawnPieces = ctx.board.us(us) & ~ctx.board.pieces(PieceType::PAWN, us) & ~ctx.board.pieces(PieceType::KING, us);

        if (nonPawnPieces.count() >= 1)
        {
          int R = depth >= 6 ? 3 : 2;
          ctx.board.makeNullMove();
          int nullScore = -alphaBeta(ctx, depth - 1 - R, -beta, -beta + 1,
                                     ply + 1, false, extensions);
          ctx.board.unmakeNullMove();
          if (ctx.stop)
            return 0;
          if (nullScore >= beta)
            return beta;
        }
      }

      // Futility pruning setup
      constexpr int FUTILITY_MARGIN[4] = {0, 150, 300, 500};
      bool canFutilityPrune = false;
      if (!inCheck && !isPV && depth <= 3)
      {
        if (staticEval + FUTILITY_MARGIN[depth] <= alpha)
          canFutilityPrune = true;
      }

      // Generate and order moves
      Movelist moves;
      movegen::legalmoves(moves, ctx.board);

      if (moves.empty())
        return inCheck ? (-MATE_SCORE + ply) : 0; // checkmate or stalemate

      int safeKillerPly = std::min(ply, MAX_PLY - 1);
      Color us = ctx.board.sideToMove(); // capture BEFORE any makeMove

      // Countermove lookup: indexed by the side that's about to move (us)
      // and the from/to of the opponent's move that led to this node.
      Move counterMove = Move::NO_MOVE;
      if (prevMove != Move::NO_MOVE)
        counterMove = ctx.counterMoves[static_cast<int>(us)][prevMove.from().index()][prevMove.to().index()];

      orderMoves(ctx.board, moves, ttMove,
                 ctx.killers[safeKillerPly], counterMove,
                 ctx.history[static_cast<int>(us)]);

      int origAlpha = alpha;
      Move bestMove = moves[0];
      int bestScore = -INF;

      // Tracks quiet moves tried this node so far (that did NOT cause a
      // cutoff) so that, if a later move DOES cause a cutoff, we can apply
      // a small history malus to all of them. This sharpens move ordering
      // over time: moves that are repeatedly tried-and-failed sink in
      // priority, not just moves that succeed rising.
      std::array<Move, MAX_MOVES> quietsTried;
      int quietsTriedCount = 0;

      for (size_t i = 0; i < moves.size(); ++i)
      {
        const Move &m = moves[i];

        bool isCapture = (ctx.board.at(m.to()) != Piece::NONE && m.typeOf() != Move::CASTLING) || (m.typeOf() == Move::ENPASSANT);
        bool isPromotion = (m.typeOf() == Move::PROMOTION);

        // Futility pruning: skip hopeless quiet moves
        if (canFutilityPrune && i > 0 && !isCapture && !isPromotion)
          continue;

        ctx.board.makeMove(m);
        bool givesCheck = ctx.board.inCheck();

        int score;
        if (i == 0)
        {
          // First (and likely best) move: full-window search.
          // NOTE: previously this call passed `extensions` positionally
          // into the `nullMoveAllowed` slot (the bool parameter that
          // precedes `extensions`), silently resetting the check-extension
          // counter for the rest of this subtree. Fixed by passing both
          // explicitly.
          score = -alphaBeta(ctx, depth - 1, -beta, -alpha, ply + 1, true, extensions, m);
        }
        else
        {
          // LMR: reduce depth for later, less-promising moves. Reduction
          // grows smoothly with both depth and move index (standard
          // log-log formula, same shape used by most modern engines) rather
          // than a flat +1/+2 step — this lets us search deep move lists
          // without paying full price for moves that are usually pruned
          // anyway, while still examining every move at some depth.
          int reduction = 0;
          if (depth >= 2 && i >= 1 && !isCapture && !isPromotion && !inCheck && !givesCheck)
          {
            double lr = 0.4 + std::log(static_cast<double>(depth)) *
                                   std::log(static_cast<double>(i + 1)) / 2.0;
            reduction = static_cast<int>(lr);

            // PV nodes get reduced less aggressively — they're more likely
            // to matter for the final line.
            if (isPV && reduction > 0)
              --reduction;

            reduction = std::max(0, std::min(reduction, depth - 1));
          }

          // Zero-window (PVS) search at possibly reduced depth.
          score = -alphaBeta(ctx, depth - 1 - reduction, -alpha - 1, -alpha,
                             ply + 1, true, extensions, m);

          // Re-search at full depth if LMR failed high or PV node surprised us.
          if (score > alpha && (reduction > 0 || isPV))
            score = -alphaBeta(ctx, depth - 1, -beta, -alpha, ply + 1, true, extensions, m);
        }

        ctx.board.unmakeMove(m);

        if (ctx.stop)
          return bestScore == -INF ? alpha : bestScore;

        if (score > bestScore)
        {
          bestScore = score;
          bestMove = m;
        }
        if (score > alpha)
          alpha = score;

        if (alpha >= beta)
        {
          // Beta cutoff — update killer, countermove, and history for
          // quiet moves.
          if (!isCapture && !isPromotion)
          {
            ctx.killers[safeKillerPly][1] = ctx.killers[safeKillerPly][0];
            ctx.killers[safeKillerPly][0] = m;

            if (prevMove != Move::NO_MOVE)
              ctx.counterMoves[static_cast<int>(us)][prevMove.from().index()][prevMove.to().index()] = m;

            int &h = ctx.history[static_cast<int>(us)]
                                [m.from().index()][m.to().index()];
            h += depth * depth;
            if (h > 30'000)
              h = 30'000; // prevent overflow in long games

            // History malus: every quiet move tried earlier at this node
            // that did NOT cause a cutoff gets a small penalty, since it
            // was searched and rejected in favor of this one. Keeps the
            // history table honest about moves that look tempting by
            // position but don't actually hold up.
            for (int qi = 0; qi < quietsTriedCount; ++qi)
            {
              const Move &qm = quietsTried[qi];
              int &hq = ctx.history[static_cast<int>(us)][qm.from().index()][qm.to().index()];
              hq -= depth * depth / 2;
              if (hq < -30'000)
                hq = -30'000;
            }
          }
          break;
        }

        // No cutoff this move: if it was quiet, remember it for the malus
        // pass above in case a later move causes a cutoff.
        if (!isCapture && !isPromotion && quietsTriedCount < static_cast<int>(MAX_MOVES))
          quietsTried[quietsTriedCount++] = m;
      }

      TTFlag flag = (bestScore <= origAlpha) ? TTFlag::UPPERBOUND
                    : (bestScore >= beta)    ? TTFlag::LOWERBOUND
                                             : TTFlag::EXACT;
      ctx.tt->store(key, depth, bestScore, flag, bestMove.move());

      return bestScore;
    }

  } // namespace

  // Iterative deepening with aspiration windows + Syzygy root probe
  void iterativeDeepening(SearchContext &ctx)
  {
    ctx.startTime = std::chrono::steady_clock::now();
    ctx.allocatedMs = allocateTime(ctx.limits, ctx.board.sideToMove());
    ctx.tt->newSearch();

    // Syzygy root probe
    // If we are already in a TB position, return the perfect TB move
    // immediately without searching.
    // Conditions: TBs loaded, piece count within largest TB, no castling.
    if (ctx.tbLoaded && ctx.board.occ().count() <= static_cast<int>(TB_LARGEST) && tbCastling(ctx.board) == 0)
    {
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
          tbEp(ctx.board),
          ctx.board.sideToMove() == Color::WHITE,
          nullptr // nullptr = return single best move, no ranked list
      );

      if (result != TB_RESULT_FAILED)
      {
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

        // Match TB from/to/promo against the legal move list.
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

        if (tbMove != Move::NO_MOVE)
        {
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
          return;
        }
      }
    }

    // Normal iterative deepening with aspiration windows
    Move bestMove = Move::NO_MOVE;

    // Safety net: always have a legal move ready, even if depth 1 never
    // finishes (e.g. movetime expires mid-search). Prevents "bestmove 0000"
    // being sent, which GUIs treat as an illegal/forfeit move.
    {
      Movelist rootMoves;
      movegen::legalmoves(rootMoves, ctx.board);
      if (!rootMoves.empty())
        bestMove = rootMoves[0];
    }

    int prevScore = 0;
    int maxDepth = ctx.limits.depth > 0 ? ctx.limits.depth : MAX_PLY - 1;

    for (int depth = 1; depth <= maxDepth; ++depth)
    {
      int score;

      if (depth <= 4)
      {
        // Full window for early depths — not enough info to aspire yet.
        score = alphaBeta(ctx, depth, -INF, INF, 0);
      }
      else
      {
        // Aspiration window: start narrow, widen on failure.
        int delta = 25;
        int alpha = prevScore - delta;
        int beta = prevScore + delta;

        while (true)
        {
          score = alphaBeta(ctx, depth, alpha, beta, 0);
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

      // Retrieve best move from TT — more reliable than local tracking.
      bool found = false;
      TTEntry *tte = ctx.tt->probe(ctx.board.hash(), found);
      if (found && Move(tte->bestMove) != Move::NO_MOVE)
        bestMove = Move(tte->bestMove);

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
    }

    std::cout << "bestmove " << moveutils::moveToUci(bestMove) << std::endl;
  }

} // namespace search