#include "search.h"
#include "../eval/eval.h"
#include "../moveutils.h"
#include "../fathom/tbprobe.h"
#include <algorithm>
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
      return limits.movetime;

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

    // Move scoring for ordering.
    // Priority: TT move → winning captures (MVV-LVA) → en passant → killers → history.

    int scoreMove(const Board &board, const Move &m, const Move &ttMove,
                  const Move killers[2], const int history[64][64])
    {
      if (m == ttMove)
        return 1'000'000;

      if (m.typeOf() == Move::ENPASSANT)
        return 100'000 + eval::PAWN_VALUE * 10 - eval::PAWN_VALUE;

      Piece captured = (m.typeOf() != Move::CASTLING) ? board.at(m.to()) : Piece::NONE;
      if (captured != Piece::NONE)
      {
        int victimVal = eval::materialValue(captured.type());
        int attackerVal = eval::materialValue(board.at(m.from()).type());
        return 100'000 + victimVal * 10 - attackerVal;
      }

      if (m == killers[0])
        return 90'000;
      if (m == killers[1])
        return 80'000;

      return history[m.from().index()][m.to().index()];
    }

    void orderMoves(const Board &board, Movelist &moves, const Move &ttMove,
                    const Move killers[2], const int history[64][64])
    {
      std::vector<int> scores(moves.size());
      for (size_t i = 0; i < moves.size(); ++i)
        scores[i] = scoreMove(board, moves[i], ttMove, killers, history);

      for (size_t i = 1; i < moves.size(); ++i)
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

      alpha = std::max(alpha, -MATE_SCORE + ply);
      beta = std::min(beta, MATE_SCORE - ply);
      if (alpha >= beta)
        return alpha;

      int standPat = eval::evaluate(ctx.board);
      if (standPat >= beta)
        return beta;
      if (standPat > alpha)
        alpha = standPat;

      // Delta pruning: if even a free queen can't help, bail early.
      if (standPat + eval::QUEEN_VALUE + 200 < alpha)
        return alpha;

      Movelist moves;
      movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, ctx.board);

      int safeKillerPly = std::min(ply, MAX_PLY - 1);
      orderMoves(ctx.board, moves, Move::NO_MOVE,
                 ctx.killers[safeKillerPly],
                 ctx.history[static_cast<int>(ctx.board.sideToMove())]);

      for (const auto &m : moves)
      {
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
                  bool nullMoveAllowed = true, int extensions = 0)
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
      if (inCheck && extensions < 16){
        ++depth;
        ++extensions;
      }

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
        int staticEval = eval::evaluate(ctx.board);
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

      orderMoves(ctx.board, moves, ttMove,
                 ctx.killers[safeKillerPly],
                 ctx.history[static_cast<int>(us)]);

      int origAlpha = alpha;
      Move bestMove = moves[0];
      int bestScore = -INF;

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
          score = -alphaBeta(ctx, depth - 1, -beta, -alpha, ply + 1, extensions);
        }
        else
        {
          // LMR: reduce depth for later quiet moves.
          int reduction = 0;
          if (depth >= 3 && i >= 4 && !isCapture && !isPromotion && !inCheck && !givesCheck)
          {
            reduction = 1;
            if (i >= 8)
              ++reduction;
          }

          // Zero-window (PVS) search at possibly reduced depth.
          score = -alphaBeta(ctx, depth - 1 - reduction, -alpha - 1, -alpha,
                             ply + 1, extensions);

          // Re-search at full depth if LMR failed high or PV node surprised us.
          if (score > alpha && (reduction > 0 || isPV))
            score = -alphaBeta(ctx, depth - 1, -beta, -alpha, ply + 1, extensions);
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
          // Beta cutoff — update killer and history for quiet moves.
          if (!isCapture && !isPromotion)
          {
            ctx.killers[safeKillerPly][1] = ctx.killers[safeKillerPly][0];
            ctx.killers[safeKillerPly][0] = m;
            int &h = ctx.history[static_cast<int>(us)]
                                [m.from().index()][m.to().index()];
            h += depth * depth;
            if (h > 30'000)
              h = 30'000; // prevent overflow in long games
          }
          break;
        }
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