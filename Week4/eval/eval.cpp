#include "eval.h"
#include "pst.h"

using namespace chess;

namespace eval
{

  int materialValue(PieceType pt)
  {
    switch (pt.internal())
    {
    case PieceType::underlying::PAWN:
      return PAWN_VALUE;
    case PieceType::underlying::KNIGHT:
      return KNIGHT_VALUE;
    case PieceType::underlying::BISHOP:
      return BISHOP_VALUE;
    case PieceType::underlying::ROOK:
      return ROOK_VALUE;
    case PieceType::underlying::QUEEN:
      return QUEEN_VALUE;
    default:
      return 0;
    }
  }

  namespace
  {

    // Phase weights per piece (standard tapered-eval scheme, max total = 24)
    constexpr int PHASE_KNIGHT = 1;
    constexpr int PHASE_BISHOP = 1;
    constexpr int PHASE_ROOK = 2;
    constexpr int PHASE_QUEEN = 4;
    constexpr int MAX_PHASE = 4 * PHASE_KNIGHT + 4 * PHASE_BISHOP + 4 * PHASE_ROOK + 2 * PHASE_QUEEN; // 24

    const pst::Table &mgTable(PieceType pt)
    {
      switch (pt.internal())
      {
      case PieceType::underlying::PAWN:
        return pst::PAWN_MG;
      case PieceType::underlying::KNIGHT:
        return pst::KNIGHT_MG;
      case PieceType::underlying::BISHOP:
        return pst::BISHOP_MG;
      case PieceType::underlying::ROOK:
        return pst::ROOK_MG;
      case PieceType::underlying::QUEEN:
        return pst::QUEEN_MG;
      default:
        return pst::KING_MG;
      }
    }

    const pst::Table &egTable(PieceType pt)
    {
      switch (pt.internal())
      {
      case PieceType::underlying::PAWN:
        return pst::PAWN_EG;
      case PieceType::underlying::KNIGHT:
        return pst::KNIGHT_EG;
      case PieceType::underlying::BISHOP:
        return pst::BISHOP_EG;
      case PieceType::underlying::ROOK:
        return pst::ROOK_EG;
      case PieceType::underlying::QUEEN:
        return pst::QUEEN_EG;
      default:
        return pst::KING_EG;
      }
    }

    // Files adjacent to/including the king's file, clamped to board edges.
    inline int clampFile(int f) { return std::max(0, std::min(7, f)); }

    // Counts how many of the 3 "shield" pawn squares directly in front of the
    // king (on king file and the two adjacent files) are occupied by own pawns.
    // Missing shield pawns = bigger danger.
    int kingShieldPenalty(const Board &board, Color kingColor)
    {
      Square kingSq = board.kingSq(kingColor);
      int kf = kingSq.index() % 8;
      int kr = kingSq.index() / 8;

      int missing = 0;
      int dir = (kingColor == Color::WHITE) ? 1 : -1;
      int shieldRank = kr + dir; // one rank in front of the king

      if (shieldRank < 0 || shieldRank > 7)
        return 0; // king on back rank edge case (shouldn't happen normally)

      for (int f = clampFile(kf - 1); f <= clampFile(kf + 1); ++f)
      {
        Square sq(shieldRank * 8 + f);
        Piece p = board.at(sq);
        bool hasOwnPawn = (p.type() == PieceType::PAWN && p.color() == kingColor);
        if (!hasOwnPawn)
          ++missing;
      }
      return missing; // 0..3
    }

    // Count squares in the king's immediate
    // surroundings that are attacked by the enemy. More attacked squares =
    // more danger. Doesn't account for defenders, but correlates well with
    // real attacking chances and costs one attack-lookup per square.
    int kingAttackZonePressure(const Board &board, Color kingColor, Bitboard occ)
    {
      Square kingSq = board.kingSq(kingColor);
      Color enemy = ~kingColor;

      Bitboard zone = attacks::king(kingSq); // the 8 squares around the king
      int attackedCount = 0;

      while (zone)
      {
        Square sq = zone.pop();
        if (board.isAttacked(sq, enemy))
          ++attackedCount;
      }
      return attackedCount; // 0..8
    }

    // --- Pawn structure -----------------------------------------------
    //
    // All of the following operate on raw pawn bitboards (one per color)
    // and are intentionally cheap: a handful of bitboard ANDs per pawn,
    // no per-square loops beyond the pawns that actually exist.

    // Passed pawn bonus, indexed by rank from the pawn's own side
    // (rank 0 = own 2nd rank ... rank 6 = about to promote). Tapered
    // mg/eg like everything else; passed pawns matter far more in the
    // endgame, hence the eg table being much steeper.
    constexpr int PASSED_PAWN_MG[7] = {0, 5, 10, 20, 35, 60, 100};
    constexpr int PASSED_PAWN_EG[7] = {0, 10, 20, 40, 70, 120, 200};

    constexpr int ISOLATED_PAWN_MG = -10;
    constexpr int ISOLATED_PAWN_EG = -20;

    constexpr int DOUBLED_PAWN_MG = -8;
    constexpr int DOUBLED_PAWN_EG = -16;

    constexpr int BACKWARD_PAWN_MG = -6;
    constexpr int BACKWARD_PAWN_EG = -10;

    constexpr int ROOK_OPEN_FILE_MG = 20;
    constexpr int ROOK_OPEN_FILE_EG = 10;
    constexpr int ROOK_SEMI_OPEN_FILE_MG = 10;
    constexpr int ROOK_SEMI_OPEN_FILE_EG = 5;

    // Precomputes, for a given file index 0..7, the bitboard mask of that
    // file plus its two neighbors (clamped at the board edge). Used for
    // isolated-pawn and passed-pawn front-span checks.
    inline Bitboard adjacentFilesMask(int file)
    {
      Bitboard mask = attacks::MASK_FILE[file];
      if (file > 0)
        mask |= attacks::MASK_FILE[file - 1];
      if (file < 7)
        mask |= attacks::MASK_FILE[file + 1];
      return mask;
    }

    // The set of squares in front of `sq` (toward promotion) on the given
    // file, used for doubled-pawn and passed-pawn detection.
    inline Bitboard fileInFront(Square sq, Color side)
    {
      int file = sq.index() % 8;
      int rank = sq.index() / 8;
      Bitboard mask = attacks::MASK_FILE[file];

      if (side == Color::WHITE)
      {
        for (int r = 0; r <= rank; ++r)
          mask &= ~attacks::MASK_RANK[r];
      }
      else
      {
        for (int r = rank; r <= 7; ++r)
          mask &= ~attacks::MASK_RANK[r];
      }
      return mask;
    }

    // The "passed pawn span": own file + adjacent files, all ranks ahead
    // of the pawn. If no enemy pawn occupies this span, the pawn is passed.
    inline Bitboard passedPawnSpan(Square sq, Color side)
    {
      int file = sq.index() % 8;
      int rank = sq.index() / 8;
      Bitboard mask = adjacentFilesMask(file);

      if (side == Color::WHITE)
      {
        for (int r = 0; r <= rank; ++r)
          mask &= ~attacks::MASK_RANK[r];
      }
      else
      {
        for (int r = rank; r <= 7; ++r)
          mask &= ~attacks::MASK_RANK[r];
      }
      return mask;
    }

    struct PawnStructureScore
    {
      int mg = 0;
      int eg = 0;
    };

    // Evaluates passed/isolated/doubled/backward pawns for one side.
    // `ownPawns`/`enemyPawns` are full-board bitboards (not masked to a file).
    PawnStructureScore evaluatePawnsForSide(Bitboard ownPawns, Bitboard enemyPawns, Color side)
    {
      PawnStructureScore s;

      Bitboard pawns = ownPawns;
      while (pawns)
      {
        Square sq = pawns.pop();
        int file = sq.index() % 8;
        int rank = sq.index() / 8;

        // Doubled: another own pawn further back on the same file behind
        // this one isn't counted twice — we simply check if there is any
        // other own pawn anywhere else on this file.
        Bitboard sameFileOthers = (ownPawns & attacks::MASK_FILE[file]);
        sameFileOthers.clear(sq.index());
        if (!sameFileOthers.empty())
        {
          s.mg += DOUBLED_PAWN_MG;
          s.eg += DOUBLED_PAWN_EG;
        }

        // Isolated: no own pawns on adjacent files at all.
        Bitboard neighborFiles;
        if (file > 0)
          neighborFiles |= attacks::MASK_FILE[file - 1];
        if (file < 7)
          neighborFiles |= attacks::MASK_FILE[file + 1];
        bool isolated = (ownPawns & neighborFiles).empty();
        if (isolated)
        {
          s.mg += ISOLATED_PAWN_MG;
          s.eg += ISOLATED_PAWN_EG;
        }

        // Passed: no enemy pawns in the passed-pawn span ahead of us.
        bool passed = (enemyPawns & passedPawnSpan(sq, side)).empty();
        if (passed)
        {
          int rankFromOwnSide = (side == Color::WHITE) ? rank : (7 - rank);
          rankFromOwnSide = std::max(0, std::min(6, rankFromOwnSide));
          s.mg += PASSED_PAWN_MG[rankFromOwnSide];
          s.eg += PASSED_PAWN_EG[rankFromOwnSide];
        }

        // Backward: not isolated, but the pawn can't be supported by a
        // friendly pawn pushing up beside it (no own pawn on an adjacent
        // file at the same rank or behind), AND its stop square is covered
        // by an enemy pawn — a standard, cheap backward-pawn proxy.
        if (!isolated && !passed)
        {
          Bitboard behindOrLevelMask;
          if (side == Color::WHITE)
          {
            for (int r = 0; r <= rank; ++r)
              behindOrLevelMask |= attacks::MASK_RANK[r];
          }
          else
          {
            for (int r = rank; r <= 7; ++r)
              behindOrLevelMask |= attacks::MASK_RANK[r];
          }
          if (file > 0)
            behindOrLevelMask &= (attacks::MASK_FILE[file - 1] | attacks::MASK_FILE[std::min(7, file + 1)]);
          else
            behindOrLevelMask &= attacks::MASK_FILE[file + 1];

          bool hasSupportPawn = !(ownPawns & behindOrLevelMask).empty();

          if (!hasSupportPawn)
          {
            int stopRank = rank + (side == Color::WHITE ? 1 : -1);
            if (stopRank >= 0 && stopRank <= 7)
            {
              Square stopSq(stopRank * 8 + file);
              Bitboard enemyPawnAttackers = attacks::pawn(side, stopSq) & enemyPawns;
              if (!enemyPawnAttackers.empty())
              {
                s.mg += BACKWARD_PAWN_MG;
                s.eg += BACKWARD_PAWN_EG;
              }
            }
          }
        }
      }

      return s;
    }

    // Rook file bonuses: open file (no pawns of either color) is worth more
    // than semi-open (no own pawns, but enemy pawns remain).
    int rookFileBonusMg(Square sq, Bitboard ownPawns, Bitboard enemyPawns)
    {
      int file = sq.index() % 8;
      Bitboard fileMask = attacks::MASK_FILE[file];
      bool ownPawnOnFile = !(ownPawns & fileMask).empty();
      bool enemyPawnOnFile = !(enemyPawns & fileMask).empty();

      if (!ownPawnOnFile && !enemyPawnOnFile)
        return ROOK_OPEN_FILE_MG;
      if (!ownPawnOnFile && enemyPawnOnFile)
        return ROOK_SEMI_OPEN_FILE_MG;
      return 0;
    }

    int rookFileBonusEg(Square sq, Bitboard ownPawns, Bitboard enemyPawns)
    {
      int file = sq.index() % 8;
      Bitboard fileMask = attacks::MASK_FILE[file];
      bool ownPawnOnFile = !(ownPawns & fileMask).empty();
      bool enemyPawnOnFile = !(enemyPawns & fileMask).empty();

      if (!ownPawnOnFile && !enemyPawnOnFile)
        return ROOK_OPEN_FILE_EG;
      if (!ownPawnOnFile && enemyPawnOnFile)
        return ROOK_SEMI_OPEN_FILE_EG;
      return 0;
    }

  } // namespace

  int gamePhase(const Board &board)
  {
    int phase = 0;
    phase += PHASE_KNIGHT * board.pieces(PieceType::KNIGHT).count();
    phase += PHASE_BISHOP * board.pieces(PieceType::BISHOP).count();
    phase += PHASE_ROOK * board.pieces(PieceType::ROOK).count();
    phase += PHASE_QUEEN * board.pieces(PieceType::QUEEN).count();
    if (phase > MAX_PHASE)
      phase = MAX_PHASE;
    // Scale to 0..256
    return (phase * 256) / MAX_PHASE;
  }

  int evaluate(const Board &board)
  {
    int mg = 0;
    int eg = 0;

    int whiteBishops = 0;
    int blackBishops = 0;

    const auto occ = board.occ();

    constexpr int KNIGHT_MOBILITY_MG = 4;
    constexpr int KNIGHT_MOBILITY_EG = 2;

    constexpr int BISHOP_MOBILITY_MG = 4;
    constexpr int BISHOP_MOBILITY_EG = 2;

    constexpr int ROOK_MOBILITY_MG = 2;
    constexpr int ROOK_MOBILITY_EG = 2;

    constexpr int QUEEN_MOBILITY_MG = 1;
    constexpr int QUEEN_MOBILITY_EG = 0;

    constexpr int BISHOP_PAIR_MG = 30;
    constexpr int BISHOP_PAIR_EG = 50;

    for (int sq = 0; sq < 64; ++sq)
    {

      Piece p = board.at(static_cast<Square>(sq));
      if (p == Piece::NONE)
        continue;

      const bool isWhite = (p.color() == Color::WHITE);
      const PieceType pt = p.type();

      const int pstIndex =
          isWhite ? sq : (sq ^ 56);

      int mgVal =
          materialValue(pt) +
          mgTable(pt)[pstIndex];

      int egVal =
          materialValue(pt) +
          egTable(pt)[pstIndex];

      // Mobility

      if (pt == PieceType::KNIGHT)
      {

        int mobility =
            (attacks::knight(static_cast<Square>(sq)) & ~board.us(p.color())).count();

        mgVal += mobility * KNIGHT_MOBILITY_MG;
        egVal += mobility * KNIGHT_MOBILITY_EG;
      }

      else if (pt == PieceType::BISHOP)
      {

        if (isWhite)
          ++whiteBishops;
        else
          ++blackBishops;

        int mobility =
            (attacks::bishop(
                 static_cast<Square>(sq),
                 occ) &
             ~board.us(p.color()))
                .count();

        mgVal += mobility * BISHOP_MOBILITY_MG;
        egVal += mobility * BISHOP_MOBILITY_EG;
      }

      else if (pt == PieceType::ROOK)
      {

        int mobility =
            (attacks::rook(
                 static_cast<Square>(sq),
                 occ) &
             ~board.us(p.color()))
                .count();

        mgVal += mobility * ROOK_MOBILITY_MG;
        egVal += mobility * ROOK_MOBILITY_EG;

        // Open / semi-open file bonus.
        Bitboard ownPawnsBB = board.pieces(PieceType::PAWN, p.color());
        Bitboard enemyPawnsBB = board.pieces(PieceType::PAWN, ~p.color());
        mgVal += rookFileBonusMg(static_cast<Square>(sq), ownPawnsBB, enemyPawnsBB);
        egVal += rookFileBonusEg(static_cast<Square>(sq), ownPawnsBB, enemyPawnsBB);
      }

      else if (pt == PieceType::QUEEN)
      {

        int mobility =
            (attacks::queen(
                 static_cast<Square>(sq),
                 occ) &
             ~board.us(p.color()))
                .count();

        mgVal += mobility * QUEEN_MOBILITY_MG;
        egVal += mobility * QUEEN_MOBILITY_EG;
      }

      if (isWhite)
      {
        mg += mgVal;
        eg += egVal;
      }
      else
      {
        mg -= mgVal;
        eg -= egVal;
      }
    }

    // Bishop pair

    if (whiteBishops >= 2)
    {
      mg += BISHOP_PAIR_MG;
      eg += BISHOP_PAIR_EG;
    }

    if (blackBishops >= 2)
    {
      mg -= BISHOP_PAIR_MG;
      eg -= BISHOP_PAIR_EG;
    }

    // Pawn structure: passed / isolated / doubled / backward pawns.
    {
      Bitboard whitePawns = board.pieces(PieceType::PAWN, Color::WHITE);
      Bitboard blackPawns = board.pieces(PieceType::PAWN, Color::BLACK);

      PawnStructureScore whiteScore = evaluatePawnsForSide(whitePawns, blackPawns, Color::WHITE);
      PawnStructureScore blackScore = evaluatePawnsForSide(blackPawns, whitePawns, Color::BLACK);

      mg += whiteScore.mg - blackScore.mg;
      eg += whiteScore.eg - blackScore.eg;
    }

    constexpr int SHIELD_PENALTY_MG = 12;
    constexpr int ATTACK_ZONE_PENALTY_MG = 8;

    int whiteShieldMissing = kingShieldPenalty(board, Color::WHITE);
    int blackShieldMissing = kingShieldPenalty(board, Color::BLACK);
    int whiteAttackZone = kingAttackZonePressure(board, Color::WHITE, occ);
    int blackAttackZone = kingAttackZonePressure(board, Color::BLACK, occ);

    int whiteKingDanger = whiteShieldMissing * SHIELD_PENALTY_MG + whiteAttackZone * ATTACK_ZONE_PENALTY_MG;
    int blackKingDanger = blackShieldMissing * SHIELD_PENALTY_MG + blackAttackZone * ATTACK_ZONE_PENALTY_MG;

    mg -= (whiteKingDanger * KING_SAFETY_MG_SCALE) / 100;
    mg += (blackKingDanger * KING_SAFETY_MG_SCALE) / 100;

    // Tapered interpolation

    const int phase = gamePhase(board);

    const int score =
        (mg * phase +
         eg * (256 - phase)) /
        256;

    return (board.sideToMove() == Color::WHITE)
               ? score
               : -score;
  }

} // namespace eval