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
