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
