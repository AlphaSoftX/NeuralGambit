#pragma once
#include "../chess.hpp"

// Handcrafted Evaluation
//
// Returns a score in centipawns from the side-to-move's perspective.
// Positive = good for side to move, negative = bad.

namespace eval
{

  constexpr int PAWN_VALUE = 100;
  constexpr int KNIGHT_VALUE = 320;
  constexpr int BISHOP_VALUE = 330;
  constexpr int ROOK_VALUE = 500;
  constexpr int QUEEN_VALUE = 900;
  constexpr int KING_VALUE = 0; // king safety handled separately, not material

  int materialValue(chess::PieceType pt);

  // Full static evaluation of the position, from side-to-move perspective.
  int evaluate(const chess::Board &board);

  // Tapered eval helper: returns a 0..256 game-phase value (256 = full opening
  // material, 0 = bare kings) used to blend midgame/endgame piece-square tables.
  int gamePhase(const chess::Board &board);

} // namespace eval
