#pragma once
#include "../chess.hpp"
#include <cstdint>

// This eval is a port of Stockfish's classical (pre-NNUE) evaluation:
// piece-square tables, material imbalance, pawn structure, king safety,
// mobility, threats, space, and endgame scale-factor knowledge, combined
// into a single centipawn score from the side-to-move's perspective.
//
// The module lives in three files:
//   - eval.h    (this file): the Score type, piece values, and the public
//                interface search.cpp calls (evaluate/materialValue/
//                QUEEN_VALUE/gamePhase — unchanged from the previous
//                eval.h, so search.cpp needs no changes).
//   - pst.h:    piece-square tables (header-only).
//   - eval.cpp: everything else (material imbalance, pawn structure, king
//                safety, mobility, threats, space, endgame scale factor,
//                and the top-level orchestrator).

namespace eval
{

  // Packed (middlegame, endgame) score
  struct Score
  {
    int mg = 0;
    int eg = 0;

    constexpr Score() = default;
    // Single-argument constructor exists only so `Score x = 0;` keeps
    // working at every call site that zero-initializes a running total —
    // grep the codebase and the only literal ever passed here is 0.
    constexpr Score(int v) : mg(v), eg(v) {}
    constexpr Score(int mg_, int eg_) : mg(mg_), eg(eg_) {}

    constexpr Score operator+(const Score &o) const { return {mg + o.mg, eg + o.eg}; }
    constexpr Score operator-(const Score &o) const { return {mg - o.mg, eg - o.eg}; }
    constexpr Score operator-() const { return {-mg, -eg}; }
    constexpr Score operator*(int s) const { return {mg * s, eg * s}; }
    constexpr Score &operator+=(const Score &o)
    {
      mg += o.mg;
      eg += o.eg;
      return *this;
    }
    constexpr Score &operator-=(const Score &o)
    {
      mg -= o.mg;
      eg -= o.eg;
      return *this;
    }
    constexpr bool operator==(const Score &o) const { return mg == o.mg && eg == o.eg; }
  };

  constexpr Score SCORE_ZERO = Score(0, 0);

  constexpr Score make_score(int mg, int eg) { return Score(mg, eg); }
  constexpr int mg_value(Score s) { return s.mg; }
  constexpr int eg_value(Score s) { return s.eg; }

  // Game phase: 0 = bare kings (pure endgame), 256 = full opening
  // material.
  constexpr int PHASE_EG = 0;
  constexpr int PHASE_MG = 256;

  // Piece values

  constexpr int PAWN_VALUE_MG = 128 * 100 / 128;   // 100
  constexpr int PAWN_VALUE_EG = 213 * 100 / 128;   // 166
  constexpr int KNIGHT_VALUE_MG = 781 * 100 / 128; // 610
  constexpr int KNIGHT_VALUE_EG = 854 * 100 / 128; // 667
  constexpr int BISHOP_VALUE_MG = 825 * 100 / 128; // 644
  constexpr int BISHOP_VALUE_EG = 915 * 100 / 128; // 714
  constexpr int ROOK_VALUE_MG = 1276 * 100 / 128;  // 996
  constexpr int ROOK_VALUE_EG = 1380 * 100 / 128;  // 1078
  constexpr int QUEEN_VALUE_MG = 2538 * 100 / 128; // 1982
  constexpr int QUEEN_VALUE_EG = 2682 * 100 / 128; // 2095
  constexpr int KING_VALUE_MG = 0;
  constexpr int KING_VALUE_EG = 0;

  // Non-pawn material thresholds (MidgameLimit/EndgameLimit),
  // used to compute game phase.
  constexpr int MIDGAME_LIMIT = 15258 * 100 / 128; // 11920
  constexpr int ENDGAME_LIMIT = 3915 * 100 / 128;  // 3058

  // Flat material values used outside the tapered eval — search.cpp uses
  // these directly (eval::QUEEN_VALUE, eval::materialValue) for SEE and
  // quiescence delta-pruning, which don't taper by phase, so the mg
  // value is the natural "how much is this piece worth" figure for that.
  constexpr int PAWN_VALUE = PAWN_VALUE_MG;
  constexpr int KNIGHT_VALUE = KNIGHT_VALUE_MG;
  constexpr int BISHOP_VALUE = BISHOP_VALUE_MG;
  constexpr int ROOK_VALUE = ROOK_VALUE_MG;
  constexpr int QUEEN_VALUE = QUEEN_VALUE_MG;
  constexpr int KING_VALUE = KING_VALUE_MG;

  inline Score pieceValueScore(chess::PieceType pt)
  {
    switch (pt.internal())
    {
    case chess::PieceType::underlying::PAWN:
      return make_score(PAWN_VALUE_MG, PAWN_VALUE_EG);
    case chess::PieceType::underlying::KNIGHT:
      return make_score(KNIGHT_VALUE_MG, KNIGHT_VALUE_EG);
    case chess::PieceType::underlying::BISHOP:
      return make_score(BISHOP_VALUE_MG, BISHOP_VALUE_EG);
    case chess::PieceType::underlying::ROOK:
      return make_score(ROOK_VALUE_MG, ROOK_VALUE_EG);
    case chess::PieceType::underlying::QUEEN:
      return make_score(QUEEN_VALUE_MG, QUEEN_VALUE_EG);
    default:
      return SCORE_ZERO;
    }
  }

  // Public API

  int materialValue(chess::PieceType pt);

  // Full static evaluation of the position, from side-to-move perspective.
  // Internally clamped (see eval.cpp) so that even a pathological position
  // can never return something outside a sane centipawn range
  int evaluate(const chess::Board &board);

  // Tapered eval helper: returns a 0..256 game-phase value (256 = full
  // opening material, 0 = bare kings), used to blend mg/eg internally.
  int gamePhase(const chess::Board &board);

} // namespace eval
