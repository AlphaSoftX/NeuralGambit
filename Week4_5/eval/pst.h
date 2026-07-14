#pragma once
#include "eval.h"
#include "../chess.hpp"
#include <array>

// Piece-Square Tables
//
// All tables are White-oriented (square 0 = a1). For Black pieces, use
// pieceSquareScore() below — it mirrors the square automatically.

namespace eval::pst
{

  namespace detail
  {

    // Raw Stockfish (mg, eg) pairs
    
    struct RawPair
    {
      int mg, eg;
    };

    inline Score scalePair(RawPair p)
    {
      return make_score(p.mg * 100 / 128, p.eg * 100 / 128);
    }

    // Bonus[Rank][File] for files A-D; files E-H mirror via mapToQueenside.
    inline constexpr RawPair KnightBonus[8][4] = {
        {{-175, -96}, {-92, -65}, {-74, -49}, {-73, -21}},
        {{-77, -67}, {-41, -54}, {-27, -18}, {-15, 8}},
        {{-61, -40}, {-17, -27}, {6, -8}, {12, 29}},
        {{-35, -35}, {8, -2}, {40, 13}, {49, 28}},
        {{-34, -45}, {13, -16}, {44, 9}, {51, 39}},
        {{-9, -51}, {22, -44}, {58, -16}, {53, 17}},
        {{-67, -69}, {-27, -50}, {4, -51}, {37, 12}},
        {{-201, -100}, {-83, -88}, {-56, -56}, {-26, -17}},
    };

    inline constexpr RawPair BishopBonus[8][4] = {
        {{-53, -57}, {-5, -30}, {-8, -37}, {-23, -12}},
        {{-15, -37}, {8, -13}, {19, -17}, {4, 1}},
        {{-7, -16}, {21, -1}, {-5, -2}, {17, 10}},
        {{-5, -20}, {11, -6}, {25, 0}, {39, 17}},
        {{-12, -17}, {29, -1}, {22, -14}, {31, 15}},
        {{-16, -30}, {6, 6}, {1, 4}, {11, 6}},
        {{-17, -31}, {-14, -20}, {5, -1}, {0, 1}},
        {{-48, -46}, {1, -42}, {-14, -37}, {-23, -24}},
    };

    inline constexpr RawPair RookBonus[8][4] = {
        {{-31, -9}, {-20, -13}, {-14, -10}, {-5, -9}},
        {{-21, -12}, {-13, -9}, {-8, -1}, {6, -2}},
        {{-25, 6}, {-11, -8}, {-1, -2}, {3, -6}},
        {{-13, -6}, {-5, 1}, {-4, -9}, {-6, 7}},
        {{-27, -5}, {-15, 8}, {-4, 7}, {3, -6}},
        {{-22, 6}, {-2, 1}, {6, -7}, {12, 10}},
        {{-2, 4}, {12, 5}, {16, 20}, {18, -5}},
        {{-17, 18}, {-19, 0}, {-1, 19}, {9, 13}},
    };

    inline constexpr RawPair QueenBonus[8][4] = {
        {{3, -69}, {-5, -57}, {-5, -47}, {4, -26}},
        {{-3, -55}, {5, -31}, {8, -22}, {12, -4}},
        {{-3, -39}, {6, -18}, {13, -9}, {7, 3}},
        {{4, -23}, {5, -3}, {9, 13}, {8, 24}},
        {{0, -29}, {14, -6}, {12, 9}, {5, 21}},
        {{-4, -38}, {10, -18}, {6, -12}, {8, 1}},
        {{-5, -50}, {6, -27}, {10, -24}, {8, -8}},
        {{-2, -75}, {-2, -52}, {1, -43}, {-2, -36}},
    };

    inline constexpr RawPair KingBonus[8][4] = {
        {{271, 1}, {327, 45}, {271, 85}, {198, 76}},
        {{278, 53}, {303, 100}, {234, 133}, {179, 135}},
        {{195, 88}, {258, 130}, {169, 169}, {120, 175}},
        {{164, 103}, {190, 156}, {138, 172}, {98, 172}},
        {{154, 96}, {179, 166}, {105, 199}, {70, 199}},
        {{123, 92}, {145, 172}, {81, 184}, {31, 191}},
        {{88, 47}, {120, 121}, {65, 116}, {33, 131}},
        {{59, 11}, {89, 59}, {45, 73}, {-1, 78}},
    };

    // Pawns get their own file-asymmetric table (rank 0/7 unused: pawns
    // never sit on the back ranks).
    inline constexpr RawPair PawnBonus[8][8] = {
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{3, -10}, {3, -6}, {10, 10}, {19, 0}, {16, 14}, {19, 7}, {7, -5}, {-5, -19}},
        {{-9, -10}, {-15, -10}, {11, -10}, {15, 4}, {32, 4}, {22, 3}, {5, -6}, {-22, -4}},
        {{-8, 6}, {-23, -2}, {6, -8}, {20, -4}, {40, -13}, {17, -12}, {4, -10}, {-12, -9}},
        {{13, 9}, {0, 4}, {-13, 3}, {1, -12}, {11, -12}, {-2, -6}, {-13, 13}, {5, 8}},
        {{-5, 28}, {-12, 20}, {-7, 21}, {22, 28}, {-8, 30}, {-5, 7}, {-15, 6}, {-18, 13}},
        {{-7, 0}, {7, -11}, {-3, 12}, {-13, 21}, {5, 25}, {-16, 19}, {10, 4}, {-8, 7}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    };

    // Maps files A..H to the queenside-mirrored index used by the
    // file-symmetric tables above (A/H->0, B/G->1, C/F->2, D/E->3).
    inline int mapToQueenside(int file) { return file < 4 ? file : 7 - file; }

    // PieceType::underlying is a scoped enum and does not implicitly
    // convert to an array index — this helper makes that explicit.
    inline int idx(chess::PieceType pt) { return static_cast<int>(pt.internal()); }

    inline std::array<std::array<Score, 64>, 6> &table()
    {
      static std::array<std::array<Score, 64>, 6> psq{};
      return psq;
    }

    inline bool &initializedFlag()
    {
      static bool initialized = false;
      return initialized;
    }

  } // namespace detail

  // Must be called once at startup before pieceSquareScore()/bonus() are
  // used (eval::evaluate() calls this lazily on first use, so callers
  // don't have to remember to invoke it themselves).
  inline void init()
  {
    using namespace chess;
    using namespace detail;
    auto &psq = table();

    for (int sq = 0; sq < 64; ++sq)
    {
      int file = sq % 8;
      int rank = sq / 8;
      int qFile = mapToQueenside(file);

      psq[idx(PieceType::PAWN)][sq] = pieceValueScore(PieceType::PAWN) + scalePair(PawnBonus[rank][file]);
      psq[idx(PieceType::KNIGHT)][sq] = pieceValueScore(PieceType::KNIGHT) + scalePair(KnightBonus[rank][qFile]);
      psq[idx(PieceType::BISHOP)][sq] = pieceValueScore(PieceType::BISHOP) + scalePair(BishopBonus[rank][qFile]);
      psq[idx(PieceType::ROOK)][sq] = pieceValueScore(PieceType::ROOK) + scalePair(RookBonus[rank][qFile]);
      psq[idx(PieceType::QUEEN)][sq] = pieceValueScore(PieceType::QUEEN) + scalePair(QueenBonus[rank][qFile]);
      psq[idx(PieceType::KING)][sq] = pieceValueScore(PieceType::KING) + scalePair(KingBonus[rank][qFile]);
    }
    initializedFlag() = true;
  }

  // Returns material + piece-square bonus for a piece of type `pt` sitting
  // on `sq`, from White's perspective. Table is indexed directly
  // (already White-oriented).
  inline Score bonus(chess::PieceType pt, chess::Square sq)
  {
    if (!detail::initializedFlag())
      init();
    return detail::table()[detail::idx(pt)][sq.index()];
  }

  // Convenience wrapper that also handles Black-square mirroring, so the
  // top-level evaluator can just call this once per piece and add/subtract
  // by color.
  inline Score pieceSquareScore(chess::PieceType pt, chess::Color color, chess::Square sq)
  {
    return bonus(pt, sq.relative_square(color));
  }

} // namespace eval::pst
