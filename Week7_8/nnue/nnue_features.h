#pragma once
#include "../chess.hpp"
#include <cstdint>

namespace nnue::features
{

  constexpr int NUM_PIECE_TYPES_NO_KING = 5; // P, N, B, R, Q
  constexpr int NUM_PIECE_CODES = NUM_PIECE_TYPES_NO_KING * 2; // x friend/enemy
  constexpr int FEATURES_PER_KING_SQUARE = NUM_PIECE_CODES * 64;
  constexpr int NUM_FEATURES = 64 * FEATURES_PER_KING_SQUARE;
  constexpr int MAX_ACTIVE_FEATURES = 30; // 32 squares - 2 kings
  constexpr int PADDING_INDEX = NUM_FEATURES;

  // Vertically mirrors a 0-63 square index (flip rank, keep file) —
  // identical to converter.py's mirror_square().
  constexpr int mirror(int sq) { return sq ^ 56; }

  inline int pieceCode(chess::PieceType pt)
  {
    using PT = chess::PieceType::underlying;
    switch (pt.internal())
    {
    case PT::PAWN:
      return 0;
    case PT::KNIGHT:
      return 1;
    case PT::BISHOP:
      return 2;
    case PT::ROOK:
      return 3;
    case PT::QUEEN:
      return 4;
    default:
      return -1; // KING (or NONE)
    }
  }

  // The perspective's own king square
  inline int perspectiveKingSquare(const chess::Board &board, chess::Color perspective)
  {
    int ksq = board.kingSq(perspective).index();
    return (perspective == chess::Color::BLACK) ? mirror(ksq) : ksq;
  }

  // Computes the single feature index a given piece contributes to a
  // given perspective's accumulator.
  inline int featureIndex(chess::Color perspective, int perspectiveKingSq,
                          chess::Color pieceColor, chess::PieceType pieceType, int pieceSq)
  {
    int code = pieceCode(pieceType);
    if (code < 0)
      return -1;

    bool isFriend = (pieceColor == perspective);
    int fullCode = code + (isFriend ? 0 : NUM_PIECE_TYPES_NO_KING);
    int sq = (perspective == chess::Color::BLACK) ? mirror(pieceSq) : pieceSq;

    return perspectiveKingSq * FEATURES_PER_KING_SQUARE + fullCode * 64 + sq;
  }

  struct FeatureList
  {
    int idx[MAX_ACTIVE_FEATURES];
    int count = 0;
  };

  // Enumerates every non-king piece on the board and returns its full
  // active-feature list for one perspective.
  inline FeatureList buildFeatures(const chess::Board &board, chess::Color perspective)
  {
    FeatureList out;
    int ksq = perspectiveKingSquare(board, perspective);

    for (int sq = 0; sq < 64; ++sq)
    {
      chess::Piece p = board.at(chess::Square(sq));
      if (p == chess::Piece::NONE || p.type() == chess::PieceType::KING)
        continue;

      int idx = featureIndex(perspective, ksq, p.color(), p.type(), sq);
      out.idx[out.count++] = idx;
    }
    return out;
  }

} // namespace nnue::features
