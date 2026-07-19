#pragma once
#include "../chess.hpp"
#include "nnue_features.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace nnue
{

  constexpr int FT_SIZE = 256;     // feature-transformer output width, per perspective
  constexpr int HIDDEN_SIZE = 32;  // width of the two small dense layers

  struct Accumulator
  {
    alignas(32) std::array<int32_t, FT_SIZE> v{};
  };

  // Both perspectives together — what actually lives in one ply's slot of
  // search::SearchContext's accumulator stack.
  struct PerspectiveAccumulators
  {
    Accumulator white;
    Accumulator black;
  };

  // Loads a .nnue file written by export.py
  bool load(const std::string &path);

  // True once a net has been successfully loaded via load().
  bool isLoaded();

  // Releases the current net and reverts to "not loaded"
  void unload();

  // Enable/disable switch
  void setEnabled(bool enabled);
  bool isEnabled();

  // Accumulator helpers
  void refresh(const chess::Board &board, PerspectiveAccumulators &acc);

  void refreshOne(const chess::Board &board, chess::Color perspective, Accumulator &acc);

  void addPiece(PerspectiveAccumulators &acc, int whiteKingSq, int blackKingSqMirrored,
                chess::Color pieceColor, chess::PieceType pieceType, int pieceSq);

  void removePiece(PerspectiveAccumulators &acc, int whiteKingSq, int blackKingSqMirrored,
                   chess::Color pieceColor, chess::PieceType pieceType, int pieceSq);

  int evaluate(const PerspectiveAccumulators &acc, chess::Color sideToMove);

} // namespace nnue
