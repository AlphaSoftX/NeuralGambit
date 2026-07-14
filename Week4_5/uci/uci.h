#pragma once
#include "../chess.hpp"
#include "../tt/tt.h"
#include "../book/book.h"
#include <string>

namespace uci
{

  // Converts a move to UCI long-algebraic notation ("e2e4", "e7e8q", "0000"
  // for null/no move).
  std::string moveToUci(const chess::Move &m);

  // Main UCI command loop. Reads from stdin until "quit".
  void loop();

} // namespace uci
