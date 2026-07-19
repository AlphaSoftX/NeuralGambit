#pragma once
#include "../chess.hpp"
#include <string>
#include <vector>
#include <cstdint>

// Polyglot Opening Book (.bin format)
//
// Format: sequence of 16-byte big-endian entries:
//   uint64_t key     - Polyglot zobrist hash of the position (BEFORE the move)
//   uint16_t move    - packed move (see decode below)
//   uint16_t weight  - relative probability of choosing this move
//   uint32_t learn   - unused by us
//
// Entries are sorted by key, allowing binary search. The book is memory
// mapped fully into a small vector.

namespace book
{

  struct PolyglotEntry
  {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
  };

  class PolyglotBook
  {
  public:
    // loads the Polyglot file into a vector
    bool load(const std::string &path);

    // Returns a legal move from the book for this position, weighted-random
    // among matching entries, or Move::NO_MOVE if no book hit.
    chess::Move probe(const chess::Board &board) const;

    bool isLoaded() const { return !entries_.empty(); }

  private:
    std::vector<PolyglotEntry> entries_;
  };

  // Computes the Polyglot-specific zobrist key for a position. This is
  // DIFFERENT from chess-library's internal hash() — Polyglot uses its own
  // fixed random table (Random64), so we import it independently.
  // The table itself lives in zobrist_keys.hpp (verbatim Polyglot spec
  // constants).
  uint64_t polyglotKey(const chess::Board &board);

  // Decodes a Polyglot-packed move into a chess-library Move, validated
  // against the current position's legal moves (to resolve castling encoding
  // differences and promotions correctly).
  chess::Move decodePolyglotMove(const chess::Board &board, uint16_t packed);

} // namespace book
