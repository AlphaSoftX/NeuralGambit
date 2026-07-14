#pragma once
#include <cstdint>
#include <vector>

// Transposition Table
//
// Default 64MB table.
// Each entry is 16 bytes -> 64MB / 16B = ~4.19M entries.
// Adjust via UCI "setoption name Hash value <MB>" (wired up in uci.cpp).

enum class TTFlag : uint8_t
{
  NONE = 0,
  EXACT = 1,
  LOWERBOUND = 2, // fail-high (beta cutoff)
  UPPERBOUND = 3  // fail-low
};

struct TTEntry
{
  uint64_t key = 0;      // full zobrist key (used to verify on lookup)
  int16_t score = 0;     // score from this position's perspective
  uint16_t bestMove = 0; // packed move (from-square, to-square, promo) — pairs with chess-library's Move
  uint8_t depth = 0;
  TTFlag flag = TTFlag::NONE;
  uint8_t age = 0; // search generation, used for replacement strategy
};

class TranspositionTable
{
public:
  explicit TranspositionTable(size_t sizeMB = 64)
  {
    resize(sizeMB);
  }

  void resize(size_t sizeMB)
  {
    size_t bytes = sizeMB * 1024ULL * 1024ULL;
    size_t numEntries = bytes / sizeof(TTEntry);
    // round down to power of two for fast masking
    size_t pow2 = 1;
    while (pow2 * 2 <= numEntries)
      pow2 *= 2;
    if (pow2 == 0)
      pow2 = 1;
    mask_ = pow2 - 1;
    table_.assign(pow2, TTEntry{});
  }

  void clear()
  {
    std::fill(table_.begin(), table_.end(), TTEntry{});
    generation_ = 0;
  }

  void newSearch() { ++generation_; }

  TTEntry *probe(uint64_t key, bool &found)
  {
    TTEntry &e = table_[index(key)];
    found = (e.key == key && e.flag != TTFlag::NONE);
    return &e;
  }

  void store(uint64_t key, int depth, int score, TTFlag flag, uint16_t bestMove)
  {
    TTEntry &e = table_[index(key)];
    // Replacement strategy: prefer deeper searches, but always replace
    // stale entries from previous search generations.
    if (e.age != generation_ || depth >= e.depth || e.flag == TTFlag::NONE)
    {
      e.key = key;
      e.score = static_cast<int16_t>(score);
      e.depth = static_cast<uint8_t>(depth);
      e.flag = flag;
      e.bestMove = bestMove;
      e.age = generation_;
    }
  }

  size_t entryCount() const { return table_.size(); }

private:
  size_t index(uint64_t key) const { return key & mask_; }

  std::vector<TTEntry> table_;
  size_t mask_ = 0;
  uint8_t generation_ = 0;
};
