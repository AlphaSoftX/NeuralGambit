#pragma once
#include "../chess.hpp"
#include "../tt/tt.h"
#include <chrono>
#include <atomic>

namespace search
{

  constexpr int MAX_PLY = 128;
  constexpr int INF = 32000;
  constexpr int MATE_SCORE = 31000;

  struct Limits
  {
    int depth = 0;
    int64_t movetime = 0;
    int64_t wtime = 0, btime = 0;
    int64_t winc = 0, binc = 0;
    int movestogo = 0;
    bool infinite = false;
    uint64_t nodes = 0;
  };

  struct SearchContext
  {
    chess::Board board;
    Limits limits;
    TranspositionTable *tt;

    std::chrono::steady_clock::time_point startTime;
    int64_t allocatedMs = 0;
    std::atomic<bool> stop{false};
    uint64_t nodes = 0;
    uint64_t tbHits = 0; // how many times a TB probe succeeded mid-search

    bool tbLoaded = false; // set by uci.cpp after tb_init() succeeds

    chess::Move killers[MAX_PLY][2]{};
    int history[2][64][64]{};

    bool timeUp() const;
  };

  int64_t allocateTime(const Limits &limits, chess::Color us);
  void iterativeDeepening(SearchContext &ctx);

} // namespace search