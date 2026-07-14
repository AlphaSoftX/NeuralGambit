#pragma once
#include "../chess.hpp"
#include "../tt/tt.h"
#include <chrono>
#include <atomic>
#include <algorithm>

namespace search
{

  constexpr int MAX_PLY = 128;
  constexpr int INF = 32000;
  constexpr int MATE_SCORE = 31000;

  // Sentinel for SearchContext::staticEvalStack meaning "no static eval
  // recorded at this ply yet this search" — used by the improving
  // heuristic (search.cpp) to tell "we don't have 2 plies of history
  // yet" apart from "we do, and it got worse". Deliberately outside the
  // reachable range of both a real eval ([-EVAL_CLAMP, EVAL_CLAMP] in
  // eval.cpp) and a mate score ([-MATE_SCORE, MATE_SCORE]), so it can
  // never be mistaken for either.
  constexpr int NO_STATIC_EVAL = -32001;

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
    std::atomic<bool> stop{false}; // prevent race conditions
    uint64_t nodes = 0;
    uint64_t tbHits = 0; // how many times a TB probe succeeded mid-search

    bool tbLoaded = false; // set by uci.cpp after tb_init() succeeds

    chess::Move killers[MAX_PLY][2]{};
    int history[2][64][64]{};

    // Countermove heuristic: indexed by [side-to-move-after-the-prev-move][prevFrom][prevTo],
    // stores the quiet move that has refuted that previous move most recently.
    // Tried as an extra move-ordering hint alongside killers/history.
    chess::Move counterMoves[2][64][64]{};

    // One static-eval slot per ply, overwritten as the current search
    // path descends through that ply. Since search is depth-first, by
    // the time a node at ply P recurses into ply P+1, staticEvalStack[P]
    // is guaranteed to hold *that node's own* static eval (not some
    // unrelated sibling's); reset to NO_STATIC_EVAL at
    // the start of every iterativeDeepening() call.
    int staticEvalStack[MAX_PLY];

    SearchContext() { std::fill(std::begin(staticEvalStack), std::end(staticEvalStack), NO_STATIC_EVAL); }

    bool timeUp() const;
  };

  void iterativeDeepening(SearchContext &ctx);

} // namespace search