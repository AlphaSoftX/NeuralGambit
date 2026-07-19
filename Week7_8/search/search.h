#pragma once
#include "../chess.hpp"
#include "../tt/tt.h"
#include "../eval/eval.h"
#include "../nnue/nnue.h"
#include <chrono>
#include <atomic>
#include <algorithm>

namespace search
{

  constexpr int MAX_PLY = 128;
  constexpr int INF = 32000;
  constexpr int MATE_SCORE = 31000;

  // Sentinel for SearchContext::staticEvalStack meaning "no static eval
  // recorded at this ply yet this search"
  constexpr int NO_STATIC_EVAL = -32001;

  // One slot per ply of NNUE accumulators. 
  // + 1 due to writing for last ply as well
  constexpr int ACC_STACK_SIZE = MAX_PLY + 1;

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

    // NNUE accumulator stack — one PerspectiveAccumulators per ply. Slot
    // 0 is refreshed from the actual board at the start of every
    // iterativeDeepening() call (mirrors staticEvalStack's reset above);
    // every deeper slot is derived from its parent incrementally by
    // doMove()/doNullMove() in search.cpp. Left default-constructed
    // (all zero) and never touched if no .nnue file is loaded.
    nnue::PerspectiveAccumulators accStack[ACC_STACK_SIZE];

    // Cached copy of nnue::isLoaded() && nnue::isEnabled(), computed ONCE
    bool nnueActive = false;

    SearchContext()
    {
      std::fill(std::begin(staticEvalStack), std::end(staticEvalStack), NO_STATIC_EVAL);
    }

    bool timeUp() const;
  };

  // The one place every part of search.cpp goes to get a static evaluation.
  inline int staticEval(SearchContext &ctx, int ply)
  {
    if (ctx.nnueActive)
      return nnue::evaluate(ctx.accStack[ply], ctx.board.sideToMove());
    return eval::evaluate(ctx.board);
  }

  void iterativeDeepening(SearchContext &ctx);

} // namespace search
