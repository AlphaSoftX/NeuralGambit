#include "uci.h"
#include "../search/search.h"
#include "../moveutils.h"
#include "../fathom/tbprobe.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>

using namespace chess;

namespace uci
{

  // Forward to moveutils so uci::moveToUci keeps working for any callers.
  std::string moveToUci(const Move &m)
  {
    return moveutils::moveToUci(m);
  }

  namespace
  {

    constexpr const char *ENGINE_NAME = "MyChessEngine";
    constexpr const char *ENGINE_AUTHOR = "Arun";
    constexpr int DEFAULT_HASH_MB = 64;
    const std::string initBookPath = "";
    const std::string initSyzygyPath = "syzygy";

    struct EngineState
    {
      Board board;
      TranspositionTable tt{DEFAULT_HASH_MB};
      book::PolyglotBook book;
      bool useBook = false;
      bool tbLoaded = false; // true after a successful tb_init()
      std::thread searchThread;
      std::atomic<bool> *stopFlag = nullptr;

      EngineState()
      {
        if (!initBookPath.empty())
        {
          useBook = book.load(initBookPath);
        }
        if (!initSyzygyPath.empty())
        {
          tbLoaded = tb_init(initSyzygyPath.c_str());
        }
        std::cout << "info string loaded deps: " << useBook << " " << tbLoaded << std::endl;
      }
    };

    std::vector<std::string> split(const std::string &s)
    {
      std::istringstream iss(s);
      std::vector<std::string> out;
      std::string tok;
      while (iss >> tok)
        out.push_back(tok);
      return out;
    }

    Move parseUciMove(const Board &board, const std::string &uciStr)
    {
      Movelist legal;
      movegen::legalmoves(legal, board);
      for (const auto &m : legal)
        if (moveToUci(m) == uciStr)
          return m;
      return Move::NO_MOVE;
    }

    void handlePosition(EngineState &state, const std::vector<std::string> &tokens)
    {
      size_t i = 1;
      if (i >= tokens.size())
        return;

      if (tokens[i] == "startpos")
      {
        state.board.setFen(constants::STARTPOS);
        ++i;
      }
      else if (tokens[i] == "fen")
      {
        ++i;
        std::string fen;
        while (i < tokens.size() && tokens[i] != "moves")
        {
          if (!fen.empty())
            fen += " ";
          fen += tokens[i++];
        }
        state.board.setFen(fen);
      }

      if (i < tokens.size() && tokens[i] == "moves")
      {
        ++i;
        for (; i < tokens.size(); ++i)
        {
          Move m = parseUciMove(state.board, tokens[i]);
          if (m == Move::NO_MOVE)
            break;
          state.board.makeMove(m);
        }
      }
    }

    void handleGo(EngineState &state, const std::vector<std::string> &tokens)
    {
      search::Limits limits;

      for (size_t i = 1; i < tokens.size(); ++i)
      {
        auto nextI64 = [&](int64_t &dst)
        {
          if (i + 1 < tokens.size())
            dst = std::stoll(tokens[++i]);
        };
        if (tokens[i] == "depth")
        {
          if (i + 1 < tokens.size())
            limits.depth = std::stoi(tokens[++i]);
        }
        else if (tokens[i] == "movetime")
          nextI64(limits.movetime);
        else if (tokens[i] == "wtime")
          nextI64(limits.wtime);
        else if (tokens[i] == "btime")
          nextI64(limits.btime);
        else if (tokens[i] == "winc")
          nextI64(limits.winc);
        else if (tokens[i] == "binc")
          nextI64(limits.binc);
        else if (tokens[i] == "movestogo")
        {
          if (i + 1 < tokens.size())
            limits.movestogo = std::stoi(tokens[++i]);
        }
        else if (tokens[i] == "nodes")
        {
          if (i + 1 < tokens.size())
            limits.nodes = std::stoull(tokens[++i]);
        }
        else if (tokens[i] == "infinite")
          limits.infinite = true;
      }

      if (state.searchThread.joinable())
      {
        if (state.stopFlag)
          state.stopFlag->store(true);
        state.searchThread.join();
      }
      state.stopFlag = nullptr;

      // Opening book — instant, no search needed.
      if (state.useBook && state.book.isLoaded())
      {
        Move bookMove = state.book.probe(state.board);
        if (bookMove != Move::NO_MOVE)
        {
          std::cout << "bestmove " << moveToUci(bookMove) << std::endl;
          return;
        }
      }

      auto ctx = std::make_shared<search::SearchContext>();
      ctx->board = state.board;
      ctx->limits = limits;
      ctx->tt = &state.tt;
      ctx->tbLoaded = state.tbLoaded; // pass TB flag into the search context
      state.stopFlag = &ctx->stop;

      state.searchThread = std::thread([ctx]()
                                       { search::iterativeDeepening(*ctx); });
    }

    void handleSetOption(EngineState &state, const std::vector<std::string> &tokens)
    {
      // "setoption name <Name> value <Value>"
      std::string name, value;
      size_t i = 1;
      if (i < tokens.size() && tokens[i] == "name")
      {
        ++i;
        while (i < tokens.size() && tokens[i] != "value")
          name += tokens[i++] + " ";
        if (i < tokens.size() && tokens[i] == "value")
        {
          ++i;
          while (i < tokens.size())
            value += tokens[i++] + " ";
        }
      }
      if (!name.empty() && name.back() == ' ')
        name.pop_back();
      if (!value.empty() && value.back() == ' ')
        value.pop_back();

      if (name == "Hash")
      {
        try
        {
          state.tt.resize(std::stoul(value));
        }
        catch (...)
        {
        }
      }
      else if (name == "OwnBook")
      {
        state.useBook = (value == "true");
      }
      else if (name == "BookFile")
      {
        state.book.load(value);
      }
      else if (name == "SyzygyPath")
      {
        // tb_init() returns true on success; TB_LARGEST tells us the
        // maximum piece count of the TBs that were found.
        state.tbLoaded = tb_init(value.c_str());
        if (!state.tbLoaded)
          std::cerr << "info string Failed to load Syzygy TBs from: "
                    << value << std::endl;
        else
          std::cerr << "info string Syzygy TBs loaded. Largest: "
                    << TB_LARGEST << " pieces." << std::endl;
      }
    }

  } // namespace

  void loop()
  {
    EngineState state;
    state.board.setFen(constants::STARTPOS);

    std::string line;
    while (std::getline(std::cin, line))
    {
      auto tokens = split(line);
      if (tokens.empty())
        continue;
      const std::string &cmd = tokens[0];

      if (cmd == "uci")
      {
        std::cout << "id name " << ENGINE_NAME << "\n"
                  << "id author " << ENGINE_AUTHOR << "\n"
                  << "option name Hash type spin default "
                  << DEFAULT_HASH_MB << " min 1 max 2048\n"
                  << "option name OwnBook type check default " << (initBookPath.empty() ? "false" : "true") << "\n"
                  << "option name BookFile type string default " << (initBookPath.empty() ? "<empty>" : initBookPath) << "\n"
                  << "option name SyzygyPath type string default " << (initSyzygyPath.empty() ? "<empty>" : initSyzygyPath) << "\n"
                  << "uciok" << std::endl;
      }
      else if (cmd == "isready")
      {
        std::cout << "readyok" << std::endl;
      }
      else if (cmd == "ucinewgame")
      {
        state.tt.clear();
        state.board.setFen(constants::STARTPOS);
      }
      else if (cmd == "position")
      {
        handlePosition(state, tokens);
      }
      else if (cmd == "go")
      {
        handleGo(state, tokens);
      }
      else if (cmd == "stop")
      {
        if (state.stopFlag)
          state.stopFlag->store(true);
        if (state.searchThread.joinable())
          state.searchThread.join();
        state.stopFlag = nullptr;
      }
      else if (cmd == "setoption")
      {
        handleSetOption(state, tokens);
      }
      else if (cmd == "quit")
      {
        if (state.stopFlag)
          state.stopFlag->store(true);
        if (state.searchThread.joinable())
          state.searchThread.join();
        state.stopFlag = nullptr;
        tb_free(); // release Fathom memory before exit
        break;
      }
      // Unknown commands silently ignored per UCI spec.
    }
  }

} // namespace uci