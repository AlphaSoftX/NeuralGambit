#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
using namespace std;

#include "includes/chess.hpp"
using namespace chess;

#include "includes/json.hpp"
using json = nlohmann::json;

string targetFile = "mate_in_2.json";

class Engine
{
  Board board;

  Movelist getOrderedLegalMoves()
  {
    Movelist moves;
    movegen::legalmoves(moves, board);

    for (int i = 0; i < moves.size(); i++)
    {
      Move move = moves[i];

      if (board.givesCheck(move) != CheckType::NO_CHECK)
        moves[i].setScore(2);
      else if (board.isCapture(move))
        moves[i].setScore(1);
      else
        moves[i].setScore(0);
    }

    std::sort(moves.begin(), moves.end(), [](const Move &a, const Move &b)
              { return a.score() > b.score(); });

    return moves;
  }

public:
  Engine(string initBoardFen)
  {
    board = Board(initBoardFen);
  }

  json simulate(int alpha, int beta, bool isMaxPlayer, int depth)
  {
    Movelist moves = getOrderedLegalMoves();

    json result;
    result["moves"] = "";

    // temp code
    if (depth == 10)
    {
      result["score"] = 0;
      return result;
    }

    if (moves.empty())
    {
      if (board.inCheck())
        result["score"] = isMaxPlayer ? -100 + depth : 100 - depth;
      else
        result["score"] = 0;
    }
    else if (isMaxPlayer)
    {
      result["score"] = -200;

      for (auto &move : moves)
      {
        board.makeMove(move);
        json fResult = simulate(alpha, beta, !isMaxPlayer, depth + 1);
        board.unmakeMove(move);
        alpha = max(alpha, (int)fResult["score"]);
        if ((int)result["score"] < (int)fResult["score"])
        {
          result["score"] = fResult["score"];
          result["moves"] = " " + uci::moveToSan(board, move) + (string)fResult["moves"];
        }
        if (alpha >= beta)
          break;
      }
    }
    else
    {
      result["score"] = 200;

      for (auto &move : moves)
      {
        board.makeMove(move);
        json fResult = simulate(alpha, beta, !isMaxPlayer, depth + 1);
        board.unmakeMove(move);
        beta = min(beta, (int)fResult["score"]);
        if ((int)result["score"] > (int)fResult["score"])
        {
          result["score"] = fResult["score"];
          result["moves"] = " " + uci::moveToSan(board, move) + (string)fResult["moves"];
        }
        if (alpha >= beta)
          break;
      }
    }
    return result;
  }
};

int main()
{
  ifstream file(targetFile);
  json data = json::parse(file);

  cout << "Starting..." << endl;

  for (auto &[key, value] : data.items())
  {
    Engine engine(key);
    json result = engine.simulate(-200, 200, true, 0);

    cout << key << " : " << value << " :" << result["moves"] << endl;
  }
}