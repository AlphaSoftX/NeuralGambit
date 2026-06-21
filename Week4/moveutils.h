#pragma once
#include "chess.hpp"
#include <string>

namespace moveutils
{

  inline std::string moveToUci(const chess::Move &m)
  {
    if (m == chess::Move::NO_MOVE)
      return "0000";

    std::string s;
    chess::Square from = m.from();
    chess::Square to = m.to();

    if (m.typeOf() == chess::Move::CASTLING)
    {
      // chess-library encodes castling with to() = the rook's square
      // (e.g. e1h1), but UCI expects the king's actual destination
      // (e1g1, e1c1, e8g8, e8c8).
      bool kingSide = to.index() > from.index();
      int rank = from.index() / 8;
      to = chess::Square(rank * 8 + (kingSide ? 6 : 2)); // g-file or c-file
    }

    s += static_cast<char>('a' + (from.index() % 8));
    s += static_cast<char>('1' + (from.index() / 8));
    s += static_cast<char>('a' + (to.index() % 8));
    s += static_cast<char>('1' + (to.index() / 8));

    if (m.typeOf() == chess::Move::PROMOTION)
    {
      switch (m.promotionType().internal())
      {
      case chess::PieceType::underlying::KNIGHT:
        s += 'n';
        break;
      case chess::PieceType::underlying::BISHOP:
        s += 'b';
        break;
      case chess::PieceType::underlying::ROOK:
        s += 'r';
        break;
      default:
        s += 'q';
        break;
      }
    }
    return s;
  }

} // namespace moveutils