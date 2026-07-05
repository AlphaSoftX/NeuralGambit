#include "book.h"
#include "zobrist_keys.hpp"
#include <fstream>
#include <algorithm>
#include <random>

using namespace chess;

// The Random64[781] table used to compute Polyglot zobrist keys now lives in
// zobrist_keys.hpp as the literal, verbatim spec constants
//
// Index layout (standard Polyglot spec):
//   [0..767]   piece-on-square keys: index = 64 * pieceCode + square
//              pieceCode: 0=BP 1=WP 2=BN 3=WN 4=BB 5=WB 6=BR 7=WR
//                         8=BQ 9=WQ 10=BK 11=WK
//   [768..771] castling rights: 768=WK-side 769=WQ-side 770=BK-side 771=BQ-side
//   [772..779] en-passant file (a..h), only XORed in if an EP capture is
//              actually possible for the side to move (pawn adjacency only —
//              legality/pins are irrelevant per spec)
//   [780]      side-to-move key, XORed in only when White is to move

namespace
{

  int pieceCode(Piece p)
  {
    int code = 0;
    switch (p.type().internal())
    {
      case PieceType::underlying::PAWN:   code = 0;  break;
      case PieceType::underlying::KNIGHT: code = 2;  break;
      case PieceType::underlying::BISHOP: code = 4;  break;
      case PieceType::underlying::ROOK:   code = 6;  break;
      case PieceType::underlying::QUEEN:  code = 8;  break;
      case PieceType::underlying::KING:   code = 10; break;
      default:                            return -1; // Catches NONE or unexpected types safely
    }

    if (p.color() == Color::WHITE)
      code += 1;
        
    return code;
  }

} // namespace

namespace book
{

  uint64_t polyglotKey(const Board &board)
  {
    const auto &R = zobrist::POLYGLOT_RANDOM64;
    uint64_t key = 0;

    for (int sq = 0; sq < 64; ++sq)
    {
      Piece p = board.at(static_cast<Square>(sq));
      int code = pieceCode(p);
      if (code >= 0)
        key ^= R[64 * code + sq];
    }

    // Castling rights — relies on chess-library's CastlingRights accessor.
    auto cr = board.castlingRights();
    if (cr.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE))
      key ^= R[768];
    if (cr.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE))
      key ^= R[769];
    if (cr.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE))
      key ^= R[770];
    if (cr.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE))
      key ^= R[771];

    // En-passant: only counts if a pawn of the side to move can actually
    // capture onto that square right now.
    Square epSq = board.enpassantSq();
    if (epSq != Square::NO_SQ)
    {
      int file = epSq.index() % 8;
      key ^= R[772 + file];
    }

    if (board.sideToMove() == Color::WHITE)
      key ^= R[780];

    return key;
  }

  Move decodePolyglotMove(const Board &board, uint16_t packed)
  {
    // Polyglot packs: bits 0-2 to-file, 3-5 to-rank, 6-8 from-file,
    // 9-11 from-rank, 12-14 promotion piece (0=none,1=N,2=B,3=R,4=Q).
    int toFile = packed & 0x7;
    int toRank = (packed >> 3) & 0x7;
    int fromFile = (packed >> 6) & 0x7;
    int fromRank = (packed >> 9) & 0x7;
    int promo = (packed >> 12) & 0x7;

    Square from = Square(fromRank * 8 + fromFile);
    Square to = Square(toRank * 8 + toFile);

    // Polyglot encodes castling as the king "capturing" its own rook
    // (from=king square, to=rook square). Resolve against legal moves so we
    // get the engine's own castling representation right, and so
    // promotions/normal moves are validated as legal.
    Movelist legal;
    movegen::legalmoves(legal, board);

    for (const auto &m : legal)
    {
      if (m.from() == from && m.to() == to)
      {
        if (promo == 0)
          return m;
        PieceType promoType;
        switch (promo)
        {
        case 1:
          promoType = PieceType::KNIGHT;
          break;
        case 2:
          promoType = PieceType::BISHOP;
          break;
        case 3:
          promoType = PieceType::ROOK;
          break;
        case 4:
          promoType = PieceType::QUEEN;
          break;
        }
        if (m.typeOf() == Move::PROMOTION && m.promotionType() == promoType)
          return m;
      }
    }

    // Handle the castling "king not takes own rook" encoding explicitly if the
    // direct from/to scan above didn't match (some books encode it that way).
    for (const auto &m : legal)
    {
      if (m.typeOf() != Move::CASTLING || m.from() != from)
        continue;

      if (
          (m.to() == Square::SQ_H1 && to == Square::SQ_G1) ||
          (m.to() == Square::SQ_A1 && to == Square::SQ_C1) ||
          (m.to() == Square::SQ_H8 && to == Square::SQ_G8) ||
          (m.to() == Square::SQ_A8 && to == Square::SQ_C8))
      {
        return m;
      }
    }

    return Move::NO_MOVE;
  }

  bool PolyglotBook::load(const std::string &path)
  {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
      return false;

    entries_.clear();
    PolyglotEntry e{};
    uint8_t buf[16];

    while (f.read(reinterpret_cast<char *>(buf), 16))
    {
      e.key = (uint64_t(buf[0]) << 56) | (uint64_t(buf[1]) << 48) |
              (uint64_t(buf[2]) << 40) | (uint64_t(buf[3]) << 32) |
              (uint64_t(buf[4]) << 24) | (uint64_t(buf[5]) << 16) |
              (uint64_t(buf[6]) << 8) | uint64_t(buf[7]);
      e.move = (uint16_t(buf[8]) << 8) | buf[9];
      e.weight = (uint16_t(buf[10]) << 8) | buf[11];
      entries_.push_back(e);
    }

    std::sort(entries_.begin(), entries_.end(),
              [](const PolyglotEntry &a, const PolyglotEntry &b)
              { return a.key < b.key; });

    return !entries_.empty();
  }

  Move PolyglotBook::probe(const Board &board) const
  {
    if (entries_.empty())
      return Move::NO_MOVE;

    uint64_t key = polyglotKey(board);
    auto lo = std::lower_bound(entries_.begin(), entries_.end(), key,
                               [](const PolyglotEntry &e, uint64_t k)
                               { return e.key < k; });

    if (lo == entries_.end() || lo->key != key)
      return Move::NO_MOVE;

    auto hi = lo;
    while (hi != entries_.end() && hi->key == key)
      ++hi;

    // Weighted random choice among all matching entries.
    uint32_t totalWeight = 0;
    for (auto it = lo; it != hi; ++it)
      totalWeight += it->weight ? it->weight : 1;

    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, totalWeight - 1);
    uint32_t pick = dist(rng);

    for (auto it = lo; it != hi; ++it)
    {
      uint32_t w = it->weight ? it->weight : 1;
      if (pick < w)
        return decodePolyglotMove(board, it->move);
      pick -= w;
    }

    return decodePolyglotMove(board, lo->move);
  }

} // namespace book
