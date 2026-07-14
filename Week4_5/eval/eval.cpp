#include "eval.h"
#include "pst.h"
#include "../chess.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace chess;

// Material imbalance

namespace eval::material
{

  namespace
  {

    // Second-degree polynomial imbalance parameters
    //
    // Row/col order: [bishop-pair, pawn, knight, bishop, rook, queen]
    constexpr int QuadraticOurs[6][6] = {
        {1438, 0, 0, 0, 0, 0},
        {40, 38, 0, 0, 0, 0},
        {32, 255, -62, 0, 0, 0},
        {0, 104, 4, 0, 0, 0},
        {-26, -2, 47, 105, -208, 0},
        {-189, 24, 117, 133, -134, -6},
    };

    constexpr int QuadraticTheirs[6][6] = {
        {0, 0, 0, 0, 0, 0},
        {36, 0, 0, 0, 0, 0},
        {9, 63, 0, 0, 0, 0},
        {59, 65, 42, 0, 0, 0},
        {46, 39, 24, -24, 0, 0},
        {97, 100, -42, 137, 268, 0},
    };

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }

    // pieceCount[color][0..5] = {bishop-pair(0/1), pawns, knights, bishops,
    // rooks, queens}, matching the QuadraticOurs/Theirs row order above.
    int imbalanceForSide(const int pieceCount[2][6], int us)
    {
      int them = 1 - us;
      int bonus = 0;

      for (int pt1 = 0; pt1 < 6; ++pt1)
      {
        if (!pieceCount[us][pt1])
          continue;

        int v = 0;
        for (int pt2 = 0; pt2 <= pt1; ++pt2)
          v += QuadraticOurs[pt1][pt2] * pieceCount[us][pt2] +
               QuadraticTheirs[pt1][pt2] * pieceCount[them][pt2];

        bonus += pieceCount[us][pt1] * v;
      }

      return bonus;
    }

  } // namespace

  // Returns White's imbalance bonus minus Black's, as a packed Score
  // (Stockfish applies the same value to both mg and eg). Computed fresh
  // every call — chess-library has no material-key hash hook to cache
  // this against, but it's cheap (just popcounts), so that's fine.
  Score imbalance(const Board &board)
  {
    int pieceCount[2][6];

    for (int c = 0; c < 2; ++c)
    {
      Color color(c);
      int bishops = board.pieces(PieceType::BISHOP, color).count();

      pieceCount[c][0] = bishops > 1;
      pieceCount[c][1] = board.pieces(PieceType::PAWN, color).count();
      pieceCount[c][2] = board.pieces(PieceType::KNIGHT, color).count();
      pieceCount[c][3] = bishops;
      pieceCount[c][4] = board.pieces(PieceType::ROOK, color).count();
      pieceCount[c][5] = board.pieces(PieceType::QUEEN, color).count();
    }

    // Stockfish divides the raw polynomial by 16 to bring it down to
    // centipawn scale, then applies our own 100/128 rescale on top.
    int white = scale(imbalanceForSide(pieceCount, 0) / 16);
    int black = scale(imbalanceForSide(pieceCount, 1) / 16);

    int v = white - black;
    return make_score(v, v);
  }

  // Game phase, 0 (bare kings) .. 256 (full opening material), based on
  // total non-pawn material on the board
  int gamePhase(const Board &board)
  {
    int npm = 0;
    npm += board.pieces(PieceType::KNIGHT).count() * KNIGHT_VALUE_MG;
    npm += board.pieces(PieceType::BISHOP).count() * BISHOP_VALUE_MG;
    npm += board.pieces(PieceType::ROOK).count() * ROOK_VALUE_MG;
    npm += board.pieces(PieceType::QUEEN).count() * QUEEN_VALUE_MG;

    if (npm > MIDGAME_LIMIT)
      npm = MIDGAME_LIMIT;
    if (npm < ENDGAME_LIMIT)
      npm = ENDGAME_LIMIT;

    return ((npm - ENDGAME_LIMIT) * 256) / (MIDGAME_LIMIT - ENDGAME_LIMIT);
  }

} // namespace eval::material

// Pawn structure: isolated / doubled / backward / connected / passed pawns

namespace eval::pawn_structure
{

  struct Info
  {
    Score score = 0;              // White's pawn-structure score minus Black's
    chess::Bitboard passed[2]{};  // passed pawns, indexed by Color
  };

  namespace
  {

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }
    inline Score scaleScore(int mg, int eg) { return make_score(scale(mg), scale(eg)); }

    const Score Backward = scaleScore(9, 24);
    const Score Doubled = scaleScore(11, 56);
    const Score Isolated = scaleScore(5, 15);
    const Score WeakLever = scaleScore(0, 56);
    const Score WeakUnopposed = scaleScore(13, 27);
    constexpr int ConnectedRaw[8] = {0, 7, 8, 12, 29, 48, 86, 0};

    // PassedRank[rank] — (mg, eg) bonus by rank of a passed pawn (own-side
    // relative, index 0 = own 2nd rank .. index 6 = about to promote).
    struct RawPair
    {
      int mg, eg;
    };
    constexpr RawPair PassedRankRaw[8] = {
        {0, 0}, {10, 28}, {17, 33}, {15, 41}, {62, 72}, {168, 177}, {276, 260}, {0, 0}};

    inline Bitboard fileMask(int file) { return attacks::MASK_FILE[file]; }
    inline Bitboard rankMask(int rank) { return attacks::MASK_RANK[rank]; }

    // Adjacent files only (excludes the pawn's own file) — used for
    // isolated/backward/connected checks, which care about *support from
    // a neighboring file*, not the pawn's own file.
    inline Bitboard neighborFilesMask(int file)
    {
      Bitboard mask;
      if (file > 0)
        mask |= fileMask(file - 1);
      if (file < 7)
        mask |= fileMask(file + 1);
      return mask;
    }

    // Own file + both adjacent files (clamped at the board edge) — the
    // "passed pawn span": if no enemy pawn sits anywhere in this span
    // ahead of the pawn, nothing can ever stop or capture it on its way
    // to promotion, so it's passed.
    inline Bitboard passedSpanMask(int file)
    {
      Bitboard mask = fileMask(file);
      mask |= neighborFilesMask(file);
      return mask;
    }

    // All ranks strictly ahead of `rank`, toward promotion, for `side`.
    inline Bitboard ranksAheadMask(int rank, Color side)
    {
      Bitboard mask;
      if (side == Color::WHITE)
        for (int r = rank + 1; r <= 7; ++r)
          mask |= rankMask(r);
      else
        for (int r = rank - 1; r >= 0; --r)
          mask |= rankMask(r);
      return mask;
    }

    // All ranks level with or behind `rank`, for `side` — used by the
    // backward-pawn check (is there a friendly pawn that could have
    // escorted this one forward by pushing up beside it?).
    inline Bitboard ranksBehindOrLevelMask(int rank, Color side)
    {
      Bitboard mask;
      if (side == Color::WHITE)
        for (int r = 0; r <= rank; ++r)
          mask |= rankMask(r);
      else
        for (int r = 7; r >= rank; --r)
          mask |= rankMask(r);
      return mask;
    }

    struct SideResult
    {
      Score score = 0;
      Bitboard passed;
    };

    SideResult evaluateSide(Bitboard ownPawns, Bitboard enemyPawns, Color side)
    {
      SideResult result;

      Bitboard pawns = ownPawns;
      while (pawns)
      {
        Square sq = pawns.pop();
        int file = sq.index() % 8;
        int rank = sq.index() / 8;
        int rankFromOwn = (side == Color::WHITE) ? rank : (7 - rank);

        int upIdx = sq.index() + (side == Color::WHITE ? 8 : -8);
        int downIdx = sq.index() + (side == Color::WHITE ? -8 : 8);
        bool hasUp = upIdx >= 0 && upIdx < 64;
        bool hasDown = downIdx >= 0 && downIdx < 64;

        Bitboard opposed = enemyPawns & fileMask(file) & ranksAheadMask(rank, side);
        Bitboard blocked = hasUp ? (enemyPawns & Bitboard::fromSquare(upIdx)) : Bitboard(0ull);
        Bitboard leverPush = hasUp ? (attacks::pawn(side, Square(upIdx)) & enemyPawns) : Bitboard(0ull);
        Bitboard lever = attacks::pawn(side, sq) & enemyPawns;
        Bitboard doubled = hasDown ? (ownPawns & Bitboard::fromSquare(downIdx) & fileMask(file)) : Bitboard(0ull);

        Bitboard neighbours = ownPawns & neighborFilesMask(file);
        Bitboard phalanx = neighbours & rankMask(rank);
        Bitboard support = hasDown ? (neighbours & rankMask(downIdx / 8)) : Bitboard(0ull);

        // Passed: no enemy pawn anywhere in the own-file-plus-adjacent
        // span ahead of this pawn — nothing can ever stop or capture it.
        bool passed = (enemyPawns & passedSpanMask(file) & ranksAheadMask(rank, side)).empty();

        // Backward: no own pawn on an adjacent file sits level-with or
        // behind this one (none could have escorted it forward), and its
        // push square is either blocked or covered by an enemy pawn.
        bool backward = (neighbours & ranksBehindOrLevelMask(rank, side)).empty() &&
                         (!leverPush.empty() || !blocked.empty());

        if (passed)
          result.passed.set(sq.index());

        // --- Scoring ---
        if (!phalanx.empty() || !support.empty())
        {
          int v = ConnectedRaw[rankFromOwn] * (2 + int(!phalanx.empty()) - int(!opposed.empty())) +
                  21 * support.count();
          result.score += scaleScore(v, v * std::max(0, rankFromOwn - 2) / 4);
        }
        else if (neighbours.empty())
        {
          result.score -= Isolated;
          if (opposed.empty())
            result.score -= WeakUnopposed;
        }
        else if (backward)
        {
          result.score -= Backward;
          if (opposed.empty())
            result.score -= WeakUnopposed;
        }

        if (support.empty())
        {
          if (!doubled.empty())
            result.score -= Doubled;
          if (lever.count() > 1)
            result.score -= WeakLever;
        }

        if (passed)
          result.score += scaleScore(PassedRankRaw[rankFromOwn].mg, PassedRankRaw[rankFromOwn].eg);
      }

      return result;
    }

  } // namespace

  // Computed fresh every call — chess-library has no pawn-key hash hook
  // to cache this against, but it's a
  // handful of bitboard ops per pawn, so that's fine.
  Info evaluate(const Board &board)
  {
    Bitboard whitePawns = board.pieces(PieceType::PAWN, Color::WHITE);
    Bitboard blackPawns = board.pieces(PieceType::PAWN, Color::BLACK);

    SideResult white = evaluateSide(whitePawns, blackPawns, Color::WHITE);
    SideResult black = evaluateSide(blackPawns, whitePawns, Color::BLACK);

    Info info;
    info.score = white.score - black.score;
    info.passed[int(Color::WHITE)] = white.passed;
    info.passed[int(Color::BLACK)] = black.passed;
    return info;
  }

} // namespace eval::pawn_structure

// King safety: pawn shelter/storm + a simplified king-danger term
namespace eval::king_safety
{

  namespace
  {

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }

    constexpr int ShelterStrength[4][8] = {
        {-6, 81, 93, 58, 39, 18, 25, 0},
        {-43, 61, 35, -49, -29, -11, -63, 0},
        {-10, 75, 23, -2, 32, 3, -45, 0},
        {-39, -13, -29, -52, -48, -67, -166, 0},
    };
    constexpr int UnblockedStorm[4][8] = {
        {85, -289, -166, 97, 50, 45, 50, 0},
        {46, -25, 122, 45, 37, -10, 20, 0},
        {-6, 51, 168, 34, -2, -22, -14, 0},
        {-15, -11, 101, 4, 11, -15, -29, 0},
    };
    constexpr int BLOCKED_STORM_MG = 82, BLOCKED_STORM_EG = 82;

    // King danger constants
    constexpr int KNIGHT_ATTACK_WEIGHT = 81;
    constexpr int BISHOP_ATTACK_WEIGHT = 52;
    constexpr int ROOK_ATTACK_WEIGHT = 44;
    constexpr int QUEEN_ATTACK_WEIGHT = 10;
    constexpr int KNIGHT_SAFE_CHECK = 790;
    constexpr int BISHOP_SAFE_CHECK = 635;
    constexpr int ROOK_SAFE_CHECK = 1080;
    constexpr int QUEEN_SAFE_CHECK = 780;

    inline Bitboard fileMask(int file) { return attacks::MASK_FILE[file]; }
    inline int mapToQueenside(int file) { return file < 4 ? file : 7 - file; }

    inline int relRank(Color side, Square sq)
    {
      int rank = sq.index() / 8;
      return side == Color::WHITE ? rank : 7 - rank;
    }

    // Shelter/storm bonus for a hypothetical king placement at `ksq`
    // (called both for the actual king square and for post-castling
    // squares, so the caller can pick whichever is best).
    Score evaluateShelter(const Board &board, Color us, Square ksq)
    {
      Color them = ~us;
      Bitboard ourPawns = board.pieces(PieceType::PAWN, us);
      Bitboard theirPawns = board.pieces(PieceType::PAWN, them);

      Score bonus = make_score(scale(5), scale(5));

      int kingFile = ksq.index() % 8;
      int center = std::clamp(kingFile, 1, 6);

      for (int f = center - 1; f <= center + 1; ++f)
      {
        int d = mapToQueenside(f);
        Bitboard fileMaskBB = fileMask(f);

        int ourRank = 0;
        {
          Bitboard b = ourPawns & fileMaskBB;
          while (b)
          {
            Square sq = b.pop();
            int r = relRank(us, sq);
            ourRank = (ourRank == 0) ? r : std::min(ourRank, r);
          }
        }

        int theirRank = 0;
        {
          Bitboard b = theirPawns & fileMaskBB;
          while (b)
          {
            Square sq = b.pop();
            int r = relRank(us, sq);
            theirRank = std::max(theirRank, r);
          }
        }

        bonus += make_score(scale(ShelterStrength[d][ourRank]), 0);

        if (ourRank && ourRank == theirRank - 1)
          bonus -= (theirRank == 2) ? make_score(scale(BLOCKED_STORM_MG), scale(BLOCKED_STORM_EG)) : Score(0);
        else
          bonus -= make_score(scale(UnblockedStorm[d][theirRank]), 0);
      }

      return bonus;
    }

    // Counts enemy attackers of type `pieceBB` touching `ring`, adding
    // `weight` per attacking piece to kingAttackersWeight and 1 per piece
    // to kingAttackersCount
    template <typename AttackFn>
    void accumulateAttackers(Bitboard pieceBB, Bitboard ring, int weight, AttackFn attackFn,
                              int &count, int &weightSum)
    {
      Bitboard bb = pieceBB;
      while (bb)
      {
        Square sq = bb.pop();
        if (!(attackFn(sq) & ring).empty())
        {
          ++count;
          weightSum += weight;
        }
      }
    }

  } // namespace

  Score evaluate(const Board &board)
  {
    Score total = 0;

    for (int c = 0; c < 2; ++c)
    {
      Color us(c);
      Color them = ~us;
      Square ksq = board.kingSq(us);

      // --- Shelter/storm: best of "as-is" and any castling still available.
      Score shelter = evaluateShelter(board, us, ksq);

      auto tryCastled = [&](Board::CastlingRights::Side side, Square dest)
      {
        if (board.castlingRights().has(us, side))
        {
          Score alt = evaluateShelter(board, us, dest);
          if (mg_value(alt) > mg_value(shelter))
            shelter = alt;
        }
      };
      Square kingSideDest = Square(Square::SQ_G1).relative_square(us);
      Square queenSideDest = Square(Square::SQ_C1).relative_square(us);
      tryCastled(Board::CastlingRights::Side::KING_SIDE, kingSideDest);
      tryCastled(Board::CastlingRights::Side::QUEEN_SIDE, queenSideDest);

      // Endgame-only bonus for keeping the king close to its own pawns.
      Bitboard ownPawns = board.pieces(PieceType::PAWN, us);
      int minPawnDist = 0;
      if (!ownPawns.empty())
      {
        minPawnDist = 8;
        Bitboard b = ownPawns;
        while (b)
        {
          Square sq = b.pop();
          minPawnDist = std::min(minPawnDist, Square::distance(ksq, sq));
        }
      }
      shelter -= make_score(0, scale(16 * minPawnDist));

      // --- Simplified king danger
      Bitboard ring = attacks::king(ksq);
      Bitboard occ = board.occ();

      int attackersCount = 0, attackersWeight = 0;
      accumulateAttackers(board.pieces(PieceType::KNIGHT, them), ring, KNIGHT_ATTACK_WEIGHT,
                           [](Square s)
                           { return attacks::knight(s); },
                           attackersCount, attackersWeight);
      accumulateAttackers(board.pieces(PieceType::BISHOP, them), ring, BISHOP_ATTACK_WEIGHT,
                           [occ](Square s)
                           { return attacks::bishop(s, occ); },
                           attackersCount, attackersWeight);
      accumulateAttackers(board.pieces(PieceType::ROOK, them), ring, ROOK_ATTACK_WEIGHT,
                           [occ](Square s)
                           { return attacks::rook(s, occ); },
                           attackersCount, attackersWeight);
      accumulateAttackers(board.pieces(PieceType::QUEEN, them), ring, QUEEN_ATTACK_WEIGHT,
                           [occ](Square s)
                           { return attacks::queen(s, occ); },
                           attackersCount, attackersWeight);

      int kingDanger = attackersCount * attackersWeight;

      // Safe checks: squares (empty, reachable, not defended by us) from
      // which an enemy piece of each type could deliver check right now.
      auto unionAttacks = [&](Bitboard pieceBB, auto attackFn)
      {
        Bitboard u;
        Bitboard bb = pieceBB;
        while (bb)
          u |= attackFn(bb.pop());
        return u;
      };
      auto hasSafeCheck = [&](Bitboard landingSquares)
      {
        Bitboard b = landingSquares & ~occ;
        while (b)
        {
          Square sq = b.pop();
          if (!board.isAttacked(sq, us))
            return true;
        }
        return false;
      };

      Bitboard knightCheckSquares = attacks::knight(ksq);
      Bitboard bishopCheckSquares = attacks::bishop(ksq, occ);
      Bitboard rookCheckSquares = attacks::rook(ksq, occ);
      Bitboard queenCheckSquares = bishopCheckSquares | rookCheckSquares;

      Bitboard theirKnightAttacks = unionAttacks(board.pieces(PieceType::KNIGHT, them),
                                                  [](Square s)
                                                  { return attacks::knight(s); });
      Bitboard theirBishopAttacks = unionAttacks(board.pieces(PieceType::BISHOP, them),
                                                  [occ](Square s)
                                                  { return attacks::bishop(s, occ); });
      Bitboard theirRookAttacks = unionAttacks(board.pieces(PieceType::ROOK, them),
                                                [occ](Square s)
                                                { return attacks::rook(s, occ); });
      Bitboard theirQueenAttacks = unionAttacks(board.pieces(PieceType::QUEEN, them),
                                                 [occ](Square s)
                                                 { return attacks::queen(s, occ); });

      if (hasSafeCheck(knightCheckSquares & theirKnightAttacks))
        kingDanger += KNIGHT_SAFE_CHECK;
      if (hasSafeCheck(rookCheckSquares & theirRookAttacks))
        kingDanger += ROOK_SAFE_CHECK;
      if (hasSafeCheck(bishopCheckSquares & theirBishopAttacks))
        kingDanger += BISHOP_SAFE_CHECK;
      if (hasSafeCheck(queenCheckSquares & theirQueenAttacks))
        kingDanger += QUEEN_SAFE_CHECK;

      Score danger = 0;
      if (kingDanger > 100)
        danger = make_score(scale(kingDanger * kingDanger / 4096), scale(kingDanger / 16));

      Score sideScore = shelter - danger;
      total += (us == Color::WHITE) ? sideScore : -sideScore;
    }

    return total;
  }

} // namespace eval::king_safety

// Mobility

namespace eval::mobility
{

  namespace
  {

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }
    struct RawPair
    {
      int mg, eg;
    };
    inline Score scaleScore(RawPair p) { return make_score(scale(p.mg), scale(p.eg)); }

    // Real Stockfish evaluate.cpp MobilityBonus table, verbatim.
    constexpr RawPair KnightMobility[9] = {
        {-62, -81}, {-53, -56}, {-12, -30}, {-4, -14}, {3, 8}, {13, 15}, {22, 23}, {28, 27}, {33, 33}};
    constexpr RawPair BishopMobility[14] = {
        {-48, -59}, {-20, -23}, {16, -3}, {26, 13}, {38, 24}, {51, 42}, {55, 54},
        {63, 57}, {63, 65}, {68, 73}, {81, 78}, {81, 86}, {91, 88}, {98, 97}};
    constexpr RawPair RookMobility[15] = {
        {-58, -76}, {-27, -18}, {-15, 28}, {-10, 55}, {-5, 69}, {-2, 82}, {9, 112}, {16, 118},
        {30, 132}, {29, 142}, {32, 155}, {38, 165}, {46, 166}, {48, 169}, {58, 171}};
    constexpr RawPair QueenMobility[28] = {
        {-39, -36}, {-21, -15}, {3, 8}, {3, 18}, {14, 34}, {22, 54}, {28, 61}, {41, 73},
        {43, 79}, {48, 92}, {56, 94}, {60, 104}, {60, 113}, {66, 120}, {67, 123}, {70, 126},
        {71, 133}, {73, 136}, {79, 140}, {88, 143}, {88, 148}, {99, 166}, {102, 170}, {102, 175},
        {106, 184}, {109, 191}, {113, 206}, {116, 212}};

    // Rook on a file with no friendly pawn (semi-open) / no pawn at all
    // (fully open). Real Stockfish evaluate.cpp constants.
    constexpr RawPair RookOnFile[2] = {{21, 4}, {47, 25}};

    inline Bitboard fileMask(int file) { return attacks::MASK_FILE[file]; }
    inline Bitboard rankMask(int rank) { return attacks::MASK_RANK[rank]; }

    inline Bitboard pawnAttacksBB(Bitboard pawns, Color side)
    {
      if (side == Color::WHITE)
        return attacks::pawnLeftAttacks<Color::underlying::WHITE>(pawns) |
               attacks::pawnRightAttacks<Color::underlying::WHITE>(pawns);
      return attacks::pawnLeftAttacks<Color::underlying::BLACK>(pawns) |
             attacks::pawnRightAttacks<Color::underlying::BLACK>(pawns);
    }

    Score evaluateSide(const Board &board, Color us)
    {
      Color them = ~us;
      Bitboard occ = board.occ();
      Bitboard ownPawns = board.pieces(PieceType::PAWN, us);
      Bitboard theirPawns = board.pieces(PieceType::PAWN, them);
      Bitboard queensBoth = board.pieces(PieceType::QUEEN);
      Bitboard ownRooks = board.pieces(PieceType::ROOK, us);

      // Own pawns that are blocked (something sits directly in front) or
      // still sitting on their starting/near-starting ranks
      Bitboard blockedOrLow;
      if (us == Color::WHITE)
        blockedOrLow = ownPawns & ((occ >> 8) | rankMask(1) | rankMask(2));
      else
        blockedOrLow = ownPawns & ((occ << 8) | rankMask(6) | rankMask(5));

      Bitboard enemyPawnAttacks = pawnAttacksBB(theirPawns, them);

      Bitboard mobilityArea = ~(blockedOrLow | board.pieces(PieceType::KING, us) |
                                 board.pieces(PieceType::QUEEN, us) | enemyPawnAttacks);

      Score score = 0;

      // Knights
      {
        Bitboard bb = board.pieces(PieceType::KNIGHT, us);
        while (bb)
        {
          Square sq = bb.pop();
          Bitboard atk = attacks::knight(sq);
          int mob = (atk & mobilityArea).count();
          score += scaleScore(KnightMobility[mob]);
        }
      }

      // Bishops (x-ray through any queen, either color)
      {
        Bitboard bb = board.pieces(PieceType::BISHOP, us);
        Bitboard xrayOcc = occ ^ queensBoth;
        while (bb)
        {
          Square sq = bb.pop();
          Bitboard atk = attacks::bishop(sq, xrayOcc);
          int mob = (atk & mobilityArea).count();
          score += scaleScore(BishopMobility[mob]);
        }
      }

      // Rooks (x-ray through any queen and through our own rooks — a
      // rook battery sees all the way down the file/rank)
      {
        Bitboard bb = ownRooks;
        Bitboard xrayOcc = occ ^ queensBoth ^ ownRooks;
        while (bb)
        {
          Square sq = bb.pop();
          Bitboard atk = attacks::rook(sq, xrayOcc);
          int mob = (atk & mobilityArea).count();
          score += scaleScore(RookMobility[mob]);

          int file = sq.index() % 8;
          bool ownPawnOnFile = !(ownPawns & fileMask(file)).empty();
          bool theirPawnOnFile = !(theirPawns & fileMask(file)).empty();
          if (!ownPawnOnFile)
            score += scaleScore(RookOnFile[theirPawnOnFile ? 0 : 1]);
        }
      }

      // Queens
      {
        Bitboard bb = board.pieces(PieceType::QUEEN, us);
        while (bb)
        {
          Square sq = bb.pop();
          Bitboard atk = attacks::queen(sq, occ);
          int mob = (atk & mobilityArea).count();
          score += scaleScore(QueenMobility[mob]);
        }
      }

      return score;
    }

  } // namespace

  Score evaluate(const Board &board)
  {
    return evaluateSide(board, Color::WHITE) - evaluateSide(board, Color::BLACK);
  }

} // namespace eval::mobility

// Threats

namespace eval::threats
{

  namespace
  {

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }
    struct RawPair
    {
      int mg, eg;
    };
    inline Score scaleScore(RawPair p) { return make_score(scale(p.mg), scale(p.eg)); }

    // ThreatByMinor/ThreatByRook[victim piece type]
    constexpr RawPair ThreatByMinor[6] = {{6, 32}, {59, 41}, {79, 56}, {90, 119}, {79, 161}, {0, 0}};
    constexpr RawPair ThreatByRook[6] = {{3, 44}, {38, 71}, {38, 61}, {0, 38}, {51, 38}, {0, 0}};
    const Score Hanging = scaleScore({69, 36});
    const Score ThreatByKing = scaleScore({24, 89});
    const Score ThreatBySafePawn = scaleScore({173, 94});
    const Score ThreatByPawnPush = scaleScore({48, 39});

    inline int idx(PieceType pt) { return static_cast<int>(pt.internal()); }

    struct AttackMap
    {
      Bitboard byType[6]{};
      Bitboard all;
      Bitboard doubled; // attacked by 2 or more of this side's pieces
    };

    AttackMap buildAttackMap(const Board &board, Color side)
    {
      AttackMap m;
      Bitboard occ = board.occ();

      auto add = [&](Bitboard bb)
      {
        m.doubled |= m.all & bb;
        m.all |= bb;
      };

      Bitboard pawns = board.pieces(PieceType::PAWN, side);
      {
        Bitboard u;
        Bitboard bb = pawns;
        while (bb)
          u |= attacks::pawn(side, bb.pop());
        m.byType[idx(PieceType::PAWN)] = u;
        add(u);
      }
      {
        Bitboard u;
        Bitboard bb = board.pieces(PieceType::KNIGHT, side);
        while (bb)
          u |= attacks::knight(bb.pop());
        m.byType[idx(PieceType::KNIGHT)] = u;
        add(u);
      }
      {
        Bitboard u;
        Bitboard bb = board.pieces(PieceType::BISHOP, side);
        while (bb)
          u |= attacks::bishop(bb.pop(), occ);
        m.byType[idx(PieceType::BISHOP)] = u;
        add(u);
      }
      {
        Bitboard u;
        Bitboard bb = board.pieces(PieceType::ROOK, side);
        while (bb)
          u |= attacks::rook(bb.pop(), occ);
        m.byType[idx(PieceType::ROOK)] = u;
        add(u);
      }
      {
        Bitboard u;
        Bitboard bb = board.pieces(PieceType::QUEEN, side);
        while (bb)
          u |= attacks::queen(bb.pop(), occ);
        m.byType[idx(PieceType::QUEEN)] = u;
        add(u);
      }
      {
        Bitboard u = attacks::king(board.kingSq(side));
        m.byType[idx(PieceType::KING)] = u;
        add(u);
      }

      return m;
    }

    inline Bitboard pawnAttacksBB(Bitboard pawns, Color side)
    {
      if (side == Color::WHITE)
        return attacks::pawnLeftAttacks<Color::underlying::WHITE>(pawns) |
               attacks::pawnRightAttacks<Color::underlying::WHITE>(pawns);
      return attacks::pawnLeftAttacks<Color::underlying::BLACK>(pawns) |
             attacks::pawnRightAttacks<Color::underlying::BLACK>(pawns);
    }

    Score evaluateSide(const Board &board, Color us)
    {
      Color them = ~us;
      AttackMap ours = buildAttackMap(board, us);
      AttackMap theirs = buildAttackMap(board, them);

      Bitboard theirAll = board.us(them);
      Bitboard nonPawnEnemies = theirAll & ~board.pieces(PieceType::PAWN, them);

      Bitboard stronglyProtected = theirs.byType[idx(PieceType::PAWN)] | (theirs.doubled & ~ours.doubled);
      Bitboard defended = nonPawnEnemies & stronglyProtected;
      Bitboard weak = theirAll & ~stronglyProtected & ours.all;

      Score score = 0;

      if (!(defended | weak).empty())
      {
        Bitboard minorTargets = (defended | weak) &
                                 (ours.byType[idx(PieceType::KNIGHT)] | ours.byType[idx(PieceType::BISHOP)]);
        {
          Bitboard b = minorTargets;
          while (b)
          {
            Square s = b.pop();
            PieceType victim = board.at(s).type();
            score += scaleScore(ThreatByMinor[idx(victim)]);
          }
        }

        Bitboard rookTargets = weak & ours.byType[idx(PieceType::ROOK)];
        {
          Bitboard b = rookTargets;
          while (b)
          {
            Square s = b.pop();
            PieceType victim = board.at(s).type();
            score += scaleScore(ThreatByRook[idx(victim)]);
          }
        }

        if (!(weak & ours.byType[idx(PieceType::KING)]).empty())
          score += ThreatByKing;

        Bitboard hangingMask = ~theirs.all | (nonPawnEnemies & ours.doubled);
        score += Hanging * (weak & hangingMask).count();
      }

      // Safe pawn attacks on enemy non-pawn pieces.
      Bitboard safe = ~theirs.all | ours.all;
      {
        Bitboard ownPawns = board.pieces(PieceType::PAWN, us) & safe;
        Bitboard attacked = pawnAttacksBB(ownPawns, us) & nonPawnEnemies;
        score += ThreatBySafePawn * attacked.count();
      }

      // Safe pawn pushes (one or two squares) that would attack an enemy
      // non-pawn piece next move.
      {
        Bitboard occ = board.occ();
        int shift = (us == Color::WHITE) ? 8 : -8;
        auto shiftBB = [&](Bitboard bb)
        {
          Bitboard r;
          Bitboard b = bb;
          while (b)
          {
            int i = b.pop() + shift;
            if (i >= 0 && i < 64)
              r.set(i);
          }
          return r;
        };

        Bitboard ownPawns = board.pieces(PieceType::PAWN, us);
        Bitboard oneStep = shiftBB(ownPawns) & ~occ;
        Bitboard thirdRank = (us == Color::WHITE) ? attacks::MASK_RANK[2] : attacks::MASK_RANK[5];
        Bitboard twoStep = shiftBB(oneStep & thirdRank) & ~occ;
        Bitboard pushes = (oneStep | twoStep) & ~theirs.byType[idx(PieceType::PAWN)] & safe;

        Bitboard attacked = pawnAttacksBB(pushes, us) & nonPawnEnemies;
        score += ThreatByPawnPush * attacked.count();
      }

      return score;
    }

  } // namespace

  Score evaluate(const Board &board)
  {
    return evaluateSide(board, Color::WHITE) - evaluateSide(board, Color::BLACK);
  }

} // namespace eval::threats

// Space

namespace eval::space
{

  namespace
  {

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }

    constexpr int SPACE_THRESHOLD = 12222 * 100 / 128;

    // The four central files (C, D, E, F).
    inline Bitboard centerFilesMask()
    {
      return attacks::MASK_FILE[2] | attacks::MASK_FILE[3] | attacks::MASK_FILE[4] | attacks::MASK_FILE[5];
    }

    inline Bitboard rankMask(int rank) { return attacks::MASK_RANK[rank]; }

    inline Bitboard pawnAttacksBB(Bitboard pawns, Color side)
    {
      if (side == Color::WHITE)
        return attacks::pawnLeftAttacks<Color::underlying::WHITE>(pawns) |
               attacks::pawnRightAttacks<Color::underlying::WHITE>(pawns);
      return attacks::pawnLeftAttacks<Color::underlying::BLACK>(pawns) |
             attacks::pawnRightAttacks<Color::underlying::BLACK>(pawns);
    }

    int nonPawnMaterialTotal(const Board &board)
    {
      int npm = 0;
      npm += board.pieces(PieceType::KNIGHT).count() * KNIGHT_VALUE_MG;
      npm += board.pieces(PieceType::BISHOP).count() * BISHOP_VALUE_MG;
      npm += board.pieces(PieceType::ROOK).count() * ROOK_VALUE_MG;
      npm += board.pieces(PieceType::QUEEN).count() * QUEEN_VALUE_MG;
      return npm;
    }

    Score evaluateSide(const Board &board, Color us)
    {
      Color them = ~us;
      Bitboard spaceMask = centerFilesMask() &
                            (us == Color::WHITE ? (rankMask(1) | rankMask(2) | rankMask(3))
                                                 : (rankMask(6) | rankMask(5) | rankMask(4)));

      Bitboard theirPawnAttacks = pawnAttacksBB(board.pieces(PieceType::PAWN, them), them);
      Bitboard safe = spaceMask & ~board.pieces(PieceType::PAWN, us) & ~theirPawnAttacks;

      // Squares up to 2 squares behind a friendly pawn (toward home).
      Bitboard ownPawns = board.pieces(PieceType::PAWN, us);
      int back = (us == Color::WHITE) ? -8 : 8;
      auto shiftBack = [&](Bitboard bb)
      {
        Bitboard r;
        Bitboard b = bb;
        while (b)
        {
          int i = b.pop() + back;
          if (i >= 0 && i < 64)
            r.set(i);
        }
        return r;
      };
      Bitboard behind = ownPawns;
      behind |= shiftBack(behind);
      behind |= shiftBack(shiftBack(ownPawns));

      // "Attacked by any enemy piece" — a fresh union, kept local to this
      // section rather than shared via a cross-section attack-map struct.
      Bitboard occ = board.occ();
      Bitboard theirAttacks = theirPawnAttacks;
      {
        Bitboard bb = board.pieces(PieceType::KNIGHT, them);
        while (bb)
          theirAttacks |= attacks::knight(bb.pop());
      }
      {
        Bitboard bb = board.pieces(PieceType::BISHOP, them);
        while (bb)
          theirAttacks |= attacks::bishop(bb.pop(), occ);
      }
      {
        Bitboard bb = board.pieces(PieceType::ROOK, them);
        while (bb)
          theirAttacks |= attacks::rook(bb.pop(), occ);
      }
      {
        Bitboard bb = board.pieces(PieceType::QUEEN, them);
        while (bb)
          theirAttacks |= attacks::queen(bb.pop(), occ);
      }
      theirAttacks |= attacks::king(board.kingSq(them));

      int bonus = safe.count() + (behind & safe & ~theirAttacks).count();
      int weight = board.us(us).count() - 1;

      return make_score(scale(bonus * weight * weight / 16), 0);
    }

  } // namespace

  Score evaluate(const Board &board)
  {
    if (nonPawnMaterialTotal(board) < SPACE_THRESHOLD)
      return SCORE_ZERO;

    return evaluateSide(board, Color::WHITE) - evaluateSide(board, Color::BLACK);
  }

} // namespace eval::space

// Endgame scale factor

namespace eval::scale_factor
{

  constexpr int SCALE_FACTOR_NORMAL = 64;

  namespace
  {

    bool oppositeBishops(const Board &board)
    {
      Bitboard whiteB = board.pieces(PieceType::BISHOP, Color::WHITE);
      Bitboard blackB = board.pieces(PieceType::BISHOP, Color::BLACK);
      if (whiteB.count() != 1 || blackB.count() != 1)
        return false;
      return !Square::same_color(whiteB.lsb(), blackB.lsb());
    }

    int nonPawnMaterialFor(const Board &board, Color side)
    {
      int npm = 0;
      npm += board.pieces(PieceType::KNIGHT, side).count() * KNIGHT_VALUE_MG;
      npm += board.pieces(PieceType::BISHOP, side).count() * BISHOP_VALUE_MG;
      npm += board.pieces(PieceType::ROOK, side).count() * ROOK_VALUE_MG;
      npm += board.pieces(PieceType::QUEEN, side).count() * QUEEN_VALUE_MG;
      return npm;
    }

  } // namespace

  // Returns a value in [0, 64], where 64 = "no scaling, use the endgame
  // score as-is" and lower values scale the endgame term down
  // proportionally (0 = treat as a dead draw).
  int compute(const Board &board, int egScore)
  {
    Color strongSide = egScore > 0 ? Color::WHITE : Color::BLACK;
    Color weakSide = ~strongSide;

    bool oppBishops = oppositeBishops(board);
    int npmStrong = nonPawnMaterialFor(board, strongSide);
    int npmWeak = nonPawnMaterialFor(board, weakSide);
    int pawnsStrong = board.pieces(PieceType::PAWN, strongSide).count();

    bool smallPawnlessEdge = pawnsStrong == 0 && (npmStrong - npmWeak) <= BISHOP_VALUE_MG;

    if (!oppBishops && !smallPawnlessEdge)
      return SCALE_FACTOR_NORMAL;

    int sf;
    if (oppBishops && npmStrong + npmWeak == 2 * BISHOP_VALUE_MG)
      sf = 22;
    else
      sf = std::min(SCALE_FACTOR_NORMAL, 36 + (oppBishops ? 2 : 7) * pawnsStrong);

    // Fifty-move-rule creep: as the position approaches a forced draw by
    // the fifty-move rule, scale the endgame term down further. Only
    // applied in the already-drawish-tending cases above
    int halfmoveClock = static_cast<int>(board.halfMoveClock());
    sf = std::max(0, sf - (halfmoveClock - 12) / 4);

    return sf;
  }

} // namespace eval::scale_factor

// Top-level orchestrator: sums every term above, applies initiative(),
// tapers by game phase, applies the endgame scale factor, adds tempo.

namespace eval
{

  namespace
  {

    inline int scale(int stockfishUnits) { return stockfishUnits * 100 / 128; }

    constexpr int TEMPO = 28 * 100 / 128;

    // Defensive clamp on the final centipawn score. Score is no longer a
    // bit-packed 16-bit-halves int (see eval.h), so ordinary positions
    // can no longer wrap around and corrupt themselves — but search.cpp
    // encodes mate scores as values near +-MATE_SCORE (31000) and relies
    // on "a normal eval score is nowhere near that range" to tell the two
    // apart (e.g. the RFP/razoring near-mate guards in search::pruning).
    // Clamping here guarantees a static eval can never be confused for a
    // mate score even in freak material configurations (many promoted
    // queens on the board, etc.) that don't come up in real games but
    // are exactly the kind of input a fuzzer or a hand-edited FEN can
    // produce.
    constexpr int EVAL_CLAMP = 20000;

    // Sum of material + piece-square bonus for every piece on the board,
    // from White's perspective
    Score psqtScore(const Board &board)
    {
      Score score = 0;
      for (int i = 0; i < 64; ++i)
      {
        Square sq(i);
        Piece p = board.at(sq);
        if (p == Piece::NONE)
          continue;

        Score s = pst::pieceSquareScore(p.type(), p.color(), sq);
        score += (p.color() == Color::WHITE) ? s : -s;
      }
      return score;
    }

    // a second-order
    // bonus/malus that rewards the side who can create winning chances
    // (passed pawns, pawns on both flanks, an infiltrated king) and
    // penalizes positions that look drawish ("almost unwinnable"), capped
    // so it can never flip the sign of either the mg or eg score.
    Score initiative(const Board &board, const pawn_structure::Info &pawnInfo, Score score)
    {
      Square wk = board.kingSq(Color::WHITE);
      Square bk = board.kingSq(Color::BLACK);

      int wFile = wk.index() % 8, wRank = wk.index() / 8;
      int bFile = bk.index() % 8, bRank = bk.index() / 8;

      int outflanking = std::abs(wFile - bFile) - std::abs(wRank - bRank);
      bool infiltration = wRank > 3 || bRank < 4;

      Bitboard allPawns = board.pieces(PieceType::PAWN);
      Bitboard queenSide = attacks::MASK_FILE[0] | attacks::MASK_FILE[1] | attacks::MASK_FILE[2] | attacks::MASK_FILE[3];
      Bitboard kingSide = attacks::MASK_FILE[4] | attacks::MASK_FILE[5] | attacks::MASK_FILE[6] | attacks::MASK_FILE[7];
      bool pawnsOnBothFlanks = !(allPawns & queenSide).empty() && !(allPawns & kingSide).empty();

      int passedCount = (pawnInfo.passed[int(Color::WHITE)] | pawnInfo.passed[int(Color::BLACK)]).count();
      bool almostUnwinnable = passedCount == 0 && outflanking < 0 && !pawnsOnBothFlanks;

      int nonPawnMaterial = board.pieces(PieceType::KNIGHT).count() + board.pieces(PieceType::BISHOP).count() +
                            board.pieces(PieceType::ROOK).count() + board.pieces(PieceType::QUEEN).count();

      int complexity = scale(9) * passedCount + scale(11) * allPawns.count() + scale(9) * outflanking +
                        scale(12) * int(infiltration) + scale(21) * int(pawnsOnBothFlanks) +
                        scale(51) * int(nonPawnMaterial == 0) - scale(43) * int(almostUnwinnable) - scale(100);

      int mg = mg_value(score), eg = eg_value(score);
      auto sign = [](int x)
      { return (x > 0) - (x < 0); };

      int u = sign(mg) * std::max(std::min(complexity + scale(50), 0), -std::abs(mg));
      int v = sign(eg) * std::max(complexity, -std::abs(eg));

      return make_score(u, v);
    }

  } // namespace

  int materialValue(PieceType pt)
  {
    switch (pt.internal())
    {
    case PieceType::underlying::PAWN:
      return PAWN_VALUE;
    case PieceType::underlying::KNIGHT:
      return KNIGHT_VALUE;
    case PieceType::underlying::BISHOP:
      return BISHOP_VALUE;
    case PieceType::underlying::ROOK:
      return ROOK_VALUE;
    case PieceType::underlying::QUEEN:
      return QUEEN_VALUE;
    default:
      return KING_VALUE;
    }
  }

  int gamePhase(const Board &board)
  {
    return material::gamePhase(board);
  }

  int evaluate(const Board &board)
  {
    pawn_structure::Info pawnInfo = pawn_structure::evaluate(board);

    Score score = psqtScore(board) + material::imbalance(board) + pawnInfo.score +
                  mobility::evaluate(board) + king_safety::evaluate(board) +
                  threats::evaluate(board) + space::evaluate(board);

    score += initiative(board, pawnInfo, score);

    int phase = material::gamePhase(board); // 0..256
    int sf = scale_factor::compute(board, eg_value(score));

    int v = (mg_value(score) * phase +
             eg_value(score) * (256 - phase) * sf / scale_factor::SCALE_FACTOR_NORMAL) /
            256;

    v = (board.sideToMove() == Color::WHITE) ? v : -v;
    v += TEMPO;

    // This is a no-op for every position a real game can reach; 
    // it only bites in constructed/fuzzed inputs.
    v = std::clamp(v, -EVAL_CLAMP, EVAL_CLAMP);

    return v;
  }

} // namespace eval
