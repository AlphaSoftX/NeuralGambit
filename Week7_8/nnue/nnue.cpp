#include "nnue.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace chess;

namespace nnue
{

  namespace
  {

#pragma pack(push, 1)
    struct FileHeader
    {
      char magic[4];
      uint32_t version;
      uint32_t ft_size;
      uint32_t hidden_size;
      uint32_t num_features;
      uint32_t padding_index;
      int32_t qa;
      int32_t l1_weight_scale;
      int32_t l2_weight_scale;
      int32_t out_weight_scale;
      float sigmoid_scale;
      int32_t output_cp_clamp;
      int32_t epoch;
      float val_loss;
      uint32_t seed;
    };

#pragma pack(pop)
    static_assert(sizeof(FileHeader) == 60, "FileHeader must exactly match export.py's HEADER_STRUCT layout");

    constexpr uint32_t EXPECTED_FORMAT_VERSION = 3;

    // Mirrors search.h's MATE_SCORE (31000) and MAX_PLY (128)
    constexpr int SEARCH_MATE_SCORE = 31000;
    constexpr int SEARCH_MAX_PLY = 128;

    struct Network
    {
      bool loaded = false;

      int32_t qa = 0;
      int32_t l1Scale = 1, l2Scale = 1, outScale = 1;
      float sigmoidScale = 400.0f;
      int32_t outputCpClamp = 30000;

      std::vector<int16_t> ftWeight;
      std::vector<int16_t> l1Weight;
      std::vector<int32_t> l1Bias;
      std::vector<int16_t> l2Weight;
      std::vector<int32_t> l2Bias;
      std::vector<int16_t> outWeight;
      int32_t outBias = 0;
    };

    Network g_net;

    std::atomic<bool> g_enabled{true};

    inline int64_t rescaleRound(int64_t raw, int32_t scale)
    {
      if (raw >= 0)
        return (raw + scale / 2) / scale;
      return -((-raw + scale / 2) / scale);
    }

  } // namespace

  bool load(const std::string &path)
  {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
      return false;

    FileHeader hdr{};
    if (!f.read(reinterpret_cast<char *>(&hdr), sizeof(hdr)))
      return false;

    if (std::memcmp(hdr.magic, "MNUE", 4) != 0)
      return false;
    if (hdr.version != EXPECTED_FORMAT_VERSION)
      return false;
    if (hdr.ft_size != FT_SIZE || hdr.hidden_size != HIDDEN_SIZE ||
        hdr.num_features != static_cast<uint32_t>(features::NUM_FEATURES) ||
        hdr.padding_index != static_cast<uint32_t>(features::PADDING_INDEX))
      return false;

    if (hdr.output_cp_clamp <= 0 ||
        hdr.output_cp_clamp >= SEARCH_MATE_SCORE - SEARCH_MAX_PLY)
      return false;

    Network net;
    net.qa = hdr.qa;
    net.l1Scale = hdr.l1_weight_scale;
    net.l2Scale = hdr.l2_weight_scale;
    net.outScale = hdr.out_weight_scale;
    net.sigmoidScale = hdr.sigmoid_scale;
    net.outputCpClamp = hdr.output_cp_clamp;

    if (net.qa <= 0 || net.qa > 32767 || net.l1Scale <= 0 || net.l2Scale <= 0 || net.outScale <= 0)
      return false;

    const size_t numFtRows = static_cast<size_t>(features::NUM_FEATURES) + 1;
    net.ftWeight.resize(numFtRows * FT_SIZE);
    net.l1Weight.resize(static_cast<size_t>(HIDDEN_SIZE) * (FT_SIZE * 2));
    net.l1Bias.resize(HIDDEN_SIZE);
    net.l2Weight.resize(static_cast<size_t>(HIDDEN_SIZE) * HIDDEN_SIZE);
    net.l2Bias.resize(HIDDEN_SIZE);
    net.outWeight.resize(HIDDEN_SIZE);

    auto readInto = [&](void *dst, size_t bytes) -> bool
    {
      return static_cast<bool>(f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(bytes)));
    };

    if (!readInto(net.ftWeight.data(), net.ftWeight.size() * sizeof(int16_t)))
      return false;
    if (!readInto(net.l1Weight.data(), net.l1Weight.size() * sizeof(int16_t)))
      return false;
    if (!readInto(net.l1Bias.data(), net.l1Bias.size() * sizeof(int32_t)))
      return false;
    if (!readInto(net.l2Weight.data(), net.l2Weight.size() * sizeof(int16_t)))
      return false;
    if (!readInto(net.l2Bias.data(), net.l2Bias.size() * sizeof(int32_t)))
      return false;
    if (!readInto(net.outWeight.data(), net.outWeight.size() * sizeof(int16_t)))
      return false;
    if (!readInto(&net.outBias, sizeof(int32_t)))
      return false;

    net.loaded = true;
    g_net = std::move(net); // atomic swap — previous state is fully replaced only on total success
    return true;
  }

  bool isLoaded() { return g_net.loaded; }

  void unload() { g_net = Network{}; }

  void setEnabled(bool enabled) { g_enabled.store(enabled); }

  bool isEnabled() { return g_enabled.load(); }

  void refreshOne(const Board &board, Color perspective, Accumulator &acc)
  {
    features::FeatureList fl = features::buildFeatures(board, perspective);
    acc.v.fill(0);
    for (int i = 0; i < fl.count; ++i)
    {
      const int16_t *row = &g_net.ftWeight[static_cast<size_t>(fl.idx[i]) * FT_SIZE];
#if defined(__AVX2__)
      // vector calculation instead of for loop - fast
      for (int j = 0; j < FT_SIZE; j += 8)
      {
        __m128i w16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + j));
        __m256i w32 = _mm256_cvtepi16_epi32(w16);
        __m256i accv = _mm256_load_si256(reinterpret_cast<__m256i *>(&acc.v[j]));
        accv = _mm256_add_epi32(accv, w32);
        _mm256_store_si256(reinterpret_cast<__m256i *>(&acc.v[j]), accv);
      }
#else
      for (int j = 0; j < FT_SIZE; ++j)
        acc.v[j] += row[j];
#endif
    }
  }

  void refresh(const Board &board, PerspectiveAccumulators &acc)
  {
    refreshOne(board, Color::WHITE, acc.white);
    refreshOne(board, Color::BLACK, acc.black);
  }

  namespace
  {
#if defined(__AVX2__)
    inline void applyRow(Accumulator &acc, int featureIdx, int sign)
    {
      if (featureIdx < 0)
        return;
      const int16_t *row = &g_net.ftWeight[static_cast<size_t>(featureIdx) * FT_SIZE];
      static_assert(FT_SIZE % 8 == 0, "AVX2 path assumes FT_SIZE is a multiple of 8");
      for (int j = 0; j < FT_SIZE; j += 8)
      {
        __m128i w16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + j));
        __m256i w32 = _mm256_cvtepi16_epi32(w16);
        __m256i accv = _mm256_load_si256(reinterpret_cast<__m256i *>(&acc.v[j]));
        accv = (sign > 0) ? _mm256_add_epi32(accv, w32) : _mm256_sub_epi32(accv, w32);
        _mm256_store_si256(reinterpret_cast<__m256i *>(&acc.v[j]), accv);
      }
    }
#else
    inline void applyRow(Accumulator &acc, int featureIdx, int sign)
    {
      if (featureIdx < 0)
        return;
      const int16_t *row = &g_net.ftWeight[static_cast<size_t>(featureIdx) * FT_SIZE];
      if (sign > 0)
        for (int j = 0; j < FT_SIZE; ++j)
          acc.v[j] += row[j];
      else
        for (int j = 0; j < FT_SIZE; ++j)
          acc.v[j] -= row[j];
    }
#endif
  } // namespace

  void addPiece(PerspectiveAccumulators &acc, int whiteKingSq, int blackKingSqMirrored,
                Color pieceColor, PieceType pieceType, int pieceSq)
  {
    applyRow(acc.white, features::featureIndex(Color::WHITE, whiteKingSq, pieceColor, pieceType, pieceSq), +1);
    applyRow(acc.black, features::featureIndex(Color::BLACK, blackKingSqMirrored, pieceColor, pieceType, pieceSq), +1);
  }

  void removePiece(PerspectiveAccumulators &acc, int whiteKingSq, int blackKingSqMirrored,
                   Color pieceColor, PieceType pieceType, int pieceSq)
  {
    applyRow(acc.white, features::featureIndex(Color::WHITE, whiteKingSq, pieceColor, pieceType, pieceSq), -1);
    applyRow(acc.black, features::featureIndex(Color::BLACK, blackKingSqMirrored, pieceColor, pieceType, pieceSq), -1);
  }

#if defined(__AVX2__)
  namespace
  {
    inline int64_t hsum8x32_to_i64(__m256i v)
    {
      __m128i lo128 = _mm256_castsi256_si128(v);
      __m128i hi128 = _mm256_extracti128_si256(v, 1);
      __m256i lo64 = _mm256_cvtepi32_epi64(lo128);
      __m256i hi64 = _mm256_cvtepi32_epi64(hi128);
      __m256i sum64 = _mm256_add_epi64(lo64, hi64);
      __m128i sumLo = _mm256_castsi256_si128(sum64);
      __m128i sumHi = _mm256_extracti128_si256(sum64, 1);
      __m128i sum2 = _mm_add_epi64(sumLo, sumHi);
      return _mm_cvtsi128_si64(sum2) + _mm_extract_epi64(sum2, 1);
    }

    inline int64_t dotI16(const int16_t *a, const int16_t *b, int len)
    {
      __m256i acc = _mm256_setzero_si256();
      for (int i = 0; i < len; i += 16)
      {
        __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(av, bv));
      }
      return hsum8x32_to_i64(acc);
    }
  } // namespace
#endif

  int evaluate(const PerspectiveAccumulators &acc, Color sideToMove)
  {
    const Accumulator &own = (sideToMove == Color::WHITE) ? acc.white : acc.black;
    const Accumulator &opp = (sideToMove == Color::WHITE) ? acc.black : acc.white;

#if defined(__AVX2__)
    alignas(32) int16_t x[FT_SIZE * 2];
    for (int i = 0; i < FT_SIZE; ++i)
      x[i] = static_cast<int16_t>(std::clamp<int32_t>(own.v[i], 0, g_net.qa));
    for (int i = 0; i < FT_SIZE; ++i)
      x[FT_SIZE + i] = static_cast<int16_t>(std::clamp<int32_t>(opp.v[i], 0, g_net.qa));

    std::array<int32_t, HIDDEN_SIZE> h1{};
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
      const int16_t *wRow = &g_net.l1Weight[static_cast<size_t>(j) * (FT_SIZE * 2)];
      int64_t raw = g_net.l1Bias[j] + dotI16(x, wRow, FT_SIZE * 2);
      int64_t rescaled = rescaleRound(raw, g_net.l1Scale);
      h1[j] = static_cast<int32_t>(std::clamp<int64_t>(rescaled, 0, g_net.qa));
    }

    alignas(32) int16_t h1i16[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
      h1i16[i] = static_cast<int16_t>(h1[i]); // safe: h1[i] in [0, qa], qa fits int16

    std::array<int32_t, HIDDEN_SIZE> h2{};
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
      const int16_t *wRow = &g_net.l2Weight[static_cast<size_t>(j) * HIDDEN_SIZE];
      int64_t raw = g_net.l2Bias[j] + dotI16(h1i16, wRow, HIDDEN_SIZE);
      int64_t rescaled = rescaleRound(raw, g_net.l2Scale);
      h2[j] = static_cast<int32_t>(std::clamp<int64_t>(rescaled, 0, g_net.qa));
    }
#else
    std::array<int32_t, FT_SIZE * 2> x{};
    for (int i = 0; i < FT_SIZE; ++i)
      x[i] = std::clamp<int32_t>(own.v[i], 0, g_net.qa);
    for (int i = 0; i < FT_SIZE; ++i)
      x[FT_SIZE + i] = std::clamp<int32_t>(opp.v[i], 0, g_net.qa);

    std::array<int32_t, HIDDEN_SIZE> h1{};
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
      int64_t raw = g_net.l1Bias[j];
      const int16_t *wRow = &g_net.l1Weight[static_cast<size_t>(j) * (FT_SIZE * 2)];
      for (int i = 0; i < FT_SIZE * 2; ++i)
        raw += static_cast<int64_t>(x[i]) * wRow[i];
      int64_t rescaled = rescaleRound(raw, g_net.l1Scale);
      h1[j] = static_cast<int32_t>(std::clamp<int64_t>(rescaled, 0, g_net.qa));
    }

    std::array<int32_t, HIDDEN_SIZE> h2{};
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
      int64_t raw = g_net.l2Bias[j];
      const int16_t *wRow = &g_net.l2Weight[static_cast<size_t>(j) * HIDDEN_SIZE];
      for (int i = 0; i < HIDDEN_SIZE; ++i)
        raw += static_cast<int64_t>(h1[i]) * wRow[i];
      int64_t rescaled = rescaleRound(raw, g_net.l2Scale);
      h2[j] = static_cast<int32_t>(std::clamp<int64_t>(rescaled, 0, g_net.qa));
    }
#endif
    int64_t rawOut = g_net.outBias;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
      rawOut += static_cast<int64_t>(h2[i]) * g_net.outWeight[i];

    float floatLogit = static_cast<float>(rawOut) / (static_cast<float>(g_net.qa) * static_cast<float>(g_net.outScale));
    float scoreCp = floatLogit * g_net.sigmoidScale;

    if (scoreCp > static_cast<float>(g_net.outputCpClamp))
      scoreCp = static_cast<float>(g_net.outputCpClamp);
    if (scoreCp < -static_cast<float>(g_net.outputCpClamp))
      scoreCp = -static_cast<float>(g_net.outputCpClamp);

    return static_cast<int>(scoreCp);
  }

} // namespace nnue
