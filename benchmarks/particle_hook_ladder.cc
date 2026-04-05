#include <benchmark/benchmark.h>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "multimap.h"

namespace bmi = boost::multi_index;

// ---------------------------------------------------------------------------
// Domain object
// ---------------------------------------------------------------------------
struct Particle {
  uint64_t id;
  double x;
  double y;
  double m;
  double vx{}, vy{};
  double fx{}, fy{};
  double charge{};
  uint32_t flags{};
  uint32_t zone{};
  char name[32]{"a default name"};

  Particle(uint64_t id_, double x_, double y_, double m_)
      : id(id_), x(x_), y(y_), m(m_) {}
};

constexpr size_t kParticleSize = sizeof(Particle);

struct IdHash {
  size_t operator()(uint64_t id) const { return std::hash<uint64_t>{}(id); }
};
struct IdEqual {
  bool operator()(uint64_t a, uint64_t b) const { return a == b; }
};

static constexpr size_t kN = 100'000;
static constexpr size_t kBuckets = 131071;

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------
struct TestData {
  std::vector<uint64_t> ids;
  std::vector<double> xs, ys, ms;
  std::vector<uint64_t> lookup_ids;

  TestData() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> coord(-1000.0, 1000.0);
    std::uniform_real_distribution<double> mass(1.0, 10.0);

    ids.resize(kN);
    xs.resize(kN);
    ys.resize(kN);
    ms.resize(kN);

    for (size_t i = 0; i < kN; ++i) {
      ids[i] = i + 1;
      xs[i] = coord(rng);
      ys[i] = coord(rng);
      ms[i] = std::round(mass(rng));
    }

    lookup_ids.resize(256);
    std::uniform_int_distribution<size_t> pick(0, kN - 1);
    for (auto &id : lookup_ids) id = ids[pick(rng)];
  }
};

static const TestData kData;

// ===========================================================================
// PRIMARY LADDERS
// Keep id unordered as primary index 0 in every rung
// ===========================================================================

using PrimaryOnlyMM =
    fastmm::FixedSizeMultiMap<Particle, kN,
                              fastmm::Unordered<fastmm::KeyFrom<&Particle::id>,
                                                IdHash, IdEqual, kBuckets>>;

using PrimaryListMM =
    fastmm::FixedSizeMultiMap<Particle, kN,
                              fastmm::Unordered<fastmm::KeyFrom<&Particle::id>,
                                                IdHash, IdEqual, kBuckets>,
                              fastmm::List>;

using PrimaryXMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>>;

using PrimaryXYMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>>;

using PrimaryXYListMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>,
    fastmm::List>;

using FullPrimaryMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>,
    fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>, std::less<double>>,
    fastmm::List>;

struct ById {};
struct ByX {};
struct ByY {};
struct ByM {};
struct BySeq {};

using PrimaryOnlyBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<bmi::hashed_unique<
        bmi::tag<ById>, bmi::member<Particle, uint64_t, &Particle::id>>>,
    boost::fast_pool_allocator<Particle>>;

using PrimaryListBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::sequenced<bmi::tag<BySeq>>>,
    boost::fast_pool_allocator<Particle>>;

using PrimaryXBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>>,
    boost::fast_pool_allocator<Particle>>;

using PrimaryXYBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::ordered_unique<bmi::tag<ByY>,
                            bmi::member<Particle, double, &Particle::y>>>,
    boost::fast_pool_allocator<Particle>>;

using PrimaryXYListBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::ordered_unique<bmi::tag<ByY>,
                            bmi::member<Particle, double, &Particle::y>>,
        bmi::sequenced<bmi::tag<BySeq>>>,
    boost::fast_pool_allocator<Particle>>;

using FullPrimaryBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::ordered_unique<bmi::tag<ByY>,
                            bmi::member<Particle, double, &Particle::y>>,
        bmi::ordered_non_unique<bmi::tag<ByM>,
                                bmi::member<Particle, double, &Particle::m>>,
        bmi::sequenced<bmi::tag<BySeq>>>,
    boost::fast_pool_allocator<Particle>>;

// ===========================================================================
// ORDERED LADDERS
// Keep x ordered as primary index 0 in every rung
// ===========================================================================

using OrderedOnlyMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>>;

using OrderedListMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::List>;

using OrderedHashMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>>;

using OrderedHashYMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>>;

using OrderedHashYListMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>,
    fastmm::List>;

using FullOrderedMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>,
    fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>, std::less<double>>,
    fastmm::List>;

using OrderedOnlyBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<bmi::ordered_unique<
        bmi::tag<ByX>, bmi::member<Particle, double, &Particle::x>>>,
    boost::fast_pool_allocator<Particle>>;

using OrderedListBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::sequenced<bmi::tag<BySeq>>>,
    boost::fast_pool_allocator<Particle>>;

using OrderedHashBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>>,
    boost::fast_pool_allocator<Particle>>;

using OrderedHashYBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByY>,
                            bmi::member<Particle, double, &Particle::y>>>,
    boost::fast_pool_allocator<Particle>>;

using OrderedHashYListBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByY>,
                            bmi::member<Particle, double, &Particle::y>>,
        bmi::sequenced<bmi::tag<BySeq>>>,
    boost::fast_pool_allocator<Particle>>;

using FullOrderedBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<
        bmi::ordered_unique<bmi::tag<ByX>,
                            bmi::member<Particle, double, &Particle::x>>,
        bmi::hashed_unique<bmi::tag<ById>,
                           bmi::member<Particle, uint64_t, &Particle::id>>,
        bmi::ordered_unique<bmi::tag<ByY>,
                            bmi::member<Particle, double, &Particle::y>>,
        bmi::ordered_non_unique<bmi::tag<ByM>,
                                bmi::member<Particle, double, &Particle::m>>,
        bmi::sequenced<bmi::tag<BySeq>>>,
    boost::fast_pool_allocator<Particle>>;

// ===========================================================================
// Build helpers
// ===========================================================================
template <class MM>
static auto make_mm() {
  auto m = std::make_unique<MM>();
  for (size_t i = 0; i < kN; ++i) {
    auto it = m->template insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                       kData.ms[i]);
    if (it == m->cend()) {
      std::cerr << "MM build failed at i=" << i << "\n";
      std::abort();
    }
  }
  return m;
}

template <class BMI>
static BMI make_bmi() {
  BMI m;
  for (size_t i = 0; i < kN; ++i) {
    auto [it, ok] =
        m.insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
    if (!ok) {
      std::cerr << "BMI build failed at i=" << i << "\n";
      std::abort();
    }
  }
  return m;
}

// ===========================================================================
// Primary ladder: iterate primary index 0
// ===========================================================================
template <class MM>
static void BM_Primary_MM_Iterate_Impl(benchmark::State &state) {
  auto m = make_mm<MM>();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &p : *m) sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}

template <class BMI>
static void BM_Primary_BMI_Iterate_Impl(benchmark::State &state) {
  auto m = make_bmi<BMI>();
  auto &idx = m.template get<ById>();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &p : idx) sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}

// MM primary ladder
static void BM_PrimaryOnly_MM_Iterate(benchmark::State &state) {
  BM_Primary_MM_Iterate_Impl<PrimaryOnlyMM>(state);
}
BENCHMARK(BM_PrimaryOnly_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryList_MM_Iterate(benchmark::State &state) {
  BM_Primary_MM_Iterate_Impl<PrimaryListMM>(state);
}
BENCHMARK(BM_PrimaryList_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryX_MM_Iterate(benchmark::State &state) {
  BM_Primary_MM_Iterate_Impl<PrimaryXMM>(state);
}
BENCHMARK(BM_PrimaryX_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryXY_MM_Iterate(benchmark::State &state) {
  BM_Primary_MM_Iterate_Impl<PrimaryXYMM>(state);
}
BENCHMARK(BM_PrimaryXY_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryXYList_MM_Iterate(benchmark::State &state) {
  BM_Primary_MM_Iterate_Impl<PrimaryXYListMM>(state);
}
BENCHMARK(BM_PrimaryXYList_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_FullPrimary_MM_Iterate(benchmark::State &state) {
  BM_Primary_MM_Iterate_Impl<FullPrimaryMM>(state);
}
BENCHMARK(BM_FullPrimary_MM_Iterate)->Unit(benchmark::kMicrosecond);

// BMI primary ladder
static void BM_PrimaryOnly_BMI_Iterate(benchmark::State &state) {
  BM_Primary_BMI_Iterate_Impl<PrimaryOnlyBMI>(state);
}
BENCHMARK(BM_PrimaryOnly_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryList_BMI_Iterate(benchmark::State &state) {
  BM_Primary_BMI_Iterate_Impl<PrimaryListBMI>(state);
}
BENCHMARK(BM_PrimaryList_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryX_BMI_Iterate(benchmark::State &state) {
  BM_Primary_BMI_Iterate_Impl<PrimaryXBMI>(state);
}
BENCHMARK(BM_PrimaryX_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryXY_BMI_Iterate(benchmark::State &state) {
  BM_Primary_BMI_Iterate_Impl<PrimaryXYBMI>(state);
}
BENCHMARK(BM_PrimaryXY_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryXYList_BMI_Iterate(benchmark::State &state) {
  BM_Primary_BMI_Iterate_Impl<PrimaryXYListBMI>(state);
}
BENCHMARK(BM_PrimaryXYList_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_FullPrimary_BMI_Iterate(benchmark::State &state) {
  BM_Primary_BMI_Iterate_Impl<FullPrimaryBMI>(state);
}
BENCHMARK(BM_FullPrimary_BMI_Iterate)->Unit(benchmark::kMicrosecond);

// ===========================================================================
// Ordered ladder: iterate ordered index 0
// ===========================================================================
template <class MM>
static void BM_Ordered_MM_Iterate_Impl(benchmark::State &state) {
  auto m = make_mm<MM>();
  auto &idx = m->template get<0>();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &p : idx) sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}

template <class BMI>
static void BM_Ordered_BMI_Iterate_Impl(benchmark::State &state) {
  auto m = make_bmi<BMI>();
  auto &idx = m.template get<ByX>();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &p : idx) sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}

// MM ordered ladder
static void BM_OrderedOnly_MM_Iterate(benchmark::State &state) {
  BM_Ordered_MM_Iterate_Impl<OrderedOnlyMM>(state);
}
BENCHMARK(BM_OrderedOnly_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedList_MM_Iterate(benchmark::State &state) {
  BM_Ordered_MM_Iterate_Impl<OrderedListMM>(state);
}
BENCHMARK(BM_OrderedList_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedHash_MM_Iterate(benchmark::State &state) {
  BM_Ordered_MM_Iterate_Impl<OrderedHashMM>(state);
}
BENCHMARK(BM_OrderedHash_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedHashY_MM_Iterate(benchmark::State &state) {
  BM_Ordered_MM_Iterate_Impl<OrderedHashYMM>(state);
}
BENCHMARK(BM_OrderedHashY_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedHashYList_MM_Iterate(benchmark::State &state) {
  BM_Ordered_MM_Iterate_Impl<OrderedHashYListMM>(state);
}
BENCHMARK(BM_OrderedHashYList_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_FullOrdered_MM_Iterate(benchmark::State &state) {
  BM_Ordered_MM_Iterate_Impl<FullOrderedMM>(state);
}
BENCHMARK(BM_FullOrdered_MM_Iterate)->Unit(benchmark::kMicrosecond);

// BMI ordered ladder
static void BM_OrderedOnly_BMI_Iterate(benchmark::State &state) {
  BM_Ordered_BMI_Iterate_Impl<OrderedOnlyBMI>(state);
}
BENCHMARK(BM_OrderedOnly_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedList_BMI_Iterate(benchmark::State &state) {
  BM_Ordered_BMI_Iterate_Impl<OrderedListBMI>(state);
}
BENCHMARK(BM_OrderedList_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedHash_BMI_Iterate(benchmark::State &state) {
  BM_Ordered_BMI_Iterate_Impl<OrderedHashBMI>(state);
}
BENCHMARK(BM_OrderedHash_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedHashY_BMI_Iterate(benchmark::State &state) {
  BM_Ordered_BMI_Iterate_Impl<OrderedHashYBMI>(state);
}
BENCHMARK(BM_OrderedHashY_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedHashYList_BMI_Iterate(benchmark::State &state) {
  BM_Ordered_BMI_Iterate_Impl<OrderedHashYListBMI>(state);
}
BENCHMARK(BM_OrderedHashYList_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_FullOrdered_BMI_Iterate(benchmark::State &state) {
  BM_Ordered_BMI_Iterate_Impl<FullOrderedBMI>(state);
}
BENCHMARK(BM_FullOrdered_BMI_Iterate)->Unit(benchmark::kMicrosecond);

static void print_size(const char *name, size_t size) {
  std::cout << name << " sizeof=" << size << "\n";
}

static void PrintSizeReport() {
  std::cout << "\n===== Size Report =====\n";
  print_size("Particle", sizeof(Particle));

  print_size("OrderedOnlyMM::Slot", OrderedOnlyMM::slot_size());
  print_size("OrderedListMM::Slot", OrderedListMM::slot_size());
  print_size("OrderedHashMM::Slot", OrderedHashMM::slot_size());
  print_size("OrderedHashYListMM::Slot", OrderedHashYListMM::slot_size());
  print_size("MM_OrderedHashY::Slot", OrderedHashYMM::slot_size());
  print_size("FullOrderedMM::Slot", FullOrderedMM::slot_size());
  std::cout << "=======================\n\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  PrintSizeReport();

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}