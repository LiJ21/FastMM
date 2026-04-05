#include <benchmark/benchmark.h>

#include "multimap.h"

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

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
    for (auto &id : lookup_ids)
      id = ids[pick(rng)];
  }
};

static const TestData kData;

// ===========================================================================
// PRIMARY-ONLY CONTAINERS
// ===========================================================================
using PrimaryOnlyMM =
    fastmm::FixedSizeMultiMap<Particle, kN,
                              fastmm::Unordered<fastmm::KeyFrom<&Particle::id>,
                                                IdHash, IdEqual, kBuckets>>;

struct ById {};
using PrimaryOnlyBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<bmi::hashed_unique<
        bmi::tag<ById>, bmi::member<Particle, uint64_t, &Particle::id>>>,
    boost::fast_pool_allocator<Particle>>;

static auto make_primary_mm() {
  auto m = std::make_unique<PrimaryOnlyMM>();
  for (size_t i = 0; i < kN; ++i) {
    auto it =
        m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
    if (it == m->cend()) {
      std::cerr << "PrimaryOnlyMM build failed at i=" << i << "\n";
      std::abort();
    }
  }
  return m;
}

static PrimaryOnlyBMI make_primary_bmi() {
  PrimaryOnlyBMI m;
  for (size_t i = 0; i < kN; ++i) {
    auto [it, ok] =
        m.insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
    if (!ok) {
      std::cerr << "PrimaryOnlyBMI build failed at i=" << i << "\n";
      std::abort();
    }
  }
  return m;
}

// Primary-only create
static void BM_PrimaryOnly_MM_Create(benchmark::State &state) {
  for (auto _ : state) {
    PrimaryOnlyMM m;
    for (size_t i = 0; i < kN; ++i) {
      auto it =
          m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      benchmark::DoNotOptimize(it);
    }
    benchmark::DoNotOptimize(m.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PrimaryOnly_MM_Create)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryOnly_BMI_Create(benchmark::State &state) {
  for (auto _ : state) {
    PrimaryOnlyBMI m;
    for (size_t i = 0; i < kN; ++i) {
      auto [it, ok] = m.insert(
          Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
      benchmark::DoNotOptimize(it);
      benchmark::DoNotOptimize(ok);
    }
    benchmark::DoNotOptimize(m.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PrimaryOnly_BMI_Create)->Unit(benchmark::kMicrosecond);

// Primary-only find
static void BM_PrimaryOnly_MM_Find(benchmark::State &state) {
  auto m = make_primary_mm();
  size_t i = 0;
  for (auto _ : state) {
    auto it = m->find_primary(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != m->cend()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_PrimaryOnly_MM_Find)->Unit(benchmark::kNanosecond);

static void BM_PrimaryOnly_BMI_Find(benchmark::State &state) {
  auto m = make_primary_bmi();
  auto &idx = m.get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto it = idx.find(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != idx.end()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_PrimaryOnly_BMI_Find)->Unit(benchmark::kNanosecond);

// Primary-only remove
static void BM_PrimaryOnly_MM_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_primary_mm();
    state.ResumeTiming();

    for (size_t i = 0; i < kN; ++i) {
      auto it = m->find_primary(kData.ids[i]);
      if (it != m->cend())
        m->remove(*it);
    }
    benchmark::DoNotOptimize(m->size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PrimaryOnly_MM_Remove)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryOnly_BMI_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_primary_bmi();
    state.ResumeTiming();

    auto &idx = m.get<ById>();
    for (size_t i = 0; i < kN; ++i) {
      auto it = idx.find(kData.ids[i]);
      if (it != idx.end())
        idx.erase(it);
    }
    benchmark::DoNotOptimize(m.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PrimaryOnly_BMI_Remove)->Unit(benchmark::kMicrosecond);

// Primary-only full iteration
static void BM_PrimaryOnly_MM_Iterate(benchmark::State &state) {
  auto m = make_primary_mm();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &p : *m)
      sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PrimaryOnly_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_PrimaryOnly_BMI_Iterate(benchmark::State &state) {
  auto m = make_primary_bmi();
  auto &idx = m.get<ById>();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : idx)
      sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PrimaryOnly_BMI_Iterate)->Unit(benchmark::kMicrosecond);

// ===========================================================================
// ORDERED-ONLY CONTAINERS
// ===========================================================================
using OrderedOnlyMM = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>>;

struct ByX {};
using OrderedOnlyBMI = bmi::multi_index_container<
    Particle,
    bmi::indexed_by<bmi::ordered_unique<
        bmi::tag<ByX>, bmi::member<Particle, double, &Particle::x>>>,
    boost::fast_pool_allocator<Particle>>;

static auto make_ordered_mm() {
  auto m = std::make_unique<OrderedOnlyMM>();
  for (size_t i = 0; i < kN; ++i) {
    auto it =
        m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
    if (it == m->cend()) {
      std::cerr << "OrderedOnlyMM build failed at i=" << i << "\n";
      std::abort();
    }
  }
  return m;
}

static OrderedOnlyBMI make_ordered_bmi() {
  OrderedOnlyBMI m;
  for (size_t i = 0; i < kN; ++i) {
    auto [it, ok] =
        m.insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
    if (!ok) {
      std::cerr << "OrderedOnlyBMI build failed at i=" << i << "\n";
      std::abort();
    }
  }
  return m;
}

// Ordered-only create
static void BM_OrderedOnly_MM_Create(benchmark::State &state) {
  for (auto _ : state) {
    OrderedOnlyMM m;
    for (size_t i = 0; i < kN; ++i) {
      auto it =
          m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      benchmark::DoNotOptimize(it);
    }
    benchmark::DoNotOptimize(m.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_OrderedOnly_MM_Create)->Unit(benchmark::kMicrosecond);

static void BM_OrderedOnly_BMI_Create(benchmark::State &state) {
  for (auto _ : state) {
    OrderedOnlyBMI m;
    for (size_t i = 0; i < kN; ++i) {
      auto [it, ok] = m.insert(
          Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
      benchmark::DoNotOptimize(it);
      benchmark::DoNotOptimize(ok);
    }
    benchmark::DoNotOptimize(m.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_OrderedOnly_BMI_Create)->Unit(benchmark::kMicrosecond);

// Ordered-only find by x
static void BM_OrderedOnly_MM_Find(benchmark::State &state) {
  auto m = make_ordered_mm();
  auto &idx = m->get<0>();
  size_t i = 0;
  for (auto _ : state) {
    double x = kData.xs[kData.lookup_ids[i % kData.lookup_ids.size()] - 1];
    auto it = idx.find(x);
    if (it != idx.end()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_OrderedOnly_MM_Find)->Unit(benchmark::kNanosecond);

static void BM_OrderedOnly_BMI_Find(benchmark::State &state) {
  auto m = make_ordered_bmi();
  auto &idx = m.get<ByX>();
  size_t i = 0;
  for (auto _ : state) {
    double x = kData.xs[kData.lookup_ids[i % kData.lookup_ids.size()] - 1];
    auto it = idx.find(x);
    if (it != idx.end()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_OrderedOnly_BMI_Find)->Unit(benchmark::kNanosecond);

// Ordered-only remove by x
static void BM_OrderedOnly_MM_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_ordered_mm();
    state.ResumeTiming();

    for (size_t i = 0; i < kN; ++i) {
      m->remove<0>(kData.xs[i]);
    }
    benchmark::DoNotOptimize(m->size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_OrderedOnly_MM_Remove)->Unit(benchmark::kMicrosecond);

static void BM_OrderedOnly_BMI_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_ordered_bmi();
    state.ResumeTiming();

    auto &idx = m.get<ByX>();
    for (size_t i = 0; i < kN; ++i) {
      auto it = idx.find(kData.xs[i]);
      if (it != idx.end())
        idx.erase(it);
    }
    benchmark::DoNotOptimize(m.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_OrderedOnly_BMI_Remove)->Unit(benchmark::kMicrosecond);

// Ordered-only full traversal
static void BM_OrderedOnly_MM_Iterate(benchmark::State &state) {
  auto m = make_ordered_mm();
  auto &idx = m->get<0>();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : idx)
      sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_OrderedOnly_MM_Iterate)->Unit(benchmark::kMicrosecond);

static void BM_OrderedOnly_BMI_Iterate(benchmark::State &state) {
  auto m = make_ordered_bmi();
  auto &idx = m.get<ByX>();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : idx)
      sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_OrderedOnly_BMI_Iterate)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();