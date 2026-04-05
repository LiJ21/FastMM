#include "extern/latency_bench.h"
#include "multimap.h"

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ctime>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace bmi = boost::multi_index;

static void pin_to_core(int core)
{
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
  {
    std::cerr << "warning: failed to pin to core " << core << ": "
              << std::strerror(errno) << "\n";
  }
#else
  (void)core;
  std::cerr << "warning: CPU pinning is only supported on Linux\n";
#endif
}

struct Particle
{
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

  Particle(uint64_t id, double x, double y, double m)
      : id(id), x(x), y(y), m(m) {}
};

struct IdHash
{
  size_t operator()(uint64_t id) const { return std::hash<uint64_t>{}(id); }
};

struct IdEqual
{
  bool operator()(uint64_t a, uint64_t b) const { return a == b; }
};

#ifdef SET_KN
static constexpr size_t kN = SET_KN;
#else
static constexpr size_t kN = 100'000;
#endif
static constexpr size_t kBuckets = kN;

struct ById
{
};
struct ByX
{
};
struct ByY
{
};
struct ByM
{
};
struct BySeq
{
};

using ParticleMap = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Named<fastmm::List, BySeq>,
    fastmm::Named<fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>,
                                           std::less<double>>,
                  ByM>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>, ByY>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
        ByX>>;

using ParticleBMIAlloc =
    boost::fast_pool_allocator<Particle,
                               boost::default_user_allocator_new_delete,
                               boost::details::pool::null_mutex, kN>;

using BMIIndexedBy = bmi::indexed_by<
    bmi::hashed_unique<bmi::tag<ById>,
                       bmi::member<Particle, uint64_t, &Particle::id>>,
    bmi::ordered_unique<bmi::tag<ByX>,
                        bmi::member<Particle, double, &Particle::x>>,
    bmi::ordered_unique<bmi::tag<ByY>,
                        bmi::member<Particle, double, &Particle::y>>,
    bmi::ordered_non_unique<bmi::tag<ByM>,
                            bmi::member<Particle, double, &Particle::m>>,
    bmi::sequenced<bmi::tag<BySeq>>>;

using ParticleBMI = bmi::multi_index_container<Particle, BMIIndexedBy>;
using ParticleBMIPool = bmi::multi_index_container<Particle, BMIIndexedBy,
                                                   ParticleBMIAlloc>;

struct TestData
{
  std::vector<uint64_t> ids;
  std::vector<double> xs, ys, ms;
  std::vector<uint64_t> lookup_ids;
  std::vector<double> lookup_ms;

  TestData()
  {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> coord(-1000.0, 1000.0);
    std::uniform_real_distribution<double> mass(1.0, 10.0);

    ids.resize(kN);
    xs.resize(kN);
    ys.resize(kN);
    ms.resize(kN);
    for (size_t i = 0; i < kN; ++i)
    {
      ids[i] = i + 1;
      xs[i] = coord(rng);
      ys[i] = coord(rng);
      ms[i] = std::round(mass(rng));
    }

    lookup_ids.resize(256);
    std::uniform_int_distribution<size_t> pick(0, kN - 1);
    for (auto &id : lookup_ids)
      id = ids[pick(rng)];

    lookup_ms.resize(64);
    for (auto &m : lookup_ms)
      m = std::round(mass(rng));
  }
};

static const TestData kData;

static auto make_multimap()
{
  auto m = std::make_shared<ParticleMap>();
  for (size_t i = 0; i < kN; ++i)
    m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
  return m;
}

static auto make_bmi()
{
  auto m = std::make_shared<ParticleBMI>();
  m->reserve(kN);
  for (size_t i = 0; i < kN; ++i)
    m->insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
  return m;
}

static auto make_bmi_pool()
{
  auto m = std::make_shared<ParticleBMIPool>();
  m->reserve(kN);
  m->get<ById>().rehash(kN);
  for (size_t i = 0; i < kN; ++i)
    m->insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
  return m;
}

static std::string format_ns(double ns)
{
  char buf[32];
  double v = std::abs(ns);
  if (v < 1000.0)
    std::snprintf(buf, sizeof(buf), "%.1f ns", ns);
  else if (v < 1'000'000.0)
    std::snprintf(buf, sizeof(buf), "%.1f us", ns / 1000.0);
  else if (v < 1'000'000'000.0)
    std::snprintf(buf, sizeof(buf), "%.1f ms", ns / 1'000'000.0);
  else
    std::snprintf(buf, sizeof(buf), "%.2f s", ns / 1'000'000'000.0);
  return buf;
}

static std::string format_ns(int64_t v)
{
  return format_ns(static_cast<double>(v));
}

struct PercentileReport
{
  std::string name;
  int64_t count{};
  double mean{};
  double sample_stddev{};
  double mean_stderr{};
  int64_t p50{}, p90{}, p95{}, p99{}, p999{}, max{};

  static PercentileReport from(const std::string &name, hdr_histogram *h)
  {
    return {name,
            h->total_count,
            hdr_mean(h),
            hdr_stddev(h),
            h->total_count > 0 ? hdr_stddev(h) / std::sqrt(h->total_count)
                               : 0.0,
            hdr_value_at_percentile(h, 50.0),
            hdr_value_at_percentile(h, 90.0),
            hdr_value_at_percentile(h, 95.0),
            hdr_value_at_percentile(h, 99.0),
            hdr_value_at_percentile(h, 99.9),
            hdr_max(h)};
  }

  void print_text(std::ostream &os) const
  {
    os << std::left << std::setw(28) << name << " n=" << std::setw(8)
       << count << " mean=" << std::fixed << std::setprecision(1)
       << std::setw(9) << mean << " sample_stddev=" << std::setw(9)
       << sample_stddev << " mean_stderr=" << std::setw(9) << mean_stderr
       << " p50=" << std::setw(8) << p50
       << " p90=" << std::setw(8) << p90 << " p95=" << std::setw(8) << p95
       << " p99=" << std::setw(8) << p99 << " p99.9=" << std::setw(8)
       << p999 << " max=" << std::setw(8) << max << " (ns)\n";
  }

  void print_json(std::ostream &os) const
  {
    os << "{\"name\":\"" << name << "\",\"count\":" << count
       << ",\"mean\":" << std::fixed << std::setprecision(1) << mean
       << ",\"sample_stddev\":" << sample_stddev
       << ",\"mean_stderr\":" << mean_stderr << ",\"p50\":" << p50
       << ",\"p90\":" << p90 << ",\"p95\":" << p95
       << ",\"p99\":" << p99 << ",\"p999\":" << p999
       << ",\"max\":" << max << "}";
  }

  void print_gbench(std::ostream &os, int name_w, int time_w, int iters_w) const
  {
    os << std::left << std::setw(name_w) << name << std::right
       << std::setw(time_w) << format_ns(mean)
       << std::setw(time_w) << format_ns(p50)
       << std::setw(time_w) << format_ns(p90)
       << std::setw(time_w) << format_ns(p99)
       << std::setw(time_w) << format_ns(p999)
       << std::setw(time_w) << format_ns(max)
       << std::setw(iters_w) << count
       << "\n";
  }
};

struct HistogramBucket
{
  int64_t value{};
  int64_t lowest_equivalent_value{};
  int64_t highest_equivalent_value{};
  int64_t count{};
  int64_t cumulative_count{};
};

struct DetailedReport
{
  PercentileReport summary;
  std::vector<HistogramBucket> histogram;

  static DetailedReport from(const std::string &name, hdr_histogram *h)
  {
    DetailedReport report{PercentileReport::from(name, h), {}};

    hdr_iter iter;
    hdr_iter_recorded_init(&iter, h);
    while (hdr_iter_next(&iter))
    {
      report.histogram.push_back(
          {iter.value, iter.lowest_equivalent_value,
           iter.highest_equivalent_value, iter.count, iter.cumulative_count});
    }

    return report;
  }
};

static void write_json_string(std::ostream &os, const std::string &value)
{
  os << '"';
  for (char c : value)
  {
    switch (c)
    {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << '"';
}

static void write_summary_json(std::ostream &os, const PercentileReport &r)
{
  os << "{\"name\":";
  write_json_string(os, r.name);
  os << ",\"count\":" << r.count << ",\"mean\":" << std::fixed
     << std::setprecision(1) << r.mean << ",\"sample_stddev\":"
     << r.sample_stddev << ",\"mean_stderr\":" << r.mean_stderr
     << ",\"p50\":" << r.p50 << ",\"p90\":" << r.p90
     << ",\"p95\":" << r.p95 << ",\"p99\":" << r.p99
     << ",\"p999\":" << r.p999 << ",\"max\":" << r.max << "}";
}

static void write_histogram_json_file(const std::string &path,
                                      const std::vector<DetailedReport> &reports)
{
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open histogram json output: " + path);

  out << "{\n  \"unit\":\"ns\",\n  \"benchmarks\":[";
  for (size_t i = 0; i < reports.size(); ++i)
  {
    const auto &report = reports[i];
    out << (i ? "," : "") << "\n    {\n      \"summary\":";
    write_summary_json(out, report.summary);
    out << ",\n      \"histogram\":[";
    for (size_t j = 0; j < report.histogram.size(); ++j)
    {
      const auto &bucket = report.histogram[j];
      out << (j ? "," : "") << "\n        {\"value\":" << bucket.value
          << ",\"lowest_equivalent_value\":"
          << bucket.lowest_equivalent_value
          << ",\"highest_equivalent_value\":"
          << bucket.highest_equivalent_value << ",\"count\":"
          << bucket.count << ",\"cumulative_count\":"
          << bucket.cumulative_count << "}";
    }
    out << "\n      ]\n    }";
  }
  out << "\n  ]\n}\n";
}

struct RegisteredLatencyBench
{
  std::string name;
  std::function<DetailedReport()> run;
};

static std::vector<RegisteredLatencyBench> &registry()
{
  static std::vector<RegisteredLatencyBench> benches;
  return benches;
}

static std::string g_filter;
static int64_t g_measure_iters_override = 0;

static bool should_register(const char *name)
{
  return g_filter.empty() || std::string{name}.find(g_filter) != std::string::npos;
}

template <size_t N>
static bool should_register_any(const char *const (&names)[N])
{
  for (const char *name : names)
  {
    if (should_register(name))
      return true;
  }
  return false;
}

static void apply_iteration_override(lb::BenchConfig &cfg)
{
  if (g_measure_iters_override > 0)
    cfg.measure_iters = g_measure_iters_override;
}

template <typename Fn>
static void register_latency_bench(const char *name, lb::BenchConfig cfg, Fn &&fn)
{
  registry().push_back(
      {name, [name, cfg, fn = std::forward<Fn>(fn)]() mutable {
         lb::LatencyBench<> bench(cfg);
         bench.run(fn);
         return DetailedReport::from(name, bench.histogram());
       }});
}

template <typename Setup, typename Fn>
static void register_latency_bench(const char *name, lb::BenchConfig cfg,
                                   Setup &&setup, Fn &&fn)
{
  registry().push_back(
      {name,
       [name, cfg, setup = std::forward<Setup>(setup),
        fn = std::forward<Fn>(fn)]() mutable {
         lb::LatencyBench<> bench(cfg);
         bench.run(setup, fn);
         return DetailedReport::from(name, bench.histogram());
       }});
}

static lb::BenchConfig single_op_cfg()
{
  lb::BenchConfig cfg;
  cfg.warmup_iters = 10'000;
  cfg.measure_iters = 100'000;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig container_op_cfg()
{
  lb::BenchConfig cfg;
  cfg.warmup_iters = 2;
  cfg.measure_iters = 50;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig pass_cfg()
{
  lb::BenchConfig cfg;
  cfg.warmup_iters = 20;
  cfg.measure_iters = 300;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig level_walk_cfg()
{
  lb::BenchConfig cfg;
  cfg.warmup_iters = 100;
  cfg.measure_iters = 1'000;
  apply_iteration_override(cfg);
  return cfg;
}

static void register_create_benches()
{
  auto cfg = container_op_cfg();
  register_latency_bench(
      "MultiMap_Create", cfg, [] { return std::make_unique<ParticleMap>(); },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
          m->template insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                   kData.ms[i]);
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "DefaultBMI_Create", cfg,
      [] {
        auto m = std::make_unique<ParticleBMI>();
        m->reserve(kN);
        return m;
      },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
          m->insert(
              Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "PoolBMI_Create", cfg,
      [] {
        auto m = std::make_unique<ParticleBMIPool>();
        m->reserve(kN);
        m->template get<ById>().rehash(kN);
        return m;
      },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
          m->insert(
              Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
        lb::do_not_optimize(m->size());
      });
}

static void register_find_primary_benches()
{
  auto cfg = single_op_cfg();

  register_latency_bench("MultiMap_FindPrimary", cfg,
                         [m = make_multimap(), i = size_t{0}]() mutable {
                           auto it = m->find_primary(
                               kData.lookup_ids[i % kData.lookup_ids.size()]);
                           if (it != m->cend())
                             lb::do_not_optimize(it->x);
                           ++i;
                         });

  register_latency_bench("DefaultBMI_FindPrimary", cfg,
                         [m = make_bmi(), i = size_t{0}]() mutable {
                           auto &idx = m->get<ById>();
                           auto it = idx.find(
                               kData.lookup_ids[i % kData.lookup_ids.size()]);
                           if (it != idx.end())
                             lb::do_not_optimize(it->x);
                           ++i;
                         });

  register_latency_bench("PoolBMI_FindPrimary", cfg,
                         [m = make_bmi_pool(), i = size_t{0}]() mutable {
                           auto &idx = m->get<ById>();
                           auto it = idx.find(
                               kData.lookup_ids[i % kData.lookup_ids.size()]);
                           if (it != idx.end())
                             lb::do_not_optimize(it->x);
                           ++i;
                         });
}

static void register_remove_benches()
{
  auto cfg = container_op_cfg();

  register_latency_bench(
      "MultiMap_Remove", cfg, [] { return make_multimap(); }, [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
        {
          auto it = m->find_primary(kData.ids[i]);
          if (it != m->cend())
            m->remove(m->to_mutable(*it));
        }
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "DefaultBMI_Remove", cfg, [] { return make_bmi(); }, [](auto &m) {
        auto &idx = m->template get<ById>();
        for (size_t i = 0; i < kN; ++i)
        {
          auto it = idx.find(kData.ids[i]);
          if (it != idx.end())
            idx.erase(it);
        }
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "PoolBMI_Remove", cfg, [] { return make_bmi_pool(); }, [](auto &m) {
        auto &idx = m->template get<ById>();
        for (size_t i = 0; i < kN; ++i)
        {
          auto it = idx.find(kData.ids[i]);
          if (it != idx.end())
            idx.erase(it);
        }
        lb::do_not_optimize(m->size());
      });
}

static void register_bulk_iterate_benches()
{
  auto cfg = pass_cfg();

  register_latency_bench("MultiMap_BulkIterate", cfg, [m = make_multimap()] {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container)
      sum += p.x + p.y + p.m;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("DefaultBMI_BulkIterate", cfg, [m = make_bmi()] {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container)
      sum += p.x + p.y + p.m;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("PoolBMI_BulkIterate", cfg, [m = make_bmi_pool()] {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container)
      sum += p.x + p.y + p.m;
    lb::do_not_optimize(sum);
  });
}

static void register_mass_range_benches()
{
  auto cfg = single_op_cfg();

  register_latency_bench("MultiMap_MassRange", cfg,
                         [m = make_multimap(), i = size_t{0}]() mutable {
                           double mass =
                               kData.lookup_ms[i % kData.lookup_ms.size()];
                           auto &idx = m->get<2>();
                           auto [beg, end] = idx.equal_range(mass);
                           size_t count = 0;
                           for (auto it = beg; it != end; ++it)
                             ++count;
                           lb::do_not_optimize(count);
                           ++i;
                         });

  register_latency_bench("DefaultBMI_MassRange", cfg,
                         [m = make_bmi(), i = size_t{0}]() mutable {
                           double mass =
                               kData.lookup_ms[i % kData.lookup_ms.size()];
                           auto &idx = m->get<ByM>();
                           auto [beg, end] = idx.equal_range(mass);
                           size_t count = 0;
                           for (auto it = beg; it != end; ++it)
                             ++count;
                           lb::do_not_optimize(count);
                           ++i;
                         });

  register_latency_bench("PoolBMI_MassRange", cfg,
                         [m = make_bmi_pool(), i = size_t{0}]() mutable {
                           double mass =
                               kData.lookup_ms[i % kData.lookup_ms.size()];
                           auto &idx = m->get<ByM>();
                           auto [beg, end] = idx.equal_range(mass);
                           size_t count = 0;
                           for (auto it = beg; it != end; ++it)
                             ++count;
                           lb::do_not_optimize(count);
                           ++i;
                         });
}

static void register_ordered_iterate_benches()
{
  auto cfg = pass_cfg();

  register_latency_bench("MultiMap_OrderedIterate", cfg, [m = make_multimap()] {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("DefaultBMI_OrderedIterate", cfg, [m = make_bmi()] {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("PoolBMI_OrderedIterate", cfg, [m = make_bmi_pool()] {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });
}

static void register_modify_benches()
{
  auto cfg = single_op_cfg();

  register_latency_bench("MultiMap_Modify", cfg,
                         [m = make_multimap(), i = size_t{0}]() mutable {
                           auto id =
                               kData.lookup_ids[i % kData.lookup_ids.size()];
                           auto it = m->find_primary(id);
                           if (it != m->cend())
                           {
                             m->modify<fastmm::ReindexOnlyByTag<ByM>>(
                                 *it, [i](Particle &p) {
                                   p.m = static_cast<double>(i % 10 + 1);
                                 });
                           }
                           lb::do_not_optimize(it);
                           ++i;
                         });

  register_latency_bench("DefaultBMI_Modify", cfg,
                         [m = make_bmi(), i = size_t{0}]() mutable {
                           auto &idx = m->get<ById>();
                           auto id =
                               kData.lookup_ids[i % kData.lookup_ids.size()];
                           auto it = idx.find(id);
                           if (it != idx.end())
                           {
                             idx.modify(it, [i](Particle &p) {
                               p.m = static_cast<double>(i % 10 + 1);
                             });
                           }
                           lb::do_not_optimize(it);
                           ++i;
                         });

  register_latency_bench("PoolBMI_Modify", cfg,
                         [m = make_bmi_pool(), i = size_t{0}]() mutable {
                           auto &idx = m->get<ById>();
                           auto id =
                               kData.lookup_ids[i % kData.lookup_ids.size()];
                           auto it = idx.find(id);
                           if (it != idx.end())
                           {
                             idx.modify(it, [i](Particle &p) {
                               p.m = static_cast<double>(i % 10 + 1);
                             });
                           }
                           lb::do_not_optimize(it);
                           ++i;
                         });
}

static void register_level_walk_benches()
{
  auto cfg = level_walk_cfg();

  register_latency_bench("MultiMap_LevelWalk", cfg, [m = make_multimap()] {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    lb::do_not_optimize(levels);
  });

  register_latency_bench("DefaultBMI_LevelWalk", cfg, [m = make_bmi()] {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    lb::do_not_optimize(levels);
  });

  register_latency_bench("PoolBMI_LevelWalk", cfg, [m = make_bmi_pool()] {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    lb::do_not_optimize(levels);
  });
}

template <typename Map>
struct MixedState
{
  std::unique_ptr<Map> m;
  std::vector<uint64_t> live_ids;
  size_t next_create_idx{};
  size_t probe{};
};

template <typename Map, typename MakeMap, typename InsertInitial>
static auto make_mixed_state(MakeMap make_map, InsertInitial insert_initial)
{
  constexpr size_t kInitial = kN / 2;
  auto state = MixedState<Map>{make_map(), {}, kInitial, 0};
  state.live_ids.reserve(kN);
  for (size_t i = 0; i < kInitial; ++i)
  {
    insert_initial(*state.m, i);
    state.live_ids.push_back(kData.ids[i]);
  }
  return state;
}

static void run_multimap_mixed(MixedState<ParticleMap> &state)
{
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  auto &m = *state.m;
  for (size_t round = 0; round < kRounds; ++round)
  {
    for (size_t j = 0; j < kCreateBurst1 && state.next_create_idx < kN; ++j)
    {
      auto i = state.next_create_idx++;
      auto it =
          m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      if (it != m.cend())
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst1 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = m.find_primary(id);
      lb::do_not_optimize(it);
      if (it != m.cend())
        lb::do_not_optimize(it->x);
    }
    state.probe += kFindBurst1;

    for (size_t j = 0; j < kModifyBurst1 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = m.find_primary(id);
      if (it != m.cend())
      {
        m.modify<fastmm::ReindexOnly<2>>(*it, [round, j](Particle &p) {
          p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
        });
      }
    }
    state.probe += kModifyBurst1;

    for (size_t j = 0; j < kRemoveBurst && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids.back();
      state.live_ids.pop_back();
      auto it = m.find_primary(id);
      if (it != m.cend())
        m.remove(*it);
    }

    for (size_t j = 0; j < kCreateBurst2 && state.next_create_idx < kN; ++j)
    {
      auto i = state.next_create_idx++;
      auto it =
          m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      if (it != m.cend())
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst2 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = m.find_primary(id);
      lb::do_not_optimize(it);
      if (it != m.cend())
        lb::do_not_optimize(it->x);
    }
    state.probe += kFindBurst2;

    for (size_t j = 0; j < kModifyBurst2 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = m.find_primary(id);
      if (it != m.cend())
      {
        m.modify<fastmm::ReindexOnly<2>>(*it, [round, j](Particle &p) {
          p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
        });
      }
    }
    state.probe += kModifyBurst2;
  }

  lb::do_not_optimize(m.size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_mixed(MixedState<Map> &state)
{
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  auto &m = *state.m;
  auto &idx = m.template get<ById>();

  for (size_t round = 0; round < kRounds; ++round)
  {
    for (size_t j = 0; j < kCreateBurst1 && state.next_create_idx < kN; ++j)
    {
      auto i = state.next_create_idx++;
      auto [it, ok] =
          m.insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
      if (ok)
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst1 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = idx.find(id);
      lb::do_not_optimize(it);
      if (it != idx.end())
        lb::do_not_optimize(it->x);
    }
    state.probe += kFindBurst1;

    for (size_t j = 0; j < kModifyBurst1 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = idx.find(id);
      if (it != idx.end())
      {
        idx.modify(it, [round, j](Particle &p) {
          p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
        });
      }
    }
    state.probe += kModifyBurst1;

    for (size_t j = 0; j < kRemoveBurst && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids.back();
      state.live_ids.pop_back();
      auto it = idx.find(id);
      if (it != idx.end())
        idx.erase(it);
    }

    for (size_t j = 0; j < kCreateBurst2 && state.next_create_idx < kN; ++j)
    {
      auto i = state.next_create_idx++;
      auto [it, ok] =
          m.insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
      if (ok)
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst2 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = idx.find(id);
      lb::do_not_optimize(it);
      if (it != idx.end())
        lb::do_not_optimize(it->x);
    }
    state.probe += kFindBurst2;

    for (size_t j = 0; j < kModifyBurst2 && !state.live_ids.empty(); ++j)
    {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      auto it = idx.find(id);
      if (it != idx.end())
      {
        idx.modify(it, [round, j](Particle &p) {
          p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
        });
      }
    }
    state.probe += kModifyBurst2;
  }

  lb::do_not_optimize(m.size());
  lb::do_not_optimize(state.live_ids.size());
}

static void register_mixed_benches()
{
  auto cfg = container_op_cfg();

  register_latency_bench(
      "MultiMap_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleMap>(
            [] { return std::make_unique<ParticleMap>(); },
            [](ParticleMap &m, size_t i) {
              auto it = m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                       kData.ms[i]);
              if (it == m.cend())
                std::abort();
            });
      },
      [](auto &state) { run_multimap_mixed(state); });

  register_latency_bench(
      "DefaultBMI_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleBMI>(
            [] {
              auto m = std::make_unique<ParticleBMI>();
              m->reserve(kN);
              return m;
            },
            [](ParticleBMI &m, size_t i) {
              auto [it, ok] = m.insert(Particle{kData.ids[i], kData.xs[i],
                                                kData.ys[i], kData.ms[i]});
              if (!ok)
                std::abort();
            });
      },
      [](auto &state) { run_bmi_mixed(state); });

  register_latency_bench(
      "PoolBMI_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleBMIPool>(
            [] {
              auto m = std::make_unique<ParticleBMIPool>();
              m->reserve(kN);
              return m;
            },
            [](ParticleBMIPool &m, size_t i) {
              auto [it, ok] = m.insert(Particle{kData.ids[i], kData.xs[i],
                                                kData.ys[i], kData.ms[i]});
              if (!ok)
                std::abort();
            });
      },
      [](auto &state) { run_bmi_mixed(state); });
}

static void register_all_benches()
{
  const char *const create[] = {"MultiMap_Create", "DefaultBMI_Create",
                                "PoolBMI_Create"};
  const char *const find_primary[] = {"MultiMap_FindPrimary",
                                      "DefaultBMI_FindPrimary",
                                      "PoolBMI_FindPrimary"};
  const char *const remove[] = {"MultiMap_Remove", "DefaultBMI_Remove",
                                "PoolBMI_Remove"};
  const char *const bulk_iterate[] = {"MultiMap_BulkIterate",
                                      "DefaultBMI_BulkIterate",
                                      "PoolBMI_BulkIterate"};
  const char *const mass_range[] = {"MultiMap_MassRange", "DefaultBMI_MassRange",
                                    "PoolBMI_MassRange"};
  const char *const ordered_iterate[] = {"MultiMap_OrderedIterate",
                                         "DefaultBMI_OrderedIterate",
                                         "PoolBMI_OrderedIterate"};
  const char *const modify[] = {"MultiMap_Modify", "DefaultBMI_Modify",
                                "PoolBMI_Modify"};
  const char *const level_walk[] = {"MultiMap_LevelWalk", "DefaultBMI_LevelWalk",
                                    "PoolBMI_LevelWalk"};
  const char *const mixed[] = {"MultiMap_Mixed", "DefaultBMI_Mixed",
                               "PoolBMI_Mixed"};

  if (should_register_any(create))
    register_create_benches();
  if (should_register_any(find_primary))
    register_find_primary_benches();
  if (should_register_any(remove))
    register_remove_benches();
  if (should_register_any(bulk_iterate))
    register_bulk_iterate_benches();
  if (should_register_any(mass_range))
    register_mass_range_benches();
  if (should_register_any(ordered_iterate))
    register_ordered_iterate_benches();
  if (should_register_any(modify))
    register_modify_benches();
  if (should_register_any(level_walk))
    register_level_walk_benches();
  if (should_register_any(mixed))
    register_mixed_benches();
}

static void print_gbench_table(std::ostream &os,
                               const std::vector<DetailedReport> &reports,
                               bool export_histogram,
                               const std::string &hist_path)
{
  // Timestamp
  std::time_t now = std::time(nullptr);
  char tbuf[64];
  std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S+00:00", std::gmtime(&now));
  os << tbuf << "\n";

  // System info
#ifdef __linux__
  {
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    std::string cpu_model;
    double cpu_mhz = 0.0;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line))
    {
      if (cpu_model.empty() && line.rfind("model name", 0) == 0)
      {
        auto pos = line.find(':');
        if (pos != std::string::npos)
          cpu_model = line.substr(pos + 2);
      }
      if (cpu_mhz == 0.0 && line.rfind("cpu MHz", 0) == 0)
      {
        auto pos = line.find(':');
        if (pos != std::string::npos)
          cpu_mhz = std::stod(line.substr(pos + 2));
      }
    }
    os << "Run on (" << ncpus << " X " << std::fixed << std::setprecision(2)
       << cpu_mhz << " MHz CPU s)\n";
    if (!cpu_model.empty())
      os << "CPU: " << cpu_model << "\n";
  }
#endif

  // Compute name column width from actual benchmark names
  int name_w = 32;
  for (const auto &r : reports)
  {
    int n = static_cast<int>(r.summary.name.size()) + 2;
    if (n > name_w) name_w = n;
  }

  constexpr int kTimeW  = 14;
  constexpr int kItersW = 11;
  int total_w = name_w + 6 * kTimeW + kItersW;
  std::string sep(total_w, '-');

  os << sep << "\n";
  os << std::left  << std::setw(name_w) << "Benchmark" << std::right
     << std::setw(kTimeW) << "Mean"
     << std::setw(kTimeW) << "p50"
     << std::setw(kTimeW) << "p90"
     << std::setw(kTimeW) << "p99"
     << std::setw(kTimeW) << "p99.9"
     << std::setw(kTimeW) << "Max"
     << std::setw(kItersW) << "Iters"
     << "\n";
  os << sep << "\n";

  for (const auto &r : reports)
    r.summary.print_gbench(os, name_w, kTimeW, kItersW);

  os << sep << "\n";

  if (export_histogram)
    os << "histogram_json=" << hist_path << "\n";
}

static int64_t parse_i64_at_least(const char *arg, const char *option,
                                  int64_t min_value)
{
  char *end = nullptr;
  errno = 0;
  long long value = std::strtoll(arg, &end, 10);
  if (errno != 0 || end == arg || *end != '\0' || value < min_value)
  {
    std::cerr << "invalid " << option << " value: " << arg << "\n";
    std::exit(2);
  }
  return static_cast<int64_t>(value);
}

int main(int argc, char **argv)
{
  bool json = false;
  bool export_histogram_json = true;
  bool pin_core = true;
  int core = 0;
  std::string filter;
  std::string histogram_json_path = "particle_latency_histograms.json";

  for (int i = 1; i < argc; ++i)
  {
    const char *arg = argv[i];
    if (!std::strcmp(arg, "--json"))
      json = true;
    else if (!std::strcmp(arg, "--hist-json") && i + 1 < argc)
      histogram_json_path = argv[++i];
    else if (!std::strncmp(arg, "--hist-json=", 12))
      histogram_json_path = arg + 12;
    else if (!std::strcmp(arg, "--no-hist-json"))
      export_histogram_json = false;
    else if (!std::strcmp(arg, "--core") && i + 1 < argc)
      core = static_cast<int>(parse_i64_at_least(argv[++i], "--core", 0));
    else if (!std::strncmp(arg, "--core=", 7))
      core = static_cast<int>(parse_i64_at_least(arg + 7, "--core", 0));
    else if (!std::strcmp(arg, "--no-pin"))
      pin_core = false;
    else if (!std::strcmp(arg, "--iterations") && i + 1 < argc)
      g_measure_iters_override =
          parse_i64_at_least(argv[++i], "--iterations", 1);
    else if (!std::strncmp(arg, "--iterations=", 13))
      g_measure_iters_override =
          parse_i64_at_least(arg + 13, "--iterations", 1);
    else if (!std::strcmp(arg, "--filter") && i + 1 < argc)
      filter = argv[++i];
    else if (!std::strncmp(arg, "--filter=", 9))
      filter = arg + 9;
    else
    {
      std::cerr << "unknown or incomplete option: " << arg << "\n";
      return 2;
    }
  }

  if (pin_core)
    pin_to_core(core);
  g_filter = filter;
  register_all_benches();

  std::vector<DetailedReport> reports;
  for (auto &bench : registry())
  {
    if (!filter.empty() && bench.name.find(filter) == std::string::npos)
      continue;
    std::cerr << "running " << bench.name << "...\n";
    reports.push_back(bench.run());
  }

  if (export_histogram_json)
    write_histogram_json_file(histogram_json_path, reports);

  if (json)
  {
    std::cout << "[";
    for (size_t i = 0; i < reports.size(); ++i)
    {
      if (i)
        std::cout << ",";
      reports[i].summary.print_json(std::cout);
    }
    std::cout << "]\n";
  }
  else
  {
    print_gbench_table(std::cout, reports, export_histogram_json,
                       histogram_json_path);
  }

  return 0;
}
