#pragma once
#include <hdr/hdr_histogram.h>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <functional>
#include <iostream>
#include <cstring>

namespace lb {

// -- Clock policies ----------------------------------------------------------

struct ChronoClock {
    using tp = std::chrono::steady_clock::time_point;
    static tp now() noexcept { return std::chrono::steady_clock::now(); }
    static int64_t delta_ns(tp a, tp b) noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
};

#if defined(__x86_64__) || defined(_M_X64)
struct RdtscClock {
    using tp = uint64_t;
    static tp now() noexcept {
        unsigned lo, hi;
        asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
        return (uint64_t(hi) << 32) | lo;
    }
    static int64_t delta_ns(tp a, tp b) noexcept {
        static const double ns_per_tick = [] {
            auto c0 = now();
            auto t0 = std::chrono::steady_clock::now();
            volatile int sink = 0;
            for (int i = 0; i < 1'000'000; ++i) sink += i;
            auto c1 = now();
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            return ns / double(c1 - c0);
        }();
        return int64_t(double(b - a) * ns_per_tick);
    }
};
#endif

// -- Compiler barrier --------------------------------------------------------

template <typename T>
inline void do_not_optimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

inline void clobber_memory() { asm volatile("" ::: "memory"); }

// -- Config ------------------------------------------------------------------

struct BenchConfig {
    int64_t warmup_iters   = 1000;
    int64_t measure_iters  = 100'000;
    int64_t hist_min       = 1;          // ns
    int64_t hist_max       = 10'000'000'000LL; // 10s
    int     sig_digits     = 3;
};

// -- LatencyBench ------------------------------------------------------------

template <typename Clock = ChronoClock>
class LatencyBench {
public:
    explicit LatencyBench(BenchConfig cfg = {}) : cfg_(cfg) {
        if (hdr_init(cfg_.hist_min, cfg_.hist_max, cfg_.sig_digits, &hist_))
            throw std::runtime_error("hdr_init failed");
    }
    ~LatencyBench() { hdr_close(hist_); }

    LatencyBench(const LatencyBench&) = delete;
    LatencyBench& operator=(const LatencyBench&) = delete;

    template <typename Fn>
    void run(Fn&& fn) {
        for (int64_t i = 0; i < cfg_.warmup_iters; ++i) {
            clobber_memory();
            fn();
            clobber_memory();
        }
        hdr_reset(hist_);
        for (int64_t i = 0; i < cfg_.measure_iters; ++i) {
            clobber_memory();
            auto t0 = Clock::now();
            fn();
            auto t1 = Clock::now();
            hdr_record_value(hist_, Clock::delta_ns(t0, t1));
        }
    }

    template <typename Setup, typename Fn>
    void run(Setup&& setup, Fn&& fn) {
        for (int64_t i = 0; i < cfg_.warmup_iters; ++i) {
            auto state = setup();
            clobber_memory();
            fn(state);
            clobber_memory();
        }
        hdr_reset(hist_);
        for (int64_t i = 0; i < cfg_.measure_iters; ++i) {
            auto state = setup();
            clobber_memory();
            auto t0 = Clock::now();
            fn(state);
            auto t1 = Clock::now();
            hdr_record_value(hist_, Clock::delta_ns(t0, t1));
        }
    }

    hdr_histogram* histogram() const noexcept { return hist_; }

private:
    BenchConfig     cfg_;
    hdr_histogram*  hist_ = nullptr;
};

// -- Batch recording (amortised sub-ns measurements) -------------------------

template <typename Clock = ChronoClock>
class BatchBench {
public:
    explicit BatchBench(int64_t batch_size, BenchConfig cfg = {})
        : batch_(batch_size), cfg_(cfg) {
        if (hdr_init(cfg_.hist_min, cfg_.hist_max, cfg_.sig_digits, &hist_))
            throw std::runtime_error("hdr_init failed");
    }
    ~BatchBench() { hdr_close(hist_); }

    // fn(int64_t n) must execute the operation n times
    template <typename Fn>
    void run(Fn&& fn) {
        for (int64_t i = 0; i < cfg_.warmup_iters / batch_; ++i) {
            clobber_memory(); fn(batch_); clobber_memory();
        }
        hdr_reset(hist_);
        for (int64_t i = 0; i < cfg_.measure_iters; ++i) {
            clobber_memory();
            auto t0 = Clock::now();
            fn(batch_);
            auto t1 = Clock::now();
            hdr_record_value(hist_, Clock::delta_ns(t0, t1) / batch_);
        }
    }

    hdr_histogram* histogram() const noexcept { return hist_; }

private:
    int64_t         batch_;
    BenchConfig     cfg_;
    hdr_histogram*  hist_ = nullptr;
};

struct BenchReport {
    std::string name;
    int64_t count{};
    double  mean{};
    int64_t p50{}, p90{}, p99{}, p999{}, p9999{}, max{};

    static BenchReport from(const char* name, hdr_histogram* h) {
        return {
            name, h->total_count, hdr_mean(h),
            hdr_value_at_percentile(h, 50.0),
            hdr_value_at_percentile(h, 90.0),
            hdr_value_at_percentile(h, 99.0),
            hdr_value_at_percentile(h, 99.9),
            hdr_value_at_percentile(h, 99.99),
            hdr_max(h)
        };
    }

    friend std::ostream& operator<<(std::ostream& os, const BenchReport& r) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "%-24s  n=%-8ld  mean=%7.1f  p50=%7ld  p90=%7ld  "
            "p99=%7ld  p99.9=%7ld  p99.99=%7ld  max=%7ld  (ns)",
            r.name.c_str(), r.count, r.mean,
            r.p50, r.p90, r.p99, r.p999, r.p9999, r.max);
        return os << buf;
    }

    std::string to_json() const {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            R"({"name":"%s","count":%ld,"mean":%.1f,)"
            R"("p50":%ld,"p90":%ld,"p99":%ld,)"
            R"("p999":%ld,"p9999":%ld,"max":%ld})",
            name.c_str(), count, mean,
            p50, p90, p99, p999, p9999, max);
        return buf;
    }
};
struct RegisteredBench {
    std::string name;
    std::function<void(hdr_histogram*&)> run;
};

inline std::vector<RegisteredBench>& registry() {
    static std::vector<RegisteredBench> v;
    return v;
}

struct AutoReg {
    AutoReg(const char* name, std::function<void(hdr_histogram*&)> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

// Usage: LB_BENCH(name) { /* body; use lb::do_not_optimize() */ }
#define LB_BENCH(Name)                                                       \
    static void lb_bench_##Name(lb::BenchConfig);                            \
    static void lb_bench_##Name##_wrap(hdr_histogram*& out) {                \
        lb::LatencyBench<> b;                                                \
        b.run([&]{ lb_bench_##Name({}); });                                  \
        out = b.histogram();                                                 \
    }                                                                        \
    static lb::AutoReg lb_reg_##Name(#Name, lb_bench_##Name##_wrap);         \
    static void lb_bench_##Name([[maybe_unused]] lb::BenchConfig cfg)

// Simpler: register a lambda directly
inline void register_bench(const char* name, std::function<void()> fn,
                           BenchConfig cfg = {}) {
    registry().push_back({name, [fn=std::move(fn), cfg](hdr_histogram*& out) {
        LatencyBench<> b(cfg);
        b.run(fn);
        out = b.histogram();
    }});
}

inline int bench_main(int argc, char** argv) {
    bool json = false;
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--json")) json = true;
        else if (!std::strcmp(argv[i], "--filter") && i + 1 < argc) filter = argv[++i];
    }

    std::vector<BenchReport> reports;
    for (auto& b : registry()) {
        if (!filter.empty() && b.name.find(filter) == std::string::npos) continue;
        hdr_histogram* h = nullptr;
        b.run(h);
        reports.push_back(BenchReport::from(b.name.c_str(), h));
    }

    if (json) {
        std::cout << "[";
        for (size_t i = 0; i < reports.size(); ++i)
            std::cout << (i ? "," : "") << reports[i].to_json();
        std::cout << "]\n";
    } else {
        for (auto& r : reports) std::cout << r << "\n";
    }
    return 0;
}

struct CompareResult {
    struct Delta { const char* label; int64_t base; int64_t cand; double pct; };
    std::string base_name, cand_name;
    Delta deltas[6]; // p50..max
    bool  regressed = false;

    friend std::ostream& operator<<(std::ostream& os, const CompareResult& c) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%-24s vs %-24s\n", c.base_name.c_str(), c.cand_name.c_str());
        os << buf;
        for (auto& d : c.deltas) {
            const char* tag = d.pct > 5.0 ? " REGRESSED" : d.pct < -5.0 ? " IMPROVED" : "";
            std::snprintf(buf, sizeof(buf), "  %-8s %7ld -> %7ld  (%+.1f%%)%s\n",
                          d.label, d.base, d.cand, d.pct, tag);
            os << buf;
        }
        return os;
    }
};

inline CompareResult compare(const BenchReport& base, const BenchReport& cand,
                              double threshold_pct = 5.0) {
    auto pct = [](int64_t b, int64_t c) -> double {
        return b ? 100.0 * (double(c) - double(b)) / double(b) : 0.0;
    };
    CompareResult r{base.name, cand.name, {
        {"p50",    base.p50,   cand.p50,   pct(base.p50,   cand.p50)},
        {"p90",    base.p90,   cand.p90,   pct(base.p90,   cand.p90)},
        {"p99",    base.p99,   cand.p99,   pct(base.p99,   cand.p99)},
        {"p99.9",  base.p999,  cand.p999,  pct(base.p999,  cand.p999)},
        {"p99.99", base.p9999, cand.p9999, pct(base.p9999, cand.p9999)},
        {"max",    base.max,   cand.max,   pct(base.max,   cand.max)},
    }};
    for (auto& d : r.deltas)
        if (d.pct > threshold_pct) { r.regressed = true; break; }
    return r;
}
} // namespace lb
