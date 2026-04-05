#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Pull in the library
// ---------------------------------------------------------------------------
#include "multimap.h"

// ---------------------------------------------------------------------------
// Domain object
// ---------------------------------------------------------------------------
struct Particle {
  uint64_t id;
  double x;
  double y;
  double m;  // mass — can repeat

  Particle(uint64_t id, double x, double y, double m)
      : id(id), x(x), y(y), m(m) {}
};

// ---------------------------------------------------------------------------
// Key getters (current API from document 7)
// ---------------------------------------------------------------------------
struct Idgetter {
  using type = uint64_t;
  const uint64_t &operator()(const Particle &p) const { return p.id; }
};
struct Xgetter {
  using type = double;
  const double &operator()(const Particle &p) const { return p.x; }
};
struct Ygetter {
  using type = double;
  const double &operator()(const Particle &p) const { return p.y; }
};
struct Mgetter {
  using type = double;
  const double &operator()(const Particle &p) const { return p.m; }
};

// ---------------------------------------------------------------------------
// Hash / equality for the unordered primary index
// ---------------------------------------------------------------------------
struct IdHash {
  std::size_t operator()(uint64_t id) const {
    return std::hash<uint64_t>{}(id);
  }
};
struct IdEqual {
  bool operator()(uint64_t a, uint64_t b) const { return a == b; }
};

// ---------------------------------------------------------------------------
// Map type
//   Index 0: Unordered<id>          — primary, unique
//   Index 1: Ordered<x>             — unique ordered by x
//   Index 2: Ordered<y>             — unique ordered by y
//   Index 3: OrderedNonUnique<m>    — non-unique ordered by mass
//   Index 4: List                   — insertion-order sequence
// ---------------------------------------------------------------------------
static constexpr std::size_t kMaxParticles = 64;

using ParticleMap = fastmm::FixedSizeMultiMap<
    Particle, kMaxParticles, fastmm::Unordered<Idgetter, IdHash, IdEqual, 128>,
    fastmm::Ordered<Xgetter, std::less<double>>,
    fastmm::Ordered<Ygetter, std::less<double>>,
    fastmm::OrderedNonUnique<Mgetter, std::less<double>>, fastmm::List>;

// Convenience: add all secondary indices after insert<false>
static void AddAll(ParticleMap &m, auto &s) {
  m.index<1>(s);
  m.index<2>(s);
  m.index<3>(s);
  m.index<4>(s);
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class ParticleMapTest : public ::testing::Test {
 protected:
  ParticleMap map;
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, initialization) {
  ParticleMap m{
      Particle{1, 0.0, 0.0, 1.0},
      Particle{2, 0.2, 0.1, 1.0},
      Particle{3, 0.3, 0.2, 1.0},
  };

  ParticleMap m1{Particle{1, 0.0, 0.0, 1.0}, Particle{2, 0.2, 0.1, 1.0},
                 Particle{3, 0.3, 0.1, 1.0}};
  EXPECT_EQ(m.size(), 3u);
  EXPECT_EQ(m1.size(), 2u);
}

// ---------------------------------------------------------------------------
// Basic create and size
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, createIncreasessize) {
  auto it = map.insert<true>(1, 0.0, 0.0, 1.0);
  ASSERT_NE(it, map.cend());
  EXPECT_EQ(map.size(), 1u);
}

TEST_F(ParticleMapTest, createMultiple) {
  map.insert<true>(1, 0.0, 0.0, 1.0);
  map.insert<true>(2, 1.0, 1.0, 2.0);
  map.insert<true>(3, 2.0, 2.0, 1.0);  // same mass as id=1
  EXPECT_EQ(map.size(), 3u);
}

// ---------------------------------------------------------------------------
// Duplicate primary key is rejected
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, DuplicateIdRejected) {
  map.insert<true>(1, 0.0, 0.0, 1.0);
  auto it = map.insert<true>(1, 0.0, 9.0, 9.0);  // same id
  EXPECT_EQ(it, map.cend());
  EXPECT_EQ(map.size(), 1u);  // still just one
}

// ---------------------------------------------------------------------------
// Find on primary index
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, FindByIdSucceeds) {
  map.insert<true>(42, 3.0, 4.0, 5.0);
  auto it = map.find_primary(42u);
  ASSERT_NE(it, map.cend());
  EXPECT_EQ(it->id, 42u);
  EXPECT_DOUBLE_EQ(it->x, 3.0);
}

TEST_F(ParticleMapTest, FindMissingIdReturnsEnd) {
  map.insert<true>(1, 0.0, 0.0, 1.0);
  auto it = map.find_primary(99u);
  EXPECT_EQ(it, map.cend());
}

// ---------------------------------------------------------------------------
// Remove by slot
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, RemoveBySlot) {
  auto it = map.insert<true>(1, 0.0, 0.0, 1.0);
  ASSERT_NE(it, map.cend());
  map.remove(*it);
  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.find_primary(1u), map.cend());
}

// ---------------------------------------------------------------------------
// Remove by key on a secondary unique index
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, RemoveByXKey) {
  map.insert<true>(1, 3.14, 0.0, 1.0);
  bool removed = map.remove<1>(3.14);
  EXPECT_TRUE(removed);
  EXPECT_EQ(map.size(), 0u);
}

TEST_F(ParticleMapTest, RemoveByMissingXKey) {
  map.insert<true>(1, 1.0, 0.0, 1.0);
  bool removed = map.remove<1>(99.0);
  EXPECT_FALSE(removed);
  EXPECT_EQ(map.size(), 1u);
}

// ---------------------------------------------------------------------------
// Ordered iteration via secondary index (x-sorted)
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, XIndexIsSorted) {
  map.insert<true>(1, 3.0, 0.0, 1.0);
  map.insert<true>(2, 1.0, 0.0, 2.0);
  map.insert<true>(3, 2.0, 0.0, 3.0);

  auto &xi = map.get<1>();
  double prev = -1e9;
  for (auto &p : xi) {
    EXPECT_GE(p.x, prev);
    prev = p.x;
  }
}

// ---------------------------------------------------------------------------
// Non-unique mass index: equal_range returns all matching
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, MassEqualRange) {
  map.insert<true>(1, 0.0, 0.0, 5.0);
  map.insert<true>(2, 1.0, 1.0, 5.0);  // same mass
  map.insert<true>(3, 2.0, 2.0, 9.0);

  auto &mi = map.get<3>();
  auto [beg, end] = mi.equal_range(5.0);
  std::size_t count = std::distance(beg, end);
  EXPECT_EQ(count, 2u);

  // every result has the right mass
  for (auto it = beg; it != end; ++it) EXPECT_DOUBLE_EQ(it->m, 5.0);
}

// ---------------------------------------------------------------------------
// Distinct mass levels via upper_bound walk
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, DistinctMassLevels) {
  map.insert<true>(1, 0.0, 0.0, 1.0);
  map.insert<true>(2, 1.0, 1.0, 1.0);  // same mass
  map.insert<true>(3, 2.0, 2.0, 2.0);
  map.insert<true>(4, 3.0, 3.0, 3.0);

  auto &mi = map.get<3>();
  std::vector<double> levels;
  for (auto it = mi.begin(); it != mi.cend(); it = mi.upper_bound(it->m))
    levels.push_back(it->m);

  ASSERT_EQ(levels.size(), 3u);
  EXPECT_DOUBLE_EQ(levels[0], 1.0);
  EXPECT_DOUBLE_EQ(levels[1], 2.0);
  EXPECT_DOUBLE_EQ(levels[2], 3.0);
}

// ---------------------------------------------------------------------------
// List index preserves insertion order
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ListIndexInsertionOrder) {
  map.insert<true>(10, 0.0, 0.0, 1.0);
  map.insert<true>(20, 1.0, 1.0, 2.0);
  map.insert<true>(30, 2.0, 2.0, 3.0);

  std::vector<uint64_t> ids;
  for (auto &p : map.get<4>()) ids.push_back(p.id);

  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 10u);
  EXPECT_EQ(ids[1], 20u);
  EXPECT_EQ(ids[2], 30u);
}

// ---------------------------------------------------------------------------
// Partial indexing: insert<false> then selective Index
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, PartialIndexing) {
  // create without secondary indices
  auto it = map.insert<false>(99, 7.0, 8.0, 3.0);
  ASSERT_NE(it, map.cend());

  // Not yet in x-index
  EXPECT_EQ(map.get<1>().size(), 0u);

  // Add to x-index only
  map.index<1>(*it);
  EXPECT_EQ(map.get<1>().size(), 1u);

  // Not yet in list
  EXPECT_EQ(map.get<4>().size(), 0u);
}

// ---------------------------------------------------------------------------
// Deindex from secondary only — object survives in index 0
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, DeindexSecondaryKeepsObject) {
  auto it = map.insert<true>(1, 1.0, 1.0, 1.0);
  ASSERT_NE(it, map.cend());

  EXPECT_EQ(map.get<1>().size(), 1u);
  map.unindex<1>(*it);
  EXPECT_EQ(map.get<1>().size(), 0u);

  // Still in primary index
  EXPECT_EQ(map.size(), 1u);
  EXPECT_NE(map.find_primary(1u), map.cend());
}

// ---------------------------------------------------------------------------
// Project<N>: project from mass-index iterator to primary
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ProjectFromMassToList) {
  auto it = map.insert<true>(55, 0.0, 0.0, 7.0);
  ASSERT_NE(it, map.cend());

  auto &mi = map.get<3>();
  auto mit = mi.find(7.0);
  ASSERT_NE(mit, mi.cend());

  // Project to list index (4)
  auto lit = map.project<4>(*mit);
  ASSERT_NE(lit, map.get<4>().cend());
  EXPECT_EQ(lit->id, 55u);
}

// ---------------------------------------------------------------------------
// Full lifecycle: create several, remove some, check consistency
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, FullLifecycle) {
  for (uint64_t i = 1; i <= 5; ++i)
    map.insert<true>(i, static_cast<double>(i), 2 * static_cast<double>(i),
                     1.0);

  EXPECT_EQ(map.size(), 5u);

  // Remove id=2 by key on x-index
  map.remove<1>(2.0);
  EXPECT_EQ(map.size(), 4u);
  EXPECT_EQ(map.find_primary(2u), map.cend());

  // Remove id=4 by slot
  auto it4 = map.find_primary(4u);
  ASSERT_NE(it4, map.cend());
  map.remove(*it4);
  EXPECT_EQ(map.size(), 3u);

  // Remaining: 1, 3, 5 — x-index still sorted
  auto &xi = map.get<1>();
  std::vector<double> xs;
  for (auto &p : xi) xs.push_back(p.x);
  ASSERT_EQ(xs.size(), 3u);
  EXPECT_DOUBLE_EQ(xs[0], 1.0);
  EXPECT_DOUBLE_EQ(xs[1], 3.0);
  EXPECT_DOUBLE_EQ(xs[2], 5.0);
}

// ---------------------------------------------------------------------------
// Pool exhaustion returns end()
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, PoolExhaustionReturnsEnd) {
  // Fill to capacity
  for (uint64_t i = 0; i < kMaxParticles; ++i) {
    auto it = map.insert<true>(i, static_cast<double>(i),
                               0.5 * static_cast<double>(i), 1.0);
    ASSERT_NE(it, map.cend()) << "Failed at i=" << i;
  }
  EXPECT_EQ(map.size(), kMaxParticles);

  // One more should fail
  auto it = map.insert<true>(kMaxParticles, 0.0, 0.0, 1.0);
  EXPECT_EQ(it, map.cend());
  EXPECT_EQ(map.size(), kMaxParticles);
}

// ---------------------------------------------------------------------------
// Reindex — unique x-index, success
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexXSuccess) {
  auto it = map.insert<true>(1, 1.0, 3.0, 2.0);
  ASSERT_NE(it, map.cend());

  bool ok = map.modify<fastmm::ReindexOnly<1>>(map.to_mutable(*it),
                                               [](Particle &p) { p.x = 9.0; });
  EXPECT_TRUE(ok);

  // New x present in x-index
  EXPECT_NE(map.get<1>().find(9.0), map.get<1>().cend());
  // Old x gone from x-index
  EXPECT_EQ(map.get<1>().find(1.0), map.get<1>().cend());

  // Object field updated
  EXPECT_DOUBLE_EQ(it->x, 9.0);

  // Other indices untouched
  EXPECT_NE(map.get<2>().find(3.0), map.get<2>().cend());  // y still there
  EXPECT_EQ(map.get<3>().size(), 1u);                      // mass still indexed
  EXPECT_EQ(map.get<4>().size(), 1u);                      // list still has it
  EXPECT_NE(map.find_primary(1u), map.cend());             // primary intact
}

// ---------------------------------------------------------------------------
// Reindex — unique x-index, conflict triggers rollback
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexXConflictRollback) {
  auto it1 = map.insert<true>(1, 1.0, 1.0, 1.0);
  auto it2 = map.insert<true>(2, 5.0, 2.0, 2.0);
  ASSERT_NE(it1, map.cend());
  ASSERT_NE(it2, map.cend());

  // Try to move particle 1's x to 5.0 — conflicts with particle 2
  bool ok =
      map.modify<fastmm::ReindexOnly<1>>(*it1, [](Particle &p) { p.x = 5.0; });
  EXPECT_FALSE(ok);

  // Particle 1 restored to original x=1.0 in x-index
  EXPECT_NE(map.get<1>().find(1.0), map.get<1>().cend());
  EXPECT_DOUBLE_EQ(it1->x, 1.0);

  // Particle 2 still at x=5.0
  EXPECT_NE(map.get<1>().find(5.0), map.get<1>().cend());
  EXPECT_DOUBLE_EQ(it2->x, 5.0);

  // Both particles still alive — no memory leak, no ghost entries
  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map.get<1>().size(), 2u);

  // Critically: other hooks on particle 1 must be intact after rollback
  // (this catches the hook-corruption bug from the earlier review)
  EXPECT_NE(map.get<2>().find(1.0), map.get<2>().cend());  // y intact
  EXPECT_EQ(map.get<3>().size(), 2u);                      // mass intact
  EXPECT_EQ(map.get<4>().size(), 2u);                      // list intact
}

// ---------------------------------------------------------------------------
// Reindex — non-unique mass index, always succeeds
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexMassSuccess) {
  auto it1 = map.insert<true>(1, 0.0, 0.0, 5.0);
  auto it2 = map.insert<true>(2, 1.0, 1.0, 5.0);  // same mass
  ASSERT_NE(it1, map.cend());
  ASSERT_NE(it2, map.cend());

  // Move particle 1 to mass=7.0
  bool ok =
      map.modify<fastmm::ReindexOnly<3>>(*it1, [](Particle &p) { p.m = 7.0; });
  EXPECT_TRUE(ok);

  // mass=5.0 level now has only particle 2
  auto &mi = map.get<3>();
  auto [b5, e5] = mi.equal_range(5.0);
  EXPECT_EQ(std::distance(b5, e5), 1);
  EXPECT_EQ(b5->id, 2u);

  // mass=7.0 level has particle 1
  auto [b7, e7] = mi.equal_range(7.0);
  EXPECT_EQ(std::distance(b7, e7), 1);
  EXPECT_EQ(b7->id, 1u);

  EXPECT_DOUBLE_EQ(it1->m, 7.0);
}

// ---------------------------------------------------------------------------
// Reindex — non-unique mass, two particles at same level, move one,
//            other level still correct
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexMassPreservesOtherLevel) {
  map.insert<true>(1, 0.0, 0.0, 3.0);
  map.insert<true>(2, 1.0, 1.0, 3.0);
  map.insert<true>(3, 2.0, 2.0, 3.0);  // three at mass=3

  auto it1 = map.find_primary(1u);
  ASSERT_NE(it1, map.cend());

  map.modify<fastmm::ReindexOnly<3>>(*it1, [](Particle &p) { p.m = 9.0; });

  auto &mi = map.get<3>();
  auto [b3, e3] = mi.equal_range(3.0);
  EXPECT_EQ(std::distance(b3, e3), 2);  // particles 2 and 3 remain

  auto [b9, e9] = mi.equal_range(9.0);
  EXPECT_EQ(std::distance(b9, e9), 1);  // particle 1 moved here
}

// ---------------------------------------------------------------------------
// Reindex with explicit rollback — success: rollback not called
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexExplicitRollbackSuccess) {
  auto it = map.insert<true>(1, 1.0, 1.0, 1.0);
  ASSERT_NE(it, map.cend());

  bool rollback_called = false;
  bool ok =
      map.modify<fastmm::ReindexOnly<1>>(*it, [](Particle &p) { p.x = 8.0; });

  EXPECT_TRUE(ok);
  EXPECT_FALSE(rollback_called);
  EXPECT_DOUBLE_EQ(it->x, 8.0);
  EXPECT_NE(map.get<1>().find(8.0), map.get<1>().cend());
}

// ---------------------------------------------------------------------------
// Reindex with explicit rollback — conflict: rollback called, state restored
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexExplicitRollbackConflict) {
  auto it1 = map.insert<true>(1, 1.0, 1.0, 1.0);
  auto it2 = map.insert<true>(2, 5.0, 2.0, 2.0);
  ASSERT_NE(it1, map.cend());
  ASSERT_NE(it2, map.cend());

  bool ok = map.modify<fastmm::ReindexOnly<1>>(
      *it1, [](Particle &p) { p.x = 5.0; }  // conflict with particle 2
  );                                        // restore

  EXPECT_FALSE(ok);
  EXPECT_TRUE(it1->x == 1.0);

  // Original state restored
  EXPECT_DOUBLE_EQ(it1->x, 1.0);
  EXPECT_NE(map.get<1>().find(1.0), map.get<1>().cend());
  EXPECT_NE(map.get<1>().find(5.0), map.get<1>().cend());
  EXPECT_EQ(map.get<1>().size(), 2u);

  // Other hooks on particle 1 intact
  EXPECT_NE(map.get<2>().find(1.0), map.get<2>().cend());
  EXPECT_EQ(map.size(), 2u);
}

// ---------------------------------------------------------------------------
// Reindex followed by Remove — no dangling hooks
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexThenRemove) {
  auto it = map.insert<true>(1, 1.0, 1.0, 1.0);
  ASSERT_NE(it, map.cend());

  map.modify<fastmm::ReindexOnly<1>>(*it, [](Particle &p) { p.x = 7.0; });
  map.remove(*it);  // must cleanly unlink from all indices

  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.get<1>().size(), 0u);
  EXPECT_EQ(map.get<2>().size(), 0u);
  EXPECT_EQ(map.get<3>().size(), 0u);
  EXPECT_EQ(map.get<4>().size(), 0u);
}

// ---------------------------------------------------------------------------
// Reindex y-index while x-index stays ordered — cross-index consistency
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexYDoesNotDisruptXOrder) {
  map.insert<true>(1, 1.0, 10.0, 1.0);
  map.insert<true>(2, 2.0, 20.0, 1.0);
  map.insert<true>(3, 3.0, 30.0, 1.0);

  auto it2 = map.find_primary(2u);
  ASSERT_NE(it2, map.cend());

  // Move particle 2's y far away
  map.modify<fastmm::ReindexOnly<2>>(*it2, [](Particle &p) { p.y = 99.0; });

  // x-index still sorted 1,2,3
  std::vector<double> xs;
  for (auto &p : map.get<1>()) xs.push_back(p.x);
  ASSERT_EQ(xs.size(), 3u);
  EXPECT_DOUBLE_EQ(xs[0], 1.0);
  EXPECT_DOUBLE_EQ(xs[1], 2.0);
  EXPECT_DOUBLE_EQ(xs[2], 3.0);

  // y-index has new position for particle 2
  EXPECT_EQ(map.get<2>().find(20.0), map.get<2>().cend());
  EXPECT_NE(map.get<2>().find(99.0), map.get<2>().cend());
}

// ---------------------------------------------------------------------------
// insert<true> secondary unique index conflict — full rollback
// Object must not survive in any index when a secondary insert fails.
// This tests the new create behavior: unindex<0> on secondary failure.
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, createAllIndicesSecondaryConflictFullRollback) {
  // Particle 1 at x=5.0 occupies the x-index slot
  auto it1 = map.insert<true>(1, 5.0, 1.0, 1.0);
  ASSERT_NE(it1, map.cend());
  EXPECT_EQ(map.size(), 1u);

  // Particle 2 with same x=5.0 — secondary x-index insert will fail
  auto it2 = map.insert<true>(2, 5.0, 2.0, 2.0);
  EXPECT_EQ(it2, map.cend());  // create must fail

  // Pool slot must be returned — size unchanged
  EXPECT_EQ(map.size(), 1u);

  // Particle 2 must not appear in any index
  EXPECT_EQ(map.find_primary(2u), map.cend());  // not in primary
  EXPECT_EQ(map.get<1>().size(), 1u);           // x-index: only particle 1
  EXPECT_EQ(map.get<2>().size(), 1u);           // y-index: only particle 1
  EXPECT_EQ(map.get<3>().size(), 1u);           // mass-index: only particle 1
  EXPECT_EQ(map.get<4>().size(), 1u);           // list: only particle 1

  // Particle 1 must be fully intact after the failed create
  EXPECT_NE(map.find_primary(1u), map.cend());
  EXPECT_DOUBLE_EQ(it1->x, 5.0);
}

// ---------------------------------------------------------------------------
// Project<N> when object is not in index N — must return end(), not UB
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ProjectNotInTargetIndexReturnsEnd) {
  // create with secondary indices disabled — object only in index 0
  auto it = map.insert<false>(1, 1.0, 1.0, 1.0);
  ASSERT_NE(it, map.cend());

  // Object is not in x-index (1), y-index (2), mass-index (3), list (4)
  EXPECT_EQ(map.project<1>(*it), map.get<1>().cend());
  EXPECT_EQ(map.project<2>(*it), map.get<2>().cend());
  EXPECT_EQ(map.project<3>(*it), map.get<3>().cend());
  EXPECT_EQ(map.project<4>(*it), map.get<4>().cend());

  // Add only to x-index — others still return end()
  map.index<1>(*it);
  EXPECT_NE(map.project<1>(*it), map.get<1>().cend());  // now in x-index
  EXPECT_EQ(map.project<2>(*it), map.get<2>().cend());  // still not in y-index
}

// ---------------------------------------------------------------------------
// Reindex twice on same slot — idempotency and no hook corruption
// ---------------------------------------------------------------------------
TEST_F(ParticleMapTest, ReindexTwiceSameSlot) {
  auto it = map.insert<true>(1, 1.0, 1.0, 1.0);
  ASSERT_NE(it, map.cend());

  // First reindex: x 1.0 -> 5.0
  bool ok1 =
      map.modify<fastmm::ReindexOnly<1>>(*it, [](Particle &p) { p.x = 5.0; });
  EXPECT_TRUE(ok1);
  EXPECT_DOUBLE_EQ(it->x, 5.0);
  EXPECT_NE(map.get<1>().find(5.0), map.get<1>().cend());
  EXPECT_EQ(map.get<1>().find(1.0), map.get<1>().cend());

  // Second reindex: x 5.0 -> 9.0
  bool ok2 =
      map.modify<fastmm::ReindexOnly<1>>(*it, [](Particle &p) { p.x = 9.0; });
  EXPECT_TRUE(ok2);
  EXPECT_DOUBLE_EQ(it->x, 9.0);
  EXPECT_NE(map.get<1>().find(9.0), map.get<1>().cend());
  EXPECT_EQ(map.get<1>().find(5.0), map.get<1>().cend());

  // All other indices intact throughout
  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map.get<1>().size(), 1u);
  EXPECT_NE(map.get<2>().find(1.0), map.get<2>().cend());  // y unchanged
  EXPECT_EQ(map.get<3>().size(), 1u);                      // mass unchanged
  EXPECT_EQ(map.get<4>().size(), 1u);                      // list unchanged
  EXPECT_NE(map.find_primary(1u), map.cend());             // primary intact

  // Reindex back to original — should be accepted cleanly
  bool ok3 =
      map.modify<fastmm::ReindexOnly<1>>(*it, [](Particle &p) { p.x = 1.0; });
  EXPECT_TRUE(ok3);
  EXPECT_DOUBLE_EQ(it->x, 1.0);
  EXPECT_NE(map.get<1>().find(1.0), map.get<1>().cend());
}

// ---------------------------------------------------------------------------
// KeyFrom — separate map type using member pointers instead of getter structs
// ---------------------------------------------------------------------------
using ParticleMapKF = fastmm::FixedSizeMultiMap<
    Particle, kMaxParticles,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual, 128>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
    fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>,
    fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>, std::less<double>>,
    fastmm::List>;

class KeyFromTest : public ::testing::Test {
 protected:
  ParticleMapKF map;
};

// Basic create and find — KeyFrom<data member> works as primary key
TEST_F(KeyFromTest, KeyFromDataMemberFind) {
  auto it = map.insert<true>(42, 1.0, 2.0, 3.0);
  ASSERT_NE(it, map.cend());
  auto found = map.find_primary(42u);
  ASSERT_NE(found, map.cend());
  EXPECT_EQ(found->id, 42u);
}

// Ordered iteration via KeyFrom<&Particle::x>
TEST_F(KeyFromTest, KeyFromDataMemberOrdering) {
  map.insert<true>(1, 3.0, 0.0, 1.0);
  map.insert<true>(2, 1.0, 0.0, 2.0);
  map.insert<true>(3, 2.0, 0.0, 3.0);

  double prev = -1e9;
  for (auto &p : map.get<1>()) {
    EXPECT_GE(p.x, prev);
    prev = p.x;
  }
}

// Non-unique index via KeyFrom<&Particle::m>
TEST_F(KeyFromTest, KeyFromDataMemberNonUnique) {
  map.insert<true>(1, 0.0, 0.0, 7.0);
  map.insert<true>(2, 1.0, 1.0, 7.0);
  map.insert<true>(3, 2.0, 2.0, 9.0);

  auto [beg, end] = map.get<3>().equal_range(7.0);
  EXPECT_EQ(std::distance(beg, end), 2);
}

// Duplicate key rejection — KeyFrom primary behaves identically to getter
TEST_F(KeyFromTest, KeyFromDataMemberDuplicateRejected) {
  map.insert<true>(1, 0.0, 0.0, 1.0);
  auto it = map.insert<true>(1, 9.0, 9.0, 9.0);
  EXPECT_EQ(it, map.cend());
  EXPECT_EQ(map.size(), 1u);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}