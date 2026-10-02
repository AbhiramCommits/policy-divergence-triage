#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "pdt/planner.hpp"
#include "pdt/scenario.hpp"

#ifndef PDT_FIXTURE_DIR
#error "PDT_FIXTURE_DIR must be defined"
#endif

namespace {

std::string fixture_path() {
  return std::string(PDT_FIXTURE_DIR) + "/scenarios.jsonl";
}

std::vector<pdt::Scenario> load_fixtures() {
  return pdt::load_scenarios(fixture_path());
}

const pdt::Scenario* find_by_id(const std::vector<pdt::Scenario>& scs, const std::string& id) {
  for (const auto& s : scs) {
    if (s.id == id)
      return &s;
  }
  return nullptr;
}

TEST(PlannerTest, DeterministicAcrossFixtures) {
  pdt::RulePlanner planner;
  const auto scenarios = load_fixtures();
  ASSERT_EQ(scenarios.size(), 20u);
  for (const auto& sc : scenarios) {
    const auto t1 = planner.plan(sc);
    const auto t2 = planner.plan(sc);
    ASSERT_EQ(t1.states.size(), 81u) << sc.id;
    ASSERT_EQ(t1.states.size(), t2.states.size());
    ASSERT_EQ(t1.decisions.size(), t2.decisions.size());
    for (size_t i = 0; i < t1.states.size(); ++i) {
      EXPECT_EQ(t1.states[i].t, t2.states[i].t) << sc.id << " step " << i;
      EXPECT_EQ(t1.states[i].x, t2.states[i].x) << sc.id << " step " << i;
      EXPECT_EQ(t1.states[i].y, t2.states[i].y) << sc.id << " step " << i;
      EXPECT_EQ(t1.states[i].heading, t2.states[i].heading) << sc.id << " step " << i;
      EXPECT_EQ(t1.states[i].v, t2.states[i].v) << sc.id << " step " << i;
      EXPECT_EQ(t1.states[i].a, t2.states[i].a) << sc.id << " step " << i;
      EXPECT_EQ(t1.decisions[i], t2.decisions[i]) << sc.id << " step " << i;
    }
    EXPECT_EQ(t1.scenario_id, sc.id);
    EXPECT_EQ(t1.source, "rule_planner");
  }
}

TEST(PlannerTest, AccelWithinConfigBounds) {
  pdt::PlannerConfig cfg;
  pdt::RulePlanner planner(cfg);
  for (const auto& sc : load_fixtures()) {
    const auto traj = planner.plan(sc);
    for (size_t i = 0; i < traj.states.size(); ++i) {
      const double a = traj.states[i].a;
      EXPECT_TRUE(std::isfinite(a)) << sc.id << " step " << i;
      EXPECT_LE(a, cfg.idm_max_accel + 1e-12) << sc.id << " step " << i;
      EXPECT_GE(a, -cfg.idm_comfort_decel - 1e-12) << sc.id << " step " << i;
    }
  }
}

TEST(PlannerTest, YieldStopsBeforeConflictPoint) {
  pdt::RulePlanner planner;
  const auto scenarios = load_fixtures();
  const pdt::Scenario* sc = find_by_id(scenarios, "fixture_ped_00");
  ASSERT_NE(sc, nullptr);
  ASSERT_EQ(sc->tag, "ped_or_cyclist_interaction");

  const auto traj = planner.plan(*sc);
  ASSERT_EQ(traj.states.size(), traj.decisions.size());

  bool saw_yield = false;
  bool saw_stop = false;
  const double kConflictX = 30.0;
  for (size_t i = 0; i < traj.decisions.size(); ++i) {
    const auto dec = traj.decisions[i];
    if (dec == pdt::Decision::YIELD)
      saw_yield = true;
    if (dec == pdt::Decision::STOP)
      saw_stop = true;
    if (dec == pdt::Decision::YIELD || dec == pdt::Decision::STOP) {
      EXPECT_LE(traj.states[i].x, 28.6) << "ego crosses the conflict point while yielding/stopped at step " << i;
    }
    if (dec == pdt::Decision::STOP) {
      EXPECT_LE(traj.states[i].v, 0.05 + 1e-9) << "STOP state must be stationary at step " << i;
      EXPECT_LT(traj.states[i].x, kConflictX) << "STOP state must be before the conflict point at step " << i;
    }
  }
  EXPECT_TRUE(saw_yield);
  EXPECT_TRUE(saw_stop);
}

} // namespace
