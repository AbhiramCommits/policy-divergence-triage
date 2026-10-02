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

constexpr double kPi = 3.14159265358979323846;

pdt::State st(double t, double x, double y, double h, double v, double a) {
  pdt::State s;
  s.t = t;
  s.x = x;
  s.y = y;
  s.heading = h;
  s.v = v;
  s.a = a;
  return s;
}

pdt::Agent straight_vehicle(int id, double x0, double v, double a) {
  pdt::Agent ag;
  ag.id = id;
  ag.type = "vehicle";
  for (int i = 0; i <= 80; ++i) {
    const double t = i * 0.1;
    ag.track.push_back(st(t, x0 + v * t + 0.5 * a * t * t, 0.0, 0.0, std::max(0.0, v + a * t), a));
  }
  return ag;
}

pdt::Scenario straight_scenario(std::vector<pdt::Agent> agents, double ego_v, double speed_limit, double ego_y = 0.0,
                                double ego_h = 0.0) {
  pdt::Scenario sc;
  sc.id = "behavior_test";
  sc.tag = "other";
  sc.ego_init = st(0.0, 0.0, ego_y, ego_h, ego_v, 0.0);
  sc.agents = std::move(agents);
  for (int i = 0; i <= 150; ++i)
    sc.centerline.push_back({static_cast<double>(i), 0.0});
  sc.speed_limit = speed_limit;
  sc.logged_ego = {sc.ego_init};
  return sc;
}

const pdt::Scenario* find_fixture(const std::vector<pdt::Scenario>& scs, const std::string& id) {
  for (const auto& s : scs) {
    if (s.id == id)
      return &s;
  }
  return nullptr;
}

TEST(IdmTest, FreeFlowAcceleratesAtIdmMaxAccel) {
  const auto sc = straight_scenario({}, 0.0, 11.18);
  const auto traj = pdt::RulePlanner().plan(sc);
  EXPECT_NEAR(traj.states[1].a, 1.5, 1e-9);
  EXPECT_NEAR(traj.states[1].v, 0.15, 1e-9);
}

TEST(IdmTest, LeaderBrakingForcesDeceleration) {
  const auto sc = straight_scenario({straight_vehicle(1, 20.0, 8.0, -4.0)}, 10.0, 11.18);
  pdt::PlannerConfig cfg;
  const auto traj = pdt::RulePlanner(cfg).plan(sc);
  EXPECT_LT(traj.states[0].a, 0.0);
  bool decelerating = false;
  for (size_t i = 0; i + 1 < traj.states.size(); ++i) {
    if (traj.states[i].v > traj.states[i + 1].v)
      decelerating = true;
  }
  EXPECT_TRUE(decelerating);
  for (const auto& s : traj.states) {
    EXPECT_GE(s.a, -cfg.idm_comfort_decel - 1e-9);
    EXPECT_LE(s.a, cfg.idm_max_accel + 1e-9);
  }
}

TEST(IdmTest, ConvergesToLeaderSpeedAndGap) {
  const auto sc = straight_scenario({straight_vehicle(1, 25.0, 8.0, 0.0)}, 10.0, 11.18);
  const auto traj = pdt::RulePlanner().plan(sc);
  const auto& last = traj.states.back();
  EXPECT_NEAR(last.v, 8.0, 2.0);
  const double gap = 25.0 + 8.0 * last.t - last.x - 4.5;
  EXPECT_GT(gap, 4.0);
  EXPECT_LT(gap, 20.0);
}

TEST(YieldTest, DistantCrossingAssertsAndHoldsSpeed) {
  const auto fixtures = pdt::load_scenarios(std::string(PDT_FIXTURE_DIR) + "/scenarios.jsonl");
  const auto* sc = find_fixture(fixtures, "fixture_ped_02");
  ASSERT_NE(sc, nullptr);
  const auto traj = pdt::RulePlanner().plan(*sc);
  EXPECT_EQ(traj.decisions[0], pdt::Decision::ASSERT);
  double min_v = 1e9;
  for (size_t i = 0; i < 30 && i < traj.states.size(); ++i)
    min_v = std::min(min_v, traj.states[i].v);
  EXPECT_GT(min_v, 9.0);
}

TEST(YieldTest, ResumesFollowAfterPedestrianClears) {
  const auto fixtures = pdt::load_scenarios(std::string(PDT_FIXTURE_DIR) + "/scenarios.jsonl");
  const auto* sc = find_fixture(fixtures, "fixture_ped_00");
  ASSERT_NE(sc, nullptr);
  const auto traj = pdt::RulePlanner().plan(*sc);
  bool saw_stop = false;
  for (const auto d : traj.decisions)
    saw_stop = saw_stop || d == pdt::Decision::STOP;
  EXPECT_TRUE(saw_stop);
  EXPECT_EQ(traj.decisions.back(), pdt::Decision::FOLLOW);
  EXPECT_GT(traj.states.back().v, 0.5);
}

TEST(PlannerTest, PurePursuitConvergesToCenterline) {
  const auto sc = straight_scenario({}, 10.0, 11.18, /*ego_y=*/2.5);
  const auto traj = pdt::RulePlanner().plan(sc);
  EXPECT_GT(std::fabs(traj.states[0].y), 2.0);
  EXPECT_LT(std::fabs(traj.states.back().y), 1.0);
}

TEST(ScenarioIoTest, MissingFileThrows) {
  EXPECT_THROW(pdt::load_scenarios("/nonexistent/scenarios.jsonl"), std::runtime_error);
}

} // namespace
