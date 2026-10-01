#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "pdt/overrides.hpp"
#include "pdt/planner.hpp"
#include "pdt/scenario.hpp"

#ifndef PDT_FIXTURE_DIR
#error "PDT_FIXTURE_DIR must be defined"
#endif

namespace {

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

pdt::Scenario straight_scenario(std::vector<pdt::Agent> agents) {
  pdt::Scenario sc;
  sc.id = "override_test";
  sc.tag = "other";
  sc.ego_init = st(0.0, 0.0, 0.0, 0.0, 10.0, 0.0);
  sc.agents = std::move(agents);
  for (int i = 0; i <= 60; ++i) sc.centerline.push_back({static_cast<double>(i), 0.0});
  sc.speed_limit = 11.18;
  sc.logged_ego = {sc.ego_init};
  return sc;
}

pdt::Agent braking_leader(double x0) {
  pdt::Agent ag;
  ag.id = 1;
  ag.type = "vehicle";
  const double t_stop = 8.0 / 5.0;
  for (int i = 0; i <= 80; ++i) {
    const double t = i * 0.1;
    const double tb = std::min(t, t_stop);
    const double x = x0 + 8.0 * tb - 2.5 * tb * tb;
    const double v = std::max(0.0, 8.0 - 5.0 * t);
    ag.track.push_back(st(t, x, 0.0, 0.0, v, v > 0.0 ? -5.0 : 0.0));
  }
  return ag;
}

pdt::Agent stationary_vehicle(double x0) {
  pdt::Agent ag;
  ag.id = 2;
  ag.type = "vehicle";
  for (int i = 0; i <= 80; ++i) {
    ag.track.push_back(st(i * 0.1, x0, 0.0, 0.0, 0.0, 0.0));
  }
  return ag;
}

pdt::Agent parallel_pedestrian(double x0, double y0) {
  pdt::Agent ag;
  ag.id = 3;
  ag.type = "pedestrian";
  const double t_stop = 10.0 / 3.5;
  for (int i = 0; i <= 80; ++i) {
    const double t = i * 0.1;
    const double tb = std::min(t, t_stop);
    const double x = x0 + 10.0 * tb - 1.75 * tb * tb;
    const double v = std::max(0.0, 10.0 - 3.5 * t);
    ag.track.push_back(st(t, x, y0, 0.0, v, 0.0));
  }
  return ag;
}

pdt::Context make_context(double leader_gap, double leader_v, double leader_a) {
  pdt::Context ctx;
  ctx.t = 0.0;
  ctx.s_ego = 0.0;
  ctx.tangent_heading = 0.0;
  ctx.leader.present = true;
  ctx.leader.s = leader_gap + 4.5;
  ctx.leader.gap = leader_gap;
  ctx.leader.v = leader_v;
  ctx.leader.a = leader_a;
  return ctx;
}

std::unique_ptr<pdt::Override> get_override(const pdt::PlannerConfig& cfg, const std::string& name) {
  auto ovs = pdt::make_overrides(cfg);
  for (auto& ov : ovs) {
    if (ov->name() == name) return std::move(ov);
  }
  return nullptr;
}

bool has_event(const pdt::RulePlanner& planner, bool active, const std::string& reason_substr) {
  for (const auto& e : planner.events()) {
    if (e.active == active && e.reason.find(reason_substr) != std::string::npos) return true;
  }
  return false;
}

TEST(OverrideTest, OffReproducesOriginalTrajectoryByteForByte) {
  const auto fixtures = pdt::load_scenarios(std::string(PDT_FIXTURE_DIR) + "/scenarios.jsonl");
  for (const auto& sc : fixtures) {
    pdt::PlannerConfig off_cfg;
    off_cfg.overrides["early_braking"] = false;
    const auto base = pdt::RulePlanner().plan(sc);
    pdt::RulePlanner off_planner(off_cfg);
    const auto off = off_planner.plan(sc);
    ASSERT_EQ(base.states.size(), off.states.size());
    for (size_t i = 0; i < base.states.size(); ++i) {
      EXPECT_EQ(base.states[i].t, off.states[i].t) << sc.id;
      EXPECT_EQ(base.states[i].x, off.states[i].x) << sc.id;
      EXPECT_EQ(base.states[i].y, off.states[i].y) << sc.id;
      EXPECT_EQ(base.states[i].heading, off.states[i].heading) << sc.id;
      EXPECT_EQ(base.states[i].v, off.states[i].v) << sc.id;
      EXPECT_EQ(base.states[i].a, off.states[i].a) << sc.id;
      EXPECT_EQ(base.decisions[i], off.decisions[i]) << sc.id;
    }
    EXPECT_TRUE(off_planner.events().empty()) << sc.id;
  }
}

TEST(OverrideTest, ApplicableFalseWhenTtcFloorBreached) {
  pdt::PlannerConfig cfg;
  const auto ov = get_override(cfg, "early_braking");
  ASSERT_NE(ov, nullptr);

  std::vector<pdt::Agent> agents;
  agents.push_back(braking_leader(12.0));
  agents.push_back(stationary_vehicle(7.0));
  const auto sc_unsafe = straight_scenario(std::move(agents));
  const pdt::Context ctx = make_context(12.0 - 4.5, 8.0, -5.0);
  EXPECT_FALSE(ov->applicable(sc_unsafe, sc_unsafe.ego_init, ctx));

  const auto sc_safe = straight_scenario({braking_leader(30.0)});
  const pdt::Context ctx_far = make_context(30.0 - 4.5, 8.0, -5.0);
  EXPECT_TRUE(ov->applicable(sc_safe, sc_safe.ego_init, ctx_far));
}

TEST(OverrideTest, PedestrianBufferAlwaysVetoes) {
  const auto sc_ref = straight_scenario({braking_leader(20.0)});
  const auto base = pdt::RulePlanner().plan(sc_ref);

  pdt::Agent ped;
  ped.id = 3;
  ped.type = "pedestrian";
  for (const auto& s : base.states) {
    ped.track.push_back(st(s.t, s.x + 0.5, s.y + 2.0, s.heading, s.v, s.a));
  }

  std::vector<pdt::Agent> agents;
  agents.push_back(braking_leader(20.0));
  agents.push_back(ped);
  const auto sc = straight_scenario(std::move(agents));

  pdt::PlannerConfig cfg;
  cfg.overrides["early_braking"] = true;
  pdt::RulePlanner planner(cfg);
  const auto traj = planner.plan(sc);
  EXPECT_TRUE(has_event(planner, false, "veto_pedestrian_buffer"));
  EXPECT_FALSE(has_event(planner, true, "activated"));

  ASSERT_EQ(base.states.size(), traj.states.size());
  for (size_t i = 0; i < base.states.size(); ++i) {
    EXPECT_EQ(base.states[i].x, traj.states[i].x) << "vetoed override must not change the trajectory, step " << i;
    EXPECT_EQ(base.states[i].v, traj.states[i].v) << "vetoed override must not change the trajectory, step " << i;
    EXPECT_EQ(base.states[i].a, traj.states[i].a) << "vetoed override must not change the trajectory, step " << i;
  }
}

TEST(OverrideTest, EarlyBrakingActivatesAndStopsAtSafeGap) {
  pdt::PlannerConfig cfg;
  cfg.overrides["early_braking"] = true;
  const auto sc = straight_scenario({braking_leader(18.0)});

  pdt::RulePlanner planner(cfg);
  const auto traj = planner.plan(sc);
  EXPECT_TRUE(has_event(planner, true, "activated"));

  for (const auto& s : traj.states) {
    EXPECT_GE(s.a, -cfg.early_brake_max_decel - 1e-9);
    EXPECT_LE(s.a, cfg.idm_max_accel + 1e-9);
  }

  const double leader_stop_x = 18.0 + 6.4;
  const auto& last = traj.states.back();
  EXPECT_LT(last.v, 0.1);
  EXPECT_LE(last.x, leader_stop_x - 4.5 - 3.0);

  const auto base = pdt::RulePlanner().plan(sc);
  EXPECT_LT(last.x, base.states.back().x) << "override should stop earlier than the base planner";
}

}  // namespace
