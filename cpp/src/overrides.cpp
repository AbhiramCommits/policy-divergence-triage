#include "pdt/overrides.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pdt/planner.hpp"

#include "pdt/spatial.hpp"

namespace pdt {
namespace {

constexpr double kEps = 1e-9;
constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

double clampd(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

double wrap_angle(double a) {
  while (a > kPi) {
    a -= 2.0 * kPi;
  }
  while (a < -kPi) {
    a += 2.0 * kPi;
  }
  return a;
}

struct AgentPred {
  double x, y, h, v, a;
  bool pedestrian;
};

struct Pt {
  double x;
  double y;
};

double ttc_ego_agent(const Pt& pe, double ve_x, double ve_y, const Pt& pa, double va_x, double va_y) {
  const double dx = pa.x - pe.x;
  const double dy = pa.y - pe.y;
  const double dist = std::hypot(dx, dy);
  if (dist < kEps) {
    return 0.0;
  }
  const double closing = -(dx * (va_x - ve_x) + dy * (va_y - ve_y)) / dist;
  if (closing <= 1e-3) {
    return kInf;
  }
  return dist / closing;
}

LeaderInfo find_forward_obstacle(const Scenario& sc, const Context& ctx, const PlannerConfig& cfg) {
  LeaderInfo best = ctx.leader;
  const auto& cl = sc.centerline;
  const CenterlineIndex idx = make_centerline_index(cl);
  for (const Agent& ag : sc.agents) {
    if (ag.type != "vehicle") {
      continue;
    }
    State st;
    if (!agent_state_at(ag, ctx.t, st)) {
      continue;
    }
    const Proj p = project(idx, st.x, st.y);
    if (p.dist > cfg.early_brake_obstacle_lateral) {
      continue;
    }
    const double gap = p.s - ctx.s_ego - cfg.vehicle_length;
    if (p.s <= ctx.s_ego) {
      continue;
    }
    if (!best.present || gap < best.gap) {
      best = {true, p.s, gap, st.v, st.a};
    }
  }
  return best;
}

} // namespace

GuardProfile predict_guard_profile(const Scenario& scenario, const State& ego, const Context& ctx,
                                   const PlannerCommand& cmd, const PlannerConfig& cfg) {
  GuardProfile gp;
  gp.min_ttc = kInf;
  gp.min_pedestrian_dist = kInf;
  gp.max_jerk = std::fabs(cmd.accel - ego.a) / std::max(cfg.dt, kEps);

  const auto& cl = scenario.centerline;
  const CenterlineIndex idx = make_centerline_index(cl);
  const bool has_path = idx.cum.size() >= 2;

  std::vector<AgentPred> agents;
  for (const Agent& ag : scenario.agents) {
    State st;
    if (!agent_state_at(ag, ctx.t, st)) {
      continue;
    }
    agents.push_back({st.x, st.y, st.heading, st.v, st.a, ag.type == "pedestrian"});
  }

  const int n = static_cast<int>(std::ceil(cfg.override_predict_horizon / cfg.dt));
  double v_ego = std::max(0.0, ego.v);
  double s_ego = ctx.s_ego;
  const double ego_h = ego.heading;
  std::array<double, 2> pe = has_path ? point_at_s(idx, s_ego) : std::array<double, 2>{ego.x, ego.y};

  for (int k = 0; k <= n; ++k) {
    const double ve_x = v_ego * std::cos(ego_h);
    const double ve_y = v_ego * std::sin(ego_h);
    for (const AgentPred& ag : agents) {
      const double va_x = ag.v * std::cos(ag.h);
      const double va_y = ag.v * std::sin(ag.h);
      gp.min_ttc = std::min(gp.min_ttc, ttc_ego_agent({pe[0], pe[1]}, ve_x, ve_y, {ag.x, ag.y}, va_x, va_y));
      if (ag.pedestrian) {
        gp.min_pedestrian_dist = std::min(gp.min_pedestrian_dist, std::hypot(ag.x - pe[0], ag.y - pe[1]));
      }
    }
    v_ego = std::max(0.0, v_ego + cmd.accel * cfg.dt);
    s_ego += v_ego * cfg.dt;
    pe = has_path ? point_at_s(idx, s_ego)
                  : std::array<double, 2>{
                        pe[0] + v_ego * std::cos(ego_h) * cfg.dt,
                        pe[1] + v_ego * std::sin(ego_h) * cfg.dt,
                    };
    for (AgentPred& ag : agents) {
      ag.v = std::max(0.0, ag.v + ag.a * cfg.dt);
      ag.x += ag.v * std::cos(ag.h) * cfg.dt;
      ag.y += ag.v * std::sin(ag.h) * cfg.dt;
    }
  }
  return gp;
}

namespace {

class EarlyBrakingOverride : public Override {
public:
  explicit EarlyBrakingOverride(const PlannerConfig& cfg) : cfg_(cfg) {}
  std::string name() const override {
    return "early_braking";
  }

  bool applicable(const Scenario& scenario, const State& ego, const Context& ctx) const override {
    if (!preconditions(scenario, ego, ctx)) {
      return false;
    }
    const PlannerCommand cmd = compute(scenario, ego, ctx);
    const GuardProfile gp = predict_guard_profile(scenario, ego, ctx, cmd, cfg_);
    return gp.min_ttc >= cfg_.override_ttc_floor;
  }

  PlannerCommand apply(const Scenario& scenario, const State& ego, const Context& ctx) const override {
    return compute(scenario, ego, ctx);
  }

private:
  bool preconditions(const Scenario& sc, const State& ego, const Context& ctx) const {
    const LeaderInfo obs = find_forward_obstacle(sc, ctx, cfg_);
    if (!obs.present) {
      return false;
    }
    const bool hard_braking = obs.a <= cfg_.early_brake_leader_decel_threshold;
    const bool wait_behind = obs.v <= cfg_.early_brake_wait_speed && obs.gap <= cfg_.early_brake_wait_gap;
    if (!hard_braking && !wait_behind) {
      return false;
    }
    if (ego.v <= cfg_.stop_speed && !wait_behind) {
      return false;
    }
    return true;
  }

  PlannerCommand compute(const Scenario& sc, const State& ego, const Context& ctx) const {
    PlannerCommand cmd;
    cmd.decision = Decision::FOLLOW;
    cmd.accel = 0.0;
    const LeaderInfo obs = find_forward_obstacle(sc, ctx, cfg_);
    if (ego.v > cfg_.stop_speed) {
      const double d = obs.gap - cfg_.early_brake_stop_gap;
      double a = 0.0;
      if (d <= 0.0) {
        a = -cfg_.early_brake_max_decel;
      } else {
        a = -(ego.v * ego.v) / (2.0 * std::max(d, 0.5));
        a = std::max(a, -cfg_.early_brake_max_decel);
      }
      a = std::min(a, 0.0);
      const double ramp = cfg_.override_jerk_max * cfg_.dt;
      cmd.accel = std::min(clampd(a, ego.a - ramp, ego.a + ramp), 0.0);
    } else {
      cmd.decision = Decision::STOP;
    }
    return cmd;
  }

  const PlannerConfig& cfg_;
};

} // namespace

namespace {

class IntersectionCautionOverride : public Override {
public:
  explicit IntersectionCautionOverride(const PlannerConfig& cfg) : cfg_(cfg) {}
  std::string name() const override {
    return "intersection_caution";
  }

  bool applicable(const Scenario& scenario, const State& ego, const Context& ctx) const override {
    if (!preconditions(scenario, ego, ctx)) {
      return false;
    }
    const PlannerCommand cmd = compute(scenario, ego, ctx);
    const GuardProfile gp = predict_guard_profile(scenario, ego, ctx, cmd, cfg_);
    return gp.min_ttc >= cfg_.override_ttc_floor;
  }

  PlannerCommand apply(const Scenario& scenario, const State& ego, const Context& ctx) const override {
    return compute(scenario, ego, ctx);
  }

private:
  double nearest_conflict(const State& ego, const Context& ctx) const {
    double best = kInf;
    for (const ConflictInfo& c : ctx.conflicts) {
      if (!c.vehicle) {
        continue;
      }
      const double t_ego = (c.s - ctx.s_ego) / std::max(ego.v, cfg_.ttc_min_speed);
      if (std::fabs(t_ego - c.t_agent) < cfg_.intersection_caution_extended_margin) {
        best = std::min(best, c.s);
      }
    }
    return best;
  }

  bool preconditions(const Scenario& /*scenario*/, const State& ego, const Context& ctx) const {
    if (ctx.conflicts.empty()) {
      return false;
    }
    return nearest_conflict(ego, ctx) < kInf;
  }

  PlannerCommand compute(const Scenario& /*scenario*/, const State& ego, const Context& ctx) const {
    PlannerCommand cmd;
    cmd.decision = Decision::YIELD;
    cmd.accel = 0.0;
    const double s_conflict = nearest_conflict(ego, ctx);
    if (ego.v > cfg_.stop_speed && s_conflict < kInf) {
      const double d_stop_raw = s_conflict - ctx.s_ego - cfg_.intersection_caution_stop_buffer;
      double a = 0.0;
      if (d_stop_raw <= 0.0) {
        a = -cfg_.idm_comfort_decel;
      } else {
        a = -(ego.v * ego.v) / (2.0 * std::max(d_stop_raw, cfg_.yield_stop_eps));
      }
      a = std::max(a, -cfg_.idm_comfort_decel);
      a = std::min(a, 0.0);
      const double ramp = cfg_.override_jerk_max * cfg_.dt;
      cmd.accel = std::min(clampd(a, ego.a - ramp, ego.a + ramp), 0.0);
    } else if (ego.v <= cfg_.stop_speed) {
      cmd.decision = Decision::STOP;
    }
    return cmd;
  }

  const PlannerConfig& cfg_;
};

} // namespace

std::vector<std::unique_ptr<Override>> make_overrides(const PlannerConfig& cfg) {
  std::vector<std::unique_ptr<Override>> out;
  out.push_back(std::make_unique<IntersectionCautionOverride>(cfg));
  out.push_back(std::make_unique<EarlyBrakingOverride>(cfg));
  return out;
}

} // namespace pdt
