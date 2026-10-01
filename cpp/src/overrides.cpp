#include "pdt/overrides.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pdt/planner.hpp"

namespace pdt {
namespace {

constexpr double kEps = 1e-9;
constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

double wrap_angle(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

struct Point {
  double x;
  double y;
};

std::vector<double> cum_arc(const std::vector<std::array<double, 2>>& cl) {
  std::vector<double> cum(cl.size(), 0.0);
  for (size_t i = 1; i < cl.size(); ++i) {
    cum[i] = cum[i - 1] + std::hypot(cl[i][0] - cl[i - 1][0], cl[i][1] - cl[i - 1][1]);
  }
  return cum;
}

Point point_at_s(const std::vector<std::array<double, 2>>& cl, const std::vector<double>& cum, double s) {
  if (cl.empty()) return {0.0, 0.0};
  if (cl.size() == 1 || cum.back() <= kEps) return {cl[0][0], cl[0][1]};
  s = clampd(s, 0.0, cum.back());
  size_t i = 0;
  for (; i + 1 < cl.size(); ++i) {
    if (cum[i + 1] >= s) break;
  }
  if (i + 1 >= cl.size()) return {cl.back()[0], cl.back()[1]};
  const double seg = cum[i + 1] - cum[i];
  const double t = seg > kEps ? clampd((s - cum[i]) / seg, 0.0, 1.0) : 0.0;
  return {cl[i][0] + t * (cl[i + 1][0] - cl[i][0]), cl[i][1] + t * (cl[i + 1][1] - cl[i][1])};
}

bool agent_state_at(const Agent& ag, double t, State& out) {
  const auto& tr = ag.track;
  if (tr.empty()) return false;
  if (t <= tr.front().t) {
    out = tr.front();
    return true;
  }
  if (t >= tr.back().t) {
    out = tr.back();
    return true;
  }
  for (size_t i = 0; i + 1 < tr.size(); ++i) {
    if (t >= tr[i].t && t <= tr[i + 1].t) {
      const double dt = tr[i + 1].t - tr[i].t;
      const double f = dt > kEps ? (t - tr[i].t) / dt : 0.0;
      out.t = t;
      out.x = tr[i].x + f * (tr[i + 1].x - tr[i].x);
      out.y = tr[i].y + f * (tr[i + 1].y - tr[i].y);
      out.heading = wrap_angle(tr[i].heading + f * wrap_angle(tr[i + 1].heading - tr[i].heading));
      out.v = tr[i].v + f * (tr[i + 1].v - tr[i].v);
      out.a = tr[i].a + f * (tr[i + 1].a - tr[i].a);
      return true;
    }
  }
  out = tr.back();
  return true;
}

double ttc_ego_agent(const Point& pe, double ve_x, double ve_y, const Point& pa, double va_x, double va_y) {
  const double dx = pa.x - pe.x;
  const double dy = pa.y - pe.y;
  const double dist = std::hypot(dx, dy);
  if (dist < kEps) return 0.0;
  const double closing = -(dx * (va_x - ve_x) + dy * (va_y - ve_y)) / dist;
  if (closing <= 1e-3) return kInf;
  return dist / closing;
}

struct Projection {
  double s;
  double dist;
};

Projection project_pt(const std::vector<std::array<double, 2>>& cl, const std::vector<double>& cum, double x, double y) {
  Projection best{0.0, kInf};
  if (cl.size() < 2) return best;
  for (size_t i = 0; i + 1 < cl.size(); ++i) {
    const double abx = cl[i + 1][0] - cl[i][0];
    const double aby = cl[i + 1][1] - cl[i][1];
    const double len2 = abx * abx + aby * aby;
    double t = len2 > kEps ? ((x - cl[i][0]) * abx + (y - cl[i][1]) * aby) / len2 : 0.0;
    t = clampd(t, 0.0, 1.0);
    const double px = cl[i][0] + t * abx;
    const double py = cl[i][1] + t * aby;
    const double d = std::hypot(x - px, y - py);
    if (d < best.dist) {
      best.dist = d;
      best.s = cum[i] + t * std::sqrt(len2);
    }
  }
  return best;
}

LeaderInfo find_forward_obstacle(const Scenario& sc, const Context& ctx, const PlannerConfig& cfg) {
  LeaderInfo best = ctx.leader;
  const auto& cl = sc.centerline;
  const std::vector<double> cum = cum_arc(cl);
  for (const Agent& ag : sc.agents) {
    if (ag.type != "vehicle") continue;
    State st;
    if (!agent_state_at(ag, ctx.t, st)) continue;
    const Projection p = project_pt(cl, cum, st.x, st.y);
    if (p.dist > cfg.early_brake_obstacle_lateral) continue;
    const double gap = p.s - ctx.s_ego - cfg.vehicle_length;
    if (p.s <= ctx.s_ego) continue;
    if (!best.present || gap < best.gap) {
      best = {true, p.s, gap, st.v, st.a};
    }
  }
  return best;
}

struct AgentPred {
  double x, y, h, v, a;
  bool pedestrian;
};

}  // namespace

GuardProfile predict_guard_profile(const Scenario& scenario, const State& ego, const Context& ctx,
                                   const PlannerCommand& cmd, const PlannerConfig& cfg) {
  GuardProfile gp;
  gp.min_ttc = kInf;
  gp.min_pedestrian_dist = kInf;
  gp.max_jerk = std::fabs(cmd.accel - ego.a) / std::max(cfg.dt, kEps);

  const auto& cl = scenario.centerline;
  const std::vector<double> cum = cum_arc(cl);
  const bool has_path = cl.size() >= 2;

  std::vector<AgentPred> agents;
  for (const Agent& ag : scenario.agents) {
    State st;
    if (!agent_state_at(ag, ctx.t, st)) continue;
    agents.push_back({st.x, st.y, st.heading, st.v, st.a, ag.type == "pedestrian"});
  }

  const int n = static_cast<int>(std::ceil(cfg.override_predict_horizon / cfg.dt));
  double v_ego = std::max(0.0, ego.v);
  double s_ego = ctx.s_ego;
  const double ego_h = ego.heading;
  Point pe = has_path ? point_at_s(cl, cum, s_ego) : Point{ego.x, ego.y};

  for (int k = 0; k <= n; ++k) {
    const double ve_x = v_ego * std::cos(ego_h);
    const double ve_y = v_ego * std::sin(ego_h);
    for (const AgentPred& ag : agents) {
      const double va_x = ag.v * std::cos(ag.h);
      const double va_y = ag.v * std::sin(ag.h);
      gp.min_ttc = std::min(gp.min_ttc, ttc_ego_agent(pe, ve_x, ve_y, {ag.x, ag.y}, va_x, va_y));
      if (ag.pedestrian) {
        gp.min_pedestrian_dist = std::min(gp.min_pedestrian_dist, std::hypot(ag.x - pe.x, ag.y - pe.y));
      }
    }
    v_ego = std::max(0.0, v_ego + cmd.accel * cfg.dt);
    s_ego += v_ego * cfg.dt;
    if (has_path) {
      pe = point_at_s(cl, cum, s_ego);
    } else {
      pe.x += v_ego * std::cos(ego_h) * cfg.dt;
      pe.y += v_ego * std::sin(ego_h) * cfg.dt;
    }
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
  std::string name() const override { return "early_braking"; }

  bool applicable(const Scenario& scenario, const State& ego, const Context& ctx) const override {
    if (!preconditions(scenario, ego, ctx)) return false;
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
    if (!obs.present) return false;
    const bool hard_braking = obs.a <= cfg_.early_brake_leader_decel_threshold;
    const bool wait_behind = obs.v <= cfg_.early_brake_wait_speed && obs.gap <= cfg_.early_brake_wait_gap;
    if (!hard_braking && !wait_behind) return false;
    if (ego.v <= cfg_.stop_speed && !wait_behind) return false;
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

}  // namespace

std::vector<std::unique_ptr<Override>> make_overrides(const PlannerConfig& cfg) {
  std::vector<std::unique_ptr<Override>> out;
  out.push_back(std::make_unique<EarlyBrakingOverride>(cfg));
  return out;
}

}  // namespace pdt
