#include "pdt/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace pdt {
namespace {

constexpr double kEps = 1e-9;
constexpr double kPi = 3.14159265358979323846;

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

struct Projection {
  double s;
  double dist;
  Point p;
};

std::vector<double> cum_arc(const std::vector<std::array<double, 2>>& cl) {
  std::vector<double> cum(cl.size(), 0.0);
  for (size_t i = 1; i < cl.size(); ++i) {
    cum[i] = cum[i - 1] + std::hypot(cl[i][0] - cl[i - 1][0], cl[i][1] - cl[i - 1][1]);
  }
  return cum;
}

Projection project(const std::vector<std::array<double, 2>>& cl, const std::vector<double>& cum, double x,
                   double y) {
  Projection best;
  best.s = 0.0;
  best.dist = std::numeric_limits<double>::infinity();
  best.p = {cl.empty() ? x : cl[0][0], cl.empty() ? y : cl[0][1]};
  if (cl.empty()) return best;
  if (cl.size() == 1) {
    best.dist = std::hypot(x - cl[0][0], y - cl[0][1]);
    best.p = {cl[0][0], cl[0][1]};
    return best;
  }
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
      best.p = {px, py};
    }
  }
  return best;
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
  const double seg_len = cum[i + 1] - cum[i];
  const double t = seg_len > kEps ? clampd((s - cum[i]) / seg_len, 0.0, 1.0) : 0.0;
  return {cl[i][0] + t * (cl[i + 1][0] - cl[i][0]), cl[i][1] + t * (cl[i + 1][1] - cl[i][1])};
}

double tangent_at_s(const std::vector<std::array<double, 2>>& cl, const std::vector<double>& cum, double s) {
  if (cl.size() < 2 || cum.back() <= kEps) return 0.0;
  s = clampd(s, 0.0, cum.back());
  size_t i = 0;
  for (; i + 1 < cl.size(); ++i) {
    if (cum[i + 1] >= s - kEps) break;
  }
  if (i + 1 >= cl.size()) i = cl.size() - 2;
  return std::atan2(cl[i + 1][1] - cl[i][1], cl[i + 1][0] - cl[i][0]);
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

struct Conflict {
  double s;
  double t_agent;
};

bool predict_conflict(const std::vector<std::array<double, 2>>& cl, const std::vector<double>& cum, const State& st,
                      const PlannerConfig& cfg, Conflict& out) {
  const double cx = std::cos(st.heading);
  const double cy = std::sin(st.heading);
  const int n = static_cast<int>(std::ceil(cfg.agent_predict_horizon / cfg.dt));
  double best_dist = std::numeric_limits<double>::infinity();
  double best_tau = 0.0;
  double best_s = 0.0;
  for (int k = 0; k <= n; ++k) {
    const double tau = k * cfg.dt;
    Projection pr = project(cl, cum, st.x + st.v * cx * tau, st.y + st.v * cy * tau);
    if (pr.dist < best_dist) {
      best_dist = pr.dist;
      best_tau = tau;
      best_s = pr.s;
    }
  }
  if (best_dist > cfg.conflict_radius) return false;
  const double fwd_tau = 0.5;
  const Projection d0 = project(cl, cum, st.x, st.y);
  const Projection d1 = project(cl, cum, st.x + st.v * cx * fwd_tau, st.y + st.v * cy * fwd_tau);
  if (best_tau <= 0.5 * cfg.dt && d1.dist > d0.dist + 1e-3) return false;
  out.t_agent = best_tau;
  out.s = best_s;
  return true;
}

double idm(double v, double v_leader, double gap, double v0, const PlannerConfig& c) {
  const double v0e = std::max(v0, kEps);
  const double s_star = c.idm_min_gap +
                        std::max(0.0, v * c.idm_time_headway +
                                          (v * (v - v_leader)) / (2.0 * std::sqrt(std::max(c.idm_max_accel * c.idm_comfort_decel, kEps))));
  const double free_term = std::pow(v / v0e, c.idm_delta);
  const double inter_term =
      std::isfinite(gap) ? std::pow(std::max(s_star, 0.0) / std::max(gap, kEps), 2.0) : 0.0;
  const double a = c.idm_max_accel * (1.0 - free_term - inter_term);
  return clampd(a, -c.idm_comfort_decel, c.idm_max_accel);
}

}  // namespace

RulePlanner::RulePlanner(PlannerConfig config) : cfg_(config) {}

Trajectory RulePlanner::plan(const Scenario& sc) {
  Trajectory traj;
  traj.scenario_id = sc.id;
  traj.source = "rule_planner";
  last_events_.clear();

  const auto& cl = sc.centerline;
  const std::vector<double> cum = cum_arc(cl);
  const int n_steps = static_cast<int>(std::round(cfg_.horizon / cfg_.dt));
  traj.states.reserve(n_steps + 1);
  traj.decisions.reserve(n_steps + 1);

  State ego = sc.ego_init;
  ego.v = std::max(0.0, ego.v);
  ego.heading = wrap_angle(ego.heading);
  ego.a = 0.0;

  double latched_yield_s = -1.0;

  auto compute = [&](const State& ego, double t) {
    struct Result {
      Decision dec;
      double a;
      Context ctx;
    };
    Result res;
    const Projection proj = project(cl, cum, ego.x, ego.y);
    const double s_ego = proj.s;
    const double tangent_h = tangent_at_s(cl, cum, s_ego);
    res.ctx.t = t;
    res.ctx.s_ego = s_ego;
    res.ctx.tangent_heading = tangent_h;

    struct C {
      double s;
      double t_agent;
    };
    std::vector<C> conflicts;
    for (const Agent& ag : sc.agents) {
      State st;
      if (!agent_state_at(ag, t, st)) continue;
      if (st.v < cfg_.min_crossing_speed) continue;
      if (std::fabs(wrap_angle(st.heading - tangent_h)) < cfg_.min_cross_angle) continue;
      Conflict cf;
      if (!predict_conflict(cl, cum, st, cfg_, cf)) continue;
      if (cf.s <= s_ego - 0.5) continue;
      conflicts.push_back({cf.s, std::max(cf.t_agent, 0.0)});
      res.ctx.conflicts.push_back({cf.s, std::max(cf.t_agent, 0.0)});
    }

    bool latch_alive = false;
    if (latched_yield_s >= 0.0) {
      for (const C& c : conflicts) {
        if (std::fabs(c.s - latched_yield_s) < 1.0) {
          latch_alive = true;
          break;
        }
      }
      if (!latch_alive) latched_yield_s = -1.0;
    }

    Decision dec = Decision::FOLLOW;
    double target_stop_s = std::numeric_limits<double>::infinity();

    if (latch_alive) {
      for (const C& c : conflicts) {
        if (std::fabs(c.s - latched_yield_s) < 1.0) target_stop_s = std::min(target_stop_s, c.s);
      }
      dec = (ego.v <= cfg_.stop_speed) ? Decision::STOP : Decision::YIELD;
    } else {
      for (const C& c : conflicts) {
        const double t_ego = (c.s - s_ego) / std::max(ego.v, cfg_.ttc_min_speed);
        if (std::fabs(t_ego - c.t_agent) < cfg_.yield_time_margin) {
          latch_alive = true;
          latched_yield_s = c.s;
          target_stop_s = std::min(target_stop_s, c.s);
        }
      }
      if (latch_alive) {
        dec = (ego.v <= cfg_.stop_speed) ? Decision::STOP : Decision::YIELD;
      } else if (!conflicts.empty()) {
        dec = Decision::ASSERT;
      } else {
        dec = Decision::FOLLOW;
      }
    }

    double a = 0.0;
    if (dec == Decision::YIELD || dec == Decision::STOP) {
      if (ego.v > cfg_.stop_speed) {
        const double d_stop_raw = target_stop_s - s_ego - cfg_.yield_stop_buffer;
        if (d_stop_raw <= 0.0) {
          a = -cfg_.idm_comfort_decel;
        } else {
          a = -(ego.v * ego.v) / (2.0 * std::max(d_stop_raw, cfg_.yield_stop_eps));
        }
      }
      a = clampd(a, -cfg_.idm_comfort_decel, cfg_.idm_max_accel);
    } else if (dec == Decision::ASSERT) {
      a = 0.0;
    } else {
      double gap = std::numeric_limits<double>::infinity();
      double v_leader = 0.0;
      double s_leader = 0.0;
      double a_leader = 0.0;
      for (const Agent& ag : sc.agents) {
        if (ag.type != "vehicle") continue;
        State st;
        if (!agent_state_at(ag, t, st)) continue;
        const Projection p = project(cl, cum, st.x, st.y);
        if (p.dist > cfg_.lane_half_width) continue;
        const double d = p.s - s_ego - cfg_.vehicle_length;
        if (p.s > s_ego && d < gap) {
          gap = d;
          v_leader = st.v;
          s_leader = p.s;
          a_leader = st.a;
        }
      }
      if (std::isfinite(gap)) {
        res.ctx.leader.present = true;
        res.ctx.leader.s = s_leader;
        res.ctx.leader.gap = gap;
        res.ctx.leader.v = v_leader;
        res.ctx.leader.a = a_leader;
      }
      a = idm(ego.v, v_leader, gap, sc.speed_limit, cfg_);
    }
    res.dec = dec;
    res.a = a;
    return res;
  };

  const auto overrides = make_overrides(cfg_);
  double handoff_a = 0.0;
  bool handoff_active = false;

  auto apply_overrides = [&](int step, const State& ego, auto& r) {
    if (cfg_.overrides.empty()) return;
    if (r.dec != Decision::FOLLOW && r.dec != Decision::ASSERT) return;
    bool applied = false;
    for (const auto& ov : overrides) {
      const auto it = cfg_.overrides.find(ov->name());
      if (it == cfg_.overrides.end() || !it->second) continue;
      if (!ov->applicable(sc, ego, r.ctx)) continue;
      const PlannerCommand cmd = ov->apply(sc, ego, r.ctx);
      const GuardProfile gp = predict_guard_profile(sc, ego, r.ctx, cmd, cfg_);
      std::string veto;
      if (gp.min_ttc < cfg_.override_ttc_floor) {
        veto = "veto_ttc_floor";
      } else if (gp.max_jerk > cfg_.override_jerk_max) {
        veto = "veto_jerk";
      } else if (gp.min_pedestrian_dist < cfg_.override_pedestrian_buffer) {
        veto = "veto_pedestrian_buffer";
      }
      if (veto.empty()) {
        r.dec = cmd.decision;
        r.a = cmd.accel;
        handoff_a = cmd.accel;
        handoff_active = true;
        applied = true;
        last_events_.push_back({step, ov->name(), true, "activated"});
      } else {
        last_events_.push_back({step, ov->name(), false, veto});
      }
    }
    if (!applied && handoff_active) {
      const double ramp = cfg_.override_jerk_max * cfg_.dt;
      const double clamped = clampd(r.a, handoff_a - ramp, handoff_a + ramp);
      if (clamped == r.a) {
        handoff_active = false;
      } else {
        r.a = clamped;
        handoff_a = r.a;
      }
    }
  };

  {
    auto r0 = compute(ego, sc.ego_init.t);
    apply_overrides(0, ego, r0);
    ego.a = r0.a;
    traj.states.push_back(ego);
    traj.decisions.push_back(r0.dec);
  }

  for (int step = 1; step <= n_steps; ++step) {
    const double t = sc.ego_init.t + step * cfg_.dt;
    auto r = compute(ego, t);
    apply_overrides(step, ego, r);

    const double v_new = std::max(0.0, ego.v + r.a * cfg_.dt);
    double x_new = ego.x;
    double y_new = ego.y;
    double h_new = ego.heading;
    if (cl.size() >= 2) {
      const double lookahead = clampd(cfg_.lookahead_gain * v_new, cfg_.lookahead_min, cfg_.lookahead_max);
      const double s_ego = project(cl, cum, ego.x, ego.y).s;
      const Point lp = point_at_s(cl, cum, s_ego + lookahead);
      const double alpha = wrap_angle(std::atan2(lp.y - ego.y, lp.x - ego.x) - ego.heading);
      h_new = wrap_angle(ego.heading + (2.0 * v_new * std::sin(alpha) / std::max(lookahead, kEps)) * cfg_.dt);
    }
    x_new = ego.x + v_new * std::cos(h_new) * cfg_.dt;
    y_new = ego.y + v_new * std::sin(h_new) * cfg_.dt;

    State next;
    next.t = t;
    next.x = x_new;
    next.y = y_new;
    next.heading = h_new;
    next.v = v_new;
    next.a = r.a;
    traj.states.push_back(next);
    traj.decisions.push_back(r.dec);
    ego = next;
  }

  return traj;
}

}  // namespace pdt
