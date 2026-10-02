#include "pdt/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "pdt/spatial.hpp"

namespace pdt {
namespace {

constexpr double kEps = 1e-9;
constexpr double kPi = 3.14159265358979323846;

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

struct Conflict {
  double s;
  double t_agent;
};

bool predict_conflict(const CenterlineIndex& idx, const State& st, const PlannerConfig& cfg, Conflict& out) {
  const double cx = std::cos(st.heading);
  const double cy = std::sin(st.heading);
  const double hmax = cfg.agent_predict_horizon;
  const double dx = st.v * cx;
  const double dy = st.v * cy;
  double best_d2 = std::numeric_limits<double>::infinity();
  double best_tau = 0.0;
  double best_s = 0.0;
  const size_t n = idx.cum.size();
  for (size_t i = 0; i + 1 < n; ++i) {
    const double ax = idx.xs[i];
    const double ay = idx.ys[i];
    const double sx = idx.xs[i + 1] - ax;
    const double sy = idx.ys[i + 1] - ay;
    const double w0x = st.x - ax;
    const double w0y = st.y - ay;
    const double a_ = dx * dx + dy * dy;
    const double b_ = dx * sx + dy * sy;
    const double c_ = sx * sx + sy * sy;
    const double dd = w0x * dx + w0y * dy;
    const double ee = w0x * sx + w0y * sy;
    const double denom = a_ * c_ - b_ * b_;
    double tau = 0.0;
    double t = 0.0;
    if (denom > 1e-12) {
      tau = (b_ * ee - c_ * dd) / denom;
      t = (a_ * ee - b_ * dd) / denom;
      t = clampd(t, 0.0, 1.0);
      tau = clampd(tau, 0.0, hmax);
    } else {
      t = c_ > kEps ? clampd((w0x * sx + w0y * sy) / c_, 0.0, 1.0) : 0.0;
    }
    const double px = st.x + dx * tau;
    const double py = st.y + dy * tau;
    const double qx = ax + sx * t;
    const double qy = ay + sy * t;
    const double ex = px - qx;
    const double ey = py - qy;
    const double d2 = ex * ex + ey * ey;
    if (d2 < best_d2) {
      best_d2 = d2;
      best_tau = tau;
      best_s = idx.cum[i] + t * std::sqrt(c_);
    }
  }
  if (std::sqrt(best_d2) > cfg.conflict_radius) {
    return false;
  }
  if (best_tau <= 0.5 * cfg.dt) {
    const double fwd_tau = 0.5;
    const Proj d0 = project(idx, st.x, st.y);
    const Proj d1 = project(idx, st.x + st.v * cx * fwd_tau, st.y + st.v * cy * fwd_tau);
    if (d1.dist > d0.dist + 1e-3) {
      return false;
    }
  }
  out.t_agent = best_tau;
  out.s = best_s;
  return true;
}

double idm(double v, double v_leader, double gap, double v0, const PlannerConfig& c) {
  const double v0e = std::max(v0, kEps);
  const double s_star =
      c.idm_min_gap + std::max(0.0, v * c.idm_time_headway +
                                        (v * (v - v_leader)) /
                                            (2.0 * std::sqrt(std::max(c.idm_max_accel * c.idm_comfort_decel, kEps))));
  const double free_term = std::pow(v / v0e, c.idm_delta);
  const double inter_term = std::isfinite(gap) ? std::pow(std::max(s_star, 0.0) / std::max(gap, kEps), 2.0) : 0.0;
  const double a = c.idm_max_accel * (1.0 - free_term - inter_term);
  return clampd(a, -c.idm_comfort_decel, c.idm_max_accel);
}

} // namespace

RulePlanner::RulePlanner(PlannerConfig config) : cfg_(std::move(config)) {}

Trajectory RulePlanner::plan(const Scenario& sc) {
  Trajectory traj;
  traj.scenario_id = sc.id;
  traj.source = "rule_planner";
  last_events_.clear();

  const auto& cl = sc.centerline;
  const CenterlineIndex idx = make_centerline_index(cl);
  const int n_steps = static_cast<int>(std::round(cfg_.horizon / cfg_.dt));
  traj.states.reserve(n_steps + 1);
  traj.decisions.reserve(n_steps + 1);

  std::vector<const Agent*> candidates;
  for (const Agent& ag : sc.agents) {
    double vmax = 0.0;
    for (const State& s : ag.track) {
      vmax = std::max(vmax, s.v);
    }
    const double pad = vmax * 0.4 + 0.5;
    double min_d = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < ag.track.size(); k += 4) {
      const double x = ag.track[k].x;
      const double y = ag.track[k].y;
      const double bx = std::max(idx.minx - x, x - idx.maxx);
      const double by = std::max(idx.miny - y, y - idx.maxy);
      const double bbox_d = std::hypot(std::max(bx, 0.0), std::max(by, 0.0));
      if (bbox_d > cfg_.conflict_radius + pad) {
        continue;
      }
      min_d = std::min(min_d, project(idx, x, y).dist);
    }
    if (ag.track.empty()) {
      continue;
    }
    const auto& last = ag.track.back();
    {
      const double bx = std::max(idx.minx - last.x, last.x - idx.maxx);
      const double by = std::max(idx.miny - last.y, last.y - idx.maxy);
      if (std::hypot(std::max(bx, 0.0), std::max(by, 0.0)) <= cfg_.conflict_radius + pad) {
        min_d = std::min(min_d, project(idx, last.x, last.y).dist);
      }
    }
    if (min_d <= cfg_.conflict_radius + pad) {
      candidates.push_back(&ag);
    }
  }

  State ego = sc.ego_init;
  ego.v = std::max(0.0, ego.v);
  ego.heading = wrap_angle(ego.heading);
  ego.a = 0.0;

  double latched_yield_s = -1.0;

  auto compute = [&](const State& ego, double t) {
    struct Result {
      Decision dec = Decision::FOLLOW;
      double a = 0.0;
      Context ctx;
    };
    Result res;
    const Proj proj = project(idx, ego.x, ego.y);
    const double s_ego = proj.s;
    const double tangent_h = tangent_at_s(idx, s_ego);
    res.ctx.t = t;
    res.ctx.s_ego = s_ego;
    res.ctx.tangent_heading = tangent_h;

    struct C {
      double s;
      double t_agent;
    };
    std::vector<C> conflicts;
    for (const Agent* agp : candidates) {
      const Agent& ag = *agp;
      State st;
      if (!agent_state_at(ag, t, st)) {
        continue;
      }
      if (st.v < cfg_.min_crossing_speed) {
        continue;
      }
      if (std::fabs(wrap_angle(st.heading - tangent_h)) < cfg_.min_cross_angle) {
        continue;
      }
      Conflict cf{};
      if (!predict_conflict(idx, st, cfg_, cf)) {
        continue;
      }
      if (cf.s <= s_ego - 0.5) {
        continue;
      }
      conflicts.push_back({cf.s, std::max(cf.t_agent, 0.0)});
      res.ctx.conflicts.push_back({cf.s, std::max(cf.t_agent, 0.0), ag.type == "vehicle"});
    }

    bool latch_alive = false;
    if (latched_yield_s >= 0.0) {
      for (const C& c : conflicts) {
        if (std::fabs(c.s - latched_yield_s) < 1.0) {
          latch_alive = true;
          break;
        }
      }
      if (!latch_alive) {
        latched_yield_s = -1.0;
      }
    }

    Decision dec = Decision::FOLLOW;
    double target_stop_s = std::numeric_limits<double>::infinity();

    if (latch_alive) {
      for (const C& c : conflicts) {
        if (std::fabs(c.s - latched_yield_s) < 1.0) {
          target_stop_s = std::min(target_stop_s, c.s);
        }
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
      for (const Agent* agp : candidates) {
        const Agent& ag = *agp;
        if (ag.type != "vehicle") {
          continue;
        }
        State st;
        if (!agent_state_at(ag, t, st)) {
          continue;
        }
        const Proj p = project(idx, st.x, st.y);
        if (p.dist > cfg_.lane_half_width) {
          continue;
        }
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
    if (cfg_.overrides.empty()) {
      return;
    }
    if (r.dec != Decision::FOLLOW && r.dec != Decision::ASSERT) {
      return;
    }
    bool applied = false;
    for (const auto& ov : overrides) {
      const auto it = cfg_.overrides.find(ov->name());
      if (it == cfg_.overrides.end() || !it->second) {
        continue;
      }
      if (!ov->applicable(sc, ego, r.ctx)) {
        continue;
      }
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
        break;
      }
      last_events_.push_back({step, ov->name(), false, veto});
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
    double h_new = ego.heading;
    double x_new = 0.0;
    double y_new = 0.0;
    if (cl.size() >= 2) {
      const double lookahead = clampd(cfg_.lookahead_gain * v_new, cfg_.lookahead_min, cfg_.lookahead_max);
      const double s_ego = r.ctx.s_ego;
      const std::array<double, 2> lp = point_at_s(idx, s_ego + lookahead);
      const double alpha = wrap_angle(std::atan2(lp[1] - ego.y, lp[0] - ego.x) - ego.heading);
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

} // namespace pdt
