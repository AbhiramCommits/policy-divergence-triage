#pragma once

#include <map>
#include <string>

#include "pdt/overrides.hpp"
#include "pdt/types.hpp"

namespace pdt {

struct PlannerConfig {
  double dt = 0.1;
  double horizon = 8.0;
  double lookahead_gain = 1.0;
  double lookahead_min = 3.0;
  double lookahead_max = 15.0;
  double idm_time_headway = 1.6;
  double idm_min_gap = 2.0;
  double idm_max_accel = 1.5;
  double idm_comfort_decel = 2.0;
  double idm_delta = 4.0;
  double yield_time_margin = 2.0;
  double yield_stop_buffer = 2.0;
  double yield_stop_eps = 0.05;
  double conflict_radius = 2.5;
  double agent_predict_horizon = 8.0;
  double min_crossing_speed = 0.2;
  double min_cross_angle = 0.35;
  double ttc_min_speed = 1.0;
  double stop_speed = 0.05;
  double vehicle_length = 4.5;
  double lane_half_width = 2.5;
  std::map<std::string, bool> overrides;
  double override_ttc_floor = 1.0;
  double override_jerk_max = 20.0;
  double override_pedestrian_buffer = 3.0;
  double override_predict_horizon = 2.0;
  double early_brake_leader_decel_threshold = -2.5;
  double early_brake_max_decel = 3.5;
  double early_brake_stop_gap = 4.0;
  double early_brake_wait_gap = 10.0;
  double early_brake_wait_speed = 1.5;
  double early_brake_obstacle_lateral = 5.0;
};

class RulePlanner {
 public:
  explicit RulePlanner(PlannerConfig config = {});
  Trajectory plan(const Scenario& scenario);
  const std::vector<OverrideEvent>& events() const { return last_events_; }

 private:
  PlannerConfig cfg_;
  std::vector<OverrideEvent> last_events_;
};

}  // namespace pdt
