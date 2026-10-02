#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pdt/types.hpp"

namespace pdt {

struct PlannerConfig;

struct PlannerCommand {
  Decision decision = Decision::FOLLOW;
  double accel = 0.0;
};

struct ConflictInfo {
  double s = 0.0;
  double t_agent = 0.0;
  bool vehicle = false;
};

struct LeaderInfo {
  bool present = false;
  double s = 0.0;
  double gap = 0.0;
  double v = 0.0;
  double a = 0.0;
};

struct Context {
  double t = 0.0;
  double s_ego = 0.0;
  double tangent_heading = 0.0;
  std::vector<ConflictInfo> conflicts;
  LeaderInfo leader;
};

struct OverrideEvent {
  int step = 0;
  std::string name;
  bool active = false;
  std::string reason;
};

struct GuardProfile {
  double min_ttc = 0.0;
  double min_pedestrian_dist = 0.0;
  double max_jerk = 0.0;
};

class Override {
public:
  virtual ~Override() = default;
  virtual std::string name() const = 0;
  virtual bool applicable(const Scenario& scenario, const State& ego, const Context& ctx) const = 0;
  virtual PlannerCommand apply(const Scenario& scenario, const State& ego, const Context& ctx) const = 0;
};

std::vector<std::unique_ptr<Override>> make_overrides(const PlannerConfig& cfg);

GuardProfile predict_guard_profile(const Scenario& scenario, const State& ego, const Context& ctx,
                                   const PlannerCommand& cmd, const PlannerConfig& cfg);

} // namespace pdt
