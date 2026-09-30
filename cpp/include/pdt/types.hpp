#pragma once

#include <array>
#include <string>
#include <vector>

namespace pdt {

struct State {
  double t = 0.0;
  double x = 0.0;
  double y = 0.0;
  double heading = 0.0;
  double v = 0.0;
  double a = 0.0;
};

struct Agent {
  int id = 0;
  std::string type;  // vehicle | pedestrian | cyclist
  std::vector<State> track;
};

struct Scenario {
  std::string id;
  std::string tag;
  State ego_init;
  std::vector<Agent> agents;
  std::vector<std::array<double, 2>> centerline;
  double speed_limit = 0.0;
  std::vector<State> logged_ego;
};

enum class Decision : int { FOLLOW = 0, YIELD = 1, ASSERT = 2, STOP = 3 };

struct Trajectory {
  std::string scenario_id;
  std::string source;
  std::vector<State> states;
  std::vector<Decision> decisions;
};

}  // namespace pdt
