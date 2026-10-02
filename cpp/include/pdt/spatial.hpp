#pragma once

#include <array>
#include <vector>

#include "pdt/types.hpp"

namespace pdt {

struct CenterlineIndex {
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> cum;
  double cell = 5.0;
  double minx = 0.0;
  double miny = 0.0;
  double maxx = 0.0;
  double maxy = 0.0;
  int w = 0;
  int h = 0;
  std::vector<std::vector<int>> cells;
};

struct Proj {
  double s = 0.0;
  double dist = 0.0;
  double px = 0.0;
  double py = 0.0;
};

CenterlineIndex make_centerline_index(const std::vector<std::array<double, 2>>& cl);

Proj project(const CenterlineIndex& idx, double x, double y);

std::array<double, 2> point_at_s(const CenterlineIndex& idx, double s);

double tangent_at_s(const CenterlineIndex& idx, double s);

bool agent_state_at(const Agent& ag, double t, State& out);

} // namespace pdt
