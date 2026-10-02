#include "pdt/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pdt {
namespace {

constexpr double kEps = 1e-9;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;

double wrap_angle(double a) {
  while (a > kPi) {
    a -= 2.0 * kPi;
  }
  while (a < -kPi) {
    a += 2.0 * kPi;
  }
  return a;
}

} // namespace

CenterlineIndex make_centerline_index(const std::vector<std::array<double, 2>>& cl) {
  CenterlineIndex idx;
  const size_t n = cl.size();
  idx.xs.resize(n);
  idx.ys.resize(n);
  idx.cum.resize(n, 0.0);
  if (n == 0) {
    return idx;
  }
  idx.minx = idx.maxx = cl[0][0];
  idx.miny = idx.maxy = cl[0][1];
  idx.xs[0] = cl[0][0];
  idx.ys[0] = cl[0][1];
  for (size_t i = 1; i < n; ++i) {
    idx.xs[i] = cl[i][0];
    idx.ys[i] = cl[i][1];
    idx.cum[i] = idx.cum[i - 1] + std::hypot(cl[i][0] - cl[i - 1][0], cl[i][1] - cl[i - 1][1]);
    idx.minx = std::min(idx.minx, cl[i][0]);
    idx.maxx = std::max(idx.maxx, cl[i][0]);
    idx.miny = std::min(idx.miny, cl[i][1]);
    idx.maxy = std::max(idx.maxy, cl[i][1]);
  }
  idx.w = std::max(1, static_cast<int>((idx.maxx - idx.minx) / idx.cell) + 1);
  idx.h = std::max(1, static_cast<int>((idx.maxy - idx.miny) / idx.cell) + 1);
  idx.cells.assign(static_cast<size_t>(idx.w) * idx.h, {});
  for (int i = 0; i + 1 < static_cast<int>(n); ++i) {
    const double x0 = std::min(cl[i][0], cl[i + 1][0]);
    const double x1 = std::max(cl[i][0], cl[i + 1][0]);
    const double y0 = std::min(cl[i][1], cl[i + 1][1]);
    const double y1 = std::max(cl[i][1], cl[i + 1][1]);
    const int ix0 = std::max(0, static_cast<int>((x0 - idx.minx) / idx.cell));
    const int ix1 = std::min(idx.w - 1, static_cast<int>((x1 - idx.minx) / idx.cell));
    const int iy0 = std::max(0, static_cast<int>((y0 - idx.miny) / idx.cell));
    const int iy1 = std::min(idx.h - 1, static_cast<int>((y1 - idx.miny) / idx.cell));
    for (int iy = iy0; iy <= iy1; ++iy) {
      for (int ix = ix0; ix <= ix1; ++ix) {
        idx.cells[static_cast<size_t>(iy) * idx.w + ix].push_back(i);
      }
    }
  }
  return idx;
}

namespace {

struct ProjectionResult {
  double s = 0.0;
  double dist = kInf;
  double px = 0.0;
  double py = 0.0;
};

void consider_segment(const CenterlineIndex& idx, int i, double x, double y, ProjectionResult& best) {
  const double ax = idx.xs[i];
  const double ay = idx.ys[i];
  const double bx = idx.xs[i + 1];
  const double by = idx.ys[i + 1];
  const double abx = bx - ax;
  const double aby = by - ay;
  const double len2 = abx * abx + aby * aby;
  double t = len2 > kEps ? ((x - ax) * abx + (y - ay) * aby) / len2 : 0.0;
  t = std::max(0.0, std::min(1.0, t));
  const double px = ax + t * abx;
  const double py = ay + t * aby;
  const double d = std::hypot(x - px, y - py);
  if (d < best.dist) {
    best.dist = d;
    best.s = idx.cum[i] + t * std::sqrt(len2);
    best.px = px;
    best.py = py;
  }
}

} // namespace

Proj project(const CenterlineIndex& idx, double x, double y) {
  Proj best;
  best.s = 0.0;
  best.dist = kInf;
  best.px = x;
  best.py = y;
  const size_t n = idx.cum.size();
  if (n < 2) {
    if (n == 1) {
      best.dist = std::hypot(x - idx.xs[0], y - idx.ys[0]);
      best.px = idx.xs[0];
      best.py = idx.ys[0];
    }
    return best;
  }

  const double R0 = 2.0 * idx.cell;
  ProjectionResult res;
  bool found = false;
  const int cx = static_cast<int>((x - idx.minx) / idx.cell);
  const int cy = static_cast<int>((y - idx.miny) / idx.cell);
  if (cx >= -2 && cy >= -2 && cx < idx.w + 2 && cy < idx.h + 2) {
    const int ix0 = std::max(0, cx - 2);
    const int ix1 = std::min(idx.w - 1, cx + 2);
    const int iy0 = std::max(0, cy - 2);
    const int iy1 = std::min(idx.h - 1, cy + 2);
    std::vector<int> cand;
    for (int iy = iy0; iy <= iy1; ++iy) {
      for (int ix = ix0; ix <= ix1; ++ix) {
        const auto& cell = idx.cells[static_cast<size_t>(iy) * idx.w + ix];
        cand.insert(cand.end(), cell.begin(), cell.end());
      }
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    for (int i : cand) {
      consider_segment(idx, i, x, y, res);
      found = true;
    }
    if (found && res.dist < R0) {
      best.s = res.s;
      best.dist = res.dist;
      best.px = res.px;
      best.py = res.py;
      return best;
    }
  }

  res = ProjectionResult{};
  for (int i = 0; i + 1 < static_cast<int>(n); ++i) {
    consider_segment(idx, i, x, y, res);
  }
  best.s = res.s;
  best.dist = res.dist;
  best.px = res.px;
  best.py = res.py;
  return best;
}

std::array<double, 2> point_at_s(const CenterlineIndex& idx, double s) {
  const size_t n = idx.cum.size();
  if (n == 0) {
    return {0.0, 0.0};
  }
  if (n == 1 || idx.cum.back() <= kEps) {
    return {idx.xs[0], idx.ys[0]};
  }
  s = std::max(0.0, std::min(s, idx.cum.back()));
  auto it = std::lower_bound(idx.cum.begin(), idx.cum.end(), s);
  size_t j = static_cast<size_t>(it - idx.cum.begin());
  if (j == 0) {
    return {idx.xs[0], idx.ys[0]};
  }
  size_t i = j - 1;
  if (i + 1 >= n) {
    i = n - 2;
  }
  const double seg = idx.cum[i + 1] - idx.cum[i];
  const double t = seg > kEps ? (s - idx.cum[i]) / seg : 0.0;
  return {idx.xs[i] + t * (idx.xs[i + 1] - idx.xs[i]), idx.ys[i] + t * (idx.ys[i + 1] - idx.ys[i])};
}

double tangent_at_s(const CenterlineIndex& idx, double s) {
  const size_t n = idx.cum.size();
  if (n < 2) {
    return 0.0;
  }
  s = std::max(0.0, std::min(s, idx.cum.back()));
  auto it = std::lower_bound(idx.cum.begin(), idx.cum.end(), s - kEps);
  size_t j = static_cast<size_t>(it - idx.cum.begin());
  size_t i = j > 0 ? j - 1 : 0;
  if (i + 1 >= n) {
    i = n - 2;
  }
  return std::atan2(idx.ys[i + 1] - idx.ys[i], idx.xs[i + 1] - idx.xs[i]);
}

bool agent_state_at(const Agent& ag, double t, State& out) {
  const auto& tr = ag.track;
  if (tr.empty()) {
    return false;
  }
  if (t <= tr.front().t) {
    out = tr.front();
    return true;
  }
  if (t >= tr.back().t) {
    out = tr.back();
    return true;
  }
  size_t lo = 0;
  size_t hi = tr.size();
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (tr[mid].t < t) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const size_t j = lo;
  size_t i = j > 0 ? j - 1 : 0;
  if (i + 1 >= tr.size()) {
    i = tr.size() - 2;
  }
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

} // namespace pdt
