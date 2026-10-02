#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>

#include "pdt/overrides.hpp"
#include "pdt/planner.hpp"
#include "pdt/scenario.hpp"
#include "pdt/types.hpp"

namespace py = pybind11;
using namespace pdt;

namespace {

PlannerConfig config_from_dict(py::dict d) {
  PlannerConfig c;
  auto get = [&](const char* key, double& out) {
    if (d.contains(key))
      out = py::cast<double>(d[key]);
  };
  get("dt", c.dt);
  get("horizon", c.horizon);
  get("lookahead_gain", c.lookahead_gain);
  get("lookahead_min", c.lookahead_min);
  get("lookahead_max", c.lookahead_max);
  get("idm_time_headway", c.idm_time_headway);
  get("idm_min_gap", c.idm_min_gap);
  get("idm_max_accel", c.idm_max_accel);
  get("idm_comfort_decel", c.idm_comfort_decel);
  get("idm_delta", c.idm_delta);
  get("yield_time_margin", c.yield_time_margin);
  get("yield_stop_buffer", c.yield_stop_buffer);
  get("yield_stop_eps", c.yield_stop_eps);
  get("conflict_radius", c.conflict_radius);
  get("agent_predict_horizon", c.agent_predict_horizon);
  get("min_crossing_speed", c.min_crossing_speed);
  get("min_cross_angle", c.min_cross_angle);
  get("ttc_min_speed", c.ttc_min_speed);
  get("stop_speed", c.stop_speed);
  get("vehicle_length", c.vehicle_length);
  get("lane_half_width", c.lane_half_width);
  get("override_ttc_floor", c.override_ttc_floor);
  get("override_jerk_max", c.override_jerk_max);
  get("override_pedestrian_buffer", c.override_pedestrian_buffer);
  get("override_predict_horizon", c.override_predict_horizon);
  get("early_brake_leader_decel_threshold", c.early_brake_leader_decel_threshold);
  get("early_brake_max_decel", c.early_brake_max_decel);
  get("early_brake_stop_gap", c.early_brake_stop_gap);
  get("early_brake_wait_gap", c.early_brake_wait_gap);
  get("early_brake_wait_speed", c.early_brake_wait_speed);
  get("early_brake_obstacle_lateral", c.early_brake_obstacle_lateral);
  get("intersection_caution_extended_margin", c.intersection_caution_extended_margin);
  get("intersection_caution_stop_buffer", c.intersection_caution_stop_buffer);
  if (d.contains("overrides")) {
    c.overrides = py::cast<std::map<std::string, bool>>(d["overrides"]);
  }
  return c;
}

} // namespace

PYBIND11_MODULE(pdt_core, m) {
  m.doc() = "C++ core for policy-divergence-triage";

  py::enum_<Decision>(m, "Decision")
      .value("FOLLOW", Decision::FOLLOW)
      .value("YIELD", Decision::YIELD)
      .value("ASSERT", Decision::ASSERT)
      .value("STOP", Decision::STOP);

  py::class_<State>(m, "State")
      .def(py::init<>())
      .def_readwrite("t", &State::t)
      .def_readwrite("x", &State::x)
      .def_readwrite("y", &State::y)
      .def_readwrite("heading", &State::heading)
      .def_readwrite("v", &State::v)
      .def_readwrite("a", &State::a)
      .def("__repr__", [](const State& s) {
        return "State(t=" + std::to_string(s.t) + ", x=" + std::to_string(s.x) + ", y=" + std::to_string(s.y) +
               ", heading=" + std::to_string(s.heading) + ", v=" + std::to_string(s.v) + ", a=" + std::to_string(s.a) +
               ")";
      });

  py::class_<Agent>(m, "Agent")
      .def(py::init<>())
      .def_readwrite("id", &Agent::id)
      .def_readwrite("type", &Agent::type)
      .def_readwrite("track", &Agent::track);

  py::class_<Scenario>(m, "Scenario")
      .def(py::init<>())
      .def_readwrite("id", &Scenario::id)
      .def_readwrite("tag", &Scenario::tag)
      .def_readwrite("ego_init", &Scenario::ego_init)
      .def_readwrite("agents", &Scenario::agents)
      .def_readwrite("centerline", &Scenario::centerline)
      .def_readwrite("speed_limit", &Scenario::speed_limit)
      .def_readwrite("logged_ego", &Scenario::logged_ego);

  py::class_<Trajectory>(m, "Trajectory")
      .def_readonly("scenario_id", &Trajectory::scenario_id)
      .def_readonly("source", &Trajectory::source)
      .def_readonly("states", &Trajectory::states)
      .def_readonly("decisions", &Trajectory::decisions);

  py::class_<PlannerConfig>(m, "PlannerConfig")
      .def(py::init<>())
      .def(py::init([](py::dict d) { return config_from_dict(d); }))
      .def_readwrite("dt", &PlannerConfig::dt)
      .def_readwrite("horizon", &PlannerConfig::horizon)
      .def_readwrite("lookahead_gain", &PlannerConfig::lookahead_gain)
      .def_readwrite("lookahead_min", &PlannerConfig::lookahead_min)
      .def_readwrite("lookahead_max", &PlannerConfig::lookahead_max)
      .def_readwrite("idm_time_headway", &PlannerConfig::idm_time_headway)
      .def_readwrite("idm_min_gap", &PlannerConfig::idm_min_gap)
      .def_readwrite("idm_max_accel", &PlannerConfig::idm_max_accel)
      .def_readwrite("idm_comfort_decel", &PlannerConfig::idm_comfort_decel)
      .def_readwrite("idm_delta", &PlannerConfig::idm_delta)
      .def_readwrite("yield_time_margin", &PlannerConfig::yield_time_margin)
      .def_readwrite("yield_stop_buffer", &PlannerConfig::yield_stop_buffer)
      .def_readwrite("yield_stop_eps", &PlannerConfig::yield_stop_eps)
      .def_readwrite("conflict_radius", &PlannerConfig::conflict_radius)
      .def_readwrite("agent_predict_horizon", &PlannerConfig::agent_predict_horizon)
      .def_readwrite("min_crossing_speed", &PlannerConfig::min_crossing_speed)
      .def_readwrite("min_cross_angle", &PlannerConfig::min_cross_angle)
      .def_readwrite("ttc_min_speed", &PlannerConfig::ttc_min_speed)
      .def_readwrite("stop_speed", &PlannerConfig::stop_speed)
      .def_readwrite("vehicle_length", &PlannerConfig::vehicle_length)
      .def_readwrite("lane_half_width", &PlannerConfig::lane_half_width)
      .def_readwrite("overrides", &PlannerConfig::overrides)
      .def_readwrite("override_ttc_floor", &PlannerConfig::override_ttc_floor)
      .def_readwrite("override_jerk_max", &PlannerConfig::override_jerk_max)
      .def_readwrite("override_pedestrian_buffer", &PlannerConfig::override_pedestrian_buffer)
      .def_readwrite("override_predict_horizon", &PlannerConfig::override_predict_horizon)
      .def_readwrite("early_brake_leader_decel_threshold", &PlannerConfig::early_brake_leader_decel_threshold)
      .def_readwrite("early_brake_max_decel", &PlannerConfig::early_brake_max_decel)
      .def_readwrite("early_brake_stop_gap", &PlannerConfig::early_brake_stop_gap)
      .def_readwrite("early_brake_wait_gap", &PlannerConfig::early_brake_wait_gap)
      .def_readwrite("early_brake_wait_speed", &PlannerConfig::early_brake_wait_speed)
      .def_readwrite("early_brake_obstacle_lateral", &PlannerConfig::early_brake_obstacle_lateral)
      .def_readwrite("intersection_caution_extended_margin", &PlannerConfig::intersection_caution_extended_margin)
      .def_readwrite("intersection_caution_stop_buffer", &PlannerConfig::intersection_caution_stop_buffer);

  py::class_<OverrideEvent>(m, "OverrideEvent")
      .def_readonly("step", &OverrideEvent::step)
      .def_readonly("name", &OverrideEvent::name)
      .def_readonly("active", &OverrideEvent::active)
      .def_readonly("reason", &OverrideEvent::reason);

  py::class_<RulePlanner>(m, "RulePlanner")
      .def(py::init<const PlannerConfig&>(), py::arg("config") = PlannerConfig{})
      .def("plan", &RulePlanner::plan, py::arg("scenario"))
      .def("events", &RulePlanner::events);

  m.def("load_scenarios", &load_scenarios, py::arg("path"));
  m.def("save_scenarios", &save_scenarios, py::arg("scenarios"), py::arg("path"));
}
