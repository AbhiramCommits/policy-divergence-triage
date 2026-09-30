#include "pdt/scenario.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace pdt {

using nlohmann::json;

void to_json(json& j, const State& s) {
  j = json::object();
  j["t"] = s.t;
  j["x"] = s.x;
  j["y"] = s.y;
  j["heading"] = s.heading;
  j["v"] = s.v;
  j["a"] = s.a;
}

void from_json(const json& j, State& s) {
  s.t = j.at("t").get<double>();
  s.x = j.at("x").get<double>();
  s.y = j.at("y").get<double>();
  s.heading = j.at("heading").get<double>();
  s.v = j.at("v").get<double>();
  s.a = j.at("a").get<double>();
}

void to_json(json& j, const Agent& a) {
  j = json::object();
  j["id"] = a.id;
  j["type"] = a.type;
  j["track"] = a.track;
}

void from_json(const json& j, Agent& a) {
  a.id = j.at("id").get<int>();
  a.type = j.at("type").get<std::string>();
  a.track = j.at("track").get<std::vector<State>>();
}

void to_json(json& j, const Scenario& sc) {
  j = json::object();
  j["id"] = sc.id;
  j["tag"] = sc.tag;
  j["ego_init"] = sc.ego_init;
  j["agents"] = sc.agents;
  j["centerline"] = json::array();
  for (const auto& pt : sc.centerline) j["centerline"].push_back({pt[0], pt[1]});
  j["speed_limit"] = sc.speed_limit;
  j["logged_ego"] = sc.logged_ego;
}

void from_json(const json& j, Scenario& sc) {
  sc.id = j.at("id").get<std::string>();
  sc.tag = j.at("tag").get<std::string>();
  sc.ego_init = j.at("ego_init").get<State>();
  sc.agents = j.at("agents").get<std::vector<Agent>>();
  sc.centerline.clear();
  for (const auto& pt : j.at("centerline")) {
    sc.centerline.push_back({pt.at(0).get<double>(), pt.at(1).get<double>()});
  }
  sc.speed_limit = j.at("speed_limit").get<double>();
  sc.logged_ego = j.at("logged_ego").get<std::vector<State>>();
}

std::vector<Scenario> load_scenarios(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) throw std::runtime_error("cannot open scenario file: " + path);
  std::vector<Scenario> out;
  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    try {
      out.push_back(json::parse(line).get<Scenario>());
    } catch (const std::exception& e) {
      throw std::runtime_error("failed to parse " + path + " line " + std::to_string(lineno) + ": " + e.what());
    }
  }
  return out;
}

void save_scenarios(const std::vector<Scenario>& scenarios, const std::string& path) {
  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) throw std::runtime_error("cannot open scenario file for writing: " + path);
  for (const auto& sc : scenarios) out << json(sc).dump() << "\n";
}

}  // namespace pdt
