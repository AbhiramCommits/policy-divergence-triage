#pragma once

#include <string>
#include <vector>

#include "pdt/types.hpp"

namespace pdt {

std::vector<Scenario> load_scenarios(const std::string& path);
void save_scenarios(const std::vector<Scenario>& scenarios, const std::string& path);

} // namespace pdt
