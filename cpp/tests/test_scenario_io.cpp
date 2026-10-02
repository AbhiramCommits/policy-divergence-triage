#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "pdt/scenario.hpp"

#ifndef PDT_FIXTURE_DIR
#error "PDT_FIXTURE_DIR must be defined"
#endif

namespace {

TEST(ScenarioIoTest, JsonlRoundtrip) {
  const std::string src = std::string(PDT_FIXTURE_DIR) + "/scenarios.jsonl";
  const auto scenarios = pdt::load_scenarios(src);
  ASSERT_EQ(scenarios.size(), 20u);

  const auto tmp = std::filesystem::temp_directory_path() / "pdt_roundtrip.jsonl";
  pdt::save_scenarios(scenarios, tmp.string());
  const auto reloaded = pdt::load_scenarios(tmp.string());
  std::filesystem::remove(tmp);

  ASSERT_EQ(reloaded.size(), scenarios.size());
  for (size_t i = 0; i < scenarios.size(); ++i) {
    EXPECT_EQ(scenarios[i].id, reloaded[i].id);
    EXPECT_EQ(scenarios[i].tag, reloaded[i].tag);
    EXPECT_EQ(scenarios[i].speed_limit, reloaded[i].speed_limit);
    EXPECT_EQ(scenarios[i].centerline.size(), reloaded[i].centerline.size());
    EXPECT_EQ(scenarios[i].agents.size(), reloaded[i].agents.size());
    EXPECT_EQ(scenarios[i].logged_ego.size(), reloaded[i].logged_ego.size());
    EXPECT_EQ(scenarios[i].ego_init.x, reloaded[i].ego_init.x);
    EXPECT_EQ(scenarios[i].ego_init.y, reloaded[i].ego_init.y);
    EXPECT_EQ(scenarios[i].ego_init.heading, reloaded[i].ego_init.heading);
    EXPECT_EQ(scenarios[i].ego_init.v, reloaded[i].ego_init.v);
    for (size_t k = 0; k < scenarios[i].centerline.size(); ++k) {
      EXPECT_EQ(scenarios[i].centerline[k][0], reloaded[i].centerline[k][0]);
      EXPECT_EQ(scenarios[i].centerline[k][1], reloaded[i].centerline[k][1]);
    }
    for (size_t a = 0; a < scenarios[i].agents.size(); ++a) {
      EXPECT_EQ(scenarios[i].agents[a].id, reloaded[i].agents[a].id);
      EXPECT_EQ(scenarios[i].agents[a].type, reloaded[i].agents[a].type);
      ASSERT_EQ(scenarios[i].agents[a].track.size(), reloaded[i].agents[a].track.size());
      for (size_t k = 0; k < scenarios[i].agents[a].track.size(); ++k) {
        EXPECT_EQ(scenarios[i].agents[a].track[k].t, reloaded[i].agents[a].track[k].t);
        EXPECT_EQ(scenarios[i].agents[a].track[k].x, reloaded[i].agents[a].track[k].x);
        EXPECT_EQ(scenarios[i].agents[a].track[k].y, reloaded[i].agents[a].track[k].y);
        EXPECT_EQ(scenarios[i].agents[a].track[k].heading, reloaded[i].agents[a].track[k].heading);
        EXPECT_EQ(scenarios[i].agents[a].track[k].v, reloaded[i].agents[a].track[k].v);
        EXPECT_EQ(scenarios[i].agents[a].track[k].a, reloaded[i].agents[a].track[k].a);
      }
    }
  }
}

} // namespace
