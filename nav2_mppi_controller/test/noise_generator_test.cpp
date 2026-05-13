// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cmath>
#include <limits>
#include <thread>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav2_mppi_controller/tools/noise_generator.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"
#include "nav2_mppi_controller/models/optimizer_settings.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/control_sequence.hpp"

// Tests noise generator object

using namespace mppi;  // NOLINT

class NoiseGeneratorTester : public NoiseGenerator
{
public:
  void setFilterInputsForTest(float model_dt, double cutoff_hz, int order)
  {
    settings_.model_dt = model_dt;
    filter_cutoff_frequency_ = cutoff_hz;
    filter_order_ = order;
  }

  bool designLowPassForTest()
  {
    return designButterworthLowPass();
  }

  double getConfiguredCutoff() const
  {
    return filter_cutoff_frequency_;
  }

  bool isFilterConfigured() const
  {
    return lpf_configured_;
  }

  bool isColoredNoiseEnabled() const
  {
    return use_colored_noise_;
  }
};

double meanAbsTemporalDifference(const Eigen::ArrayXXf & data)
{
  if (data.cols() < 2) {
    return 0.0;
  }

  double total = 0.0;
  size_t count = 0;
  for (int row = 0; row < data.rows(); ++row) {
    for (int col = 1; col < data.cols(); ++col) {
      total += std::abs(data(row, col) - data(row, col - 1));
      ++count;
    }
  }

  return count > 0 ? total / static_cast<double>(count) : 0.0;
}

double columnStd(const Eigen::ArrayXXf & data, int col)
{
  if (data.rows() <= 0 || col < 0 || col >= data.cols()) {
    return 0.0;
  }

  double sum = 0.0;
  double sum_sq = 0.0;
  for (int row = 0; row < data.rows(); ++row) {
    const double value = static_cast<double>(data(row, col));
    sum += value;
    sum_sq += value * value;
  }

  const double mean = sum / static_cast<double>(data.rows());
  return std::sqrt(std::max(0.0, sum_sq / static_cast<double>(data.rows()) - mean * mean));
}

TEST(NoiseGeneratorTest, NoiseGeneratorLifecycle)
{
  // Tests shuts down internal thread cleanly
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;

  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  std::string name = "test";
  ParametersHandler handler(node, name);

  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);
  generator.shutdown();
}

TEST(NoiseGeneratorTest, NoiseGeneratorMain)
{
  // Tests shuts down internal thread cleanly
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(true));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  // Populate a potential control sequence
  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(25);
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i;
    control_sequence.vy(i) = i;
    control_sequence.wz(i) = i;
  }

  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  // Request an update with no noise yet generated, should result in identical outputs
  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);  // sets initial sizing and zeros out noises
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // save initial state
  auto initial_cvx_0 = state.cvx(0);
  auto initial_cvy_0 = state.cvy(0);
  auto initial_cwz_0 = state.cwz(0);
  auto initial_cvx_9 = state.cvx(0, 9);
  auto initial_cvy_9 = state.cvy(0, 9);
  auto initial_cwz_9 = state.cwz(0, 9);

  EXPECT_NE(state.cvx(0), 0);
  EXPECT_EQ(state.cvy(0), 0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0), 0);
  EXPECT_NE(state.cvx(0, 9), 9);
  EXPECT_EQ(state.cvy(0, 9), 9);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), 9);

  EXPECT_NEAR(state.cvx(0), 0, 0.3);
  EXPECT_NEAR(state.cwz(0), 0, 0.3);
  EXPECT_NEAR(state.cvx(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cwz(0, 9), 9, 0.3);

  // Request an update with noise requested
  generator.generateNextNoises();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // Ensure the state has changed after generating new noises
  EXPECT_NE(state.cvx(0), initial_cvx_0);
  EXPECT_EQ(state.cvy(0), initial_cvy_0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0), initial_cwz_0);
  EXPECT_NE(state.cvx(0, 9), initial_cvx_9);
  EXPECT_EQ(state.cvy(0, 9), initial_cvy_9);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), initial_cwz_9);


  // Test holonomic setting
  generator.reset(settings, true);  // Now holonomically
  generator.generateNextNoises();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);
  EXPECT_NE(state.cvx(0), 0);
  EXPECT_NE(state.cvy(0), 0);  // Now populated in non-holonomic
  EXPECT_NE(state.cwz(0), 0);
  EXPECT_NE(state.cvx(0, 9), 9);
  EXPECT_NE(state.cvy(0, 9), 9);  // Now populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), 9);

  EXPECT_NEAR(state.cvx(0), 0, 0.3);
  EXPECT_NEAR(state.cvy(0), 0, 0.3);
  EXPECT_NEAR(state.cwz(0), 0, 0.3);
  EXPECT_NEAR(state.cvx(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cvy(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cwz(0, 9), 9, 0.3);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, RegenerateResetProvidesImmediateNoise)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node_immediate_regen");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(true));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);
  generator.setNoisedControls(state, control_sequence);

  EXPECT_GT(meanAbsTemporalDifference(state.cvx), 1e-4);
  EXPECT_GT(meanAbsTemporalDifference(state.cwz), 1e-4);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, RegenerateNextNoiseIsAvailableWithoutSleep)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node_immediate_next_regen");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(true));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);
  generator.setNoisedControls(state, control_sequence);
  const auto first_vx = state.cvx;
  const auto first_wz = state.cwz;

  generator.generateNextNoises();
  generator.setNoisedControls(state, control_sequence);

  EXPECT_GT((state.cvx - first_vx).abs().mean(), 1e-4);
  EXPECT_GT((state.cwz - first_wz).abs().mean(), 1e-4);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, LowPassFilterSmoothsPerturbations)
{
  auto filtered_node = std::make_shared<nav2::LifecycleNode>("filtered_node");
  filtered_node->declare_parameter("filtered_ns.regenerate_noises", rclcpp::ParameterValue(false));
  filtered_node->declare_parameter("filtered_ns.use_low_pass_filter", rclcpp::ParameterValue(true));
  filtered_node->declare_parameter(
    "filtered_ns.filter_cutoff_frequency", rclcpp::ParameterValue(0.8));
  filtered_node->declare_parameter("filtered_ns.filter_order", rclcpp::ParameterValue(2));
  std::string filtered_name = "filtered";
  ParametersHandler filtered_handler(filtered_node, filtered_name);

  auto raw_node = std::make_shared<nav2::LifecycleNode>("raw_node");
  raw_node->declare_parameter("raw_ns.regenerate_noises", rclcpp::ParameterValue(false));
  raw_node->declare_parameter("raw_ns.use_low_pass_filter", rclcpp::ParameterValue(false));
  std::string raw_name = "raw";
  ParametersHandler raw_handler(raw_node, raw_name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 128;
  settings.time_steps = 80;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State filtered_state, raw_state;
  filtered_state.reset(settings.batch_size, settings.time_steps);
  raw_state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator filtered_generator;
  filtered_generator.initialize(settings, false, "filtered_ns", &filtered_handler);
  filtered_generator.reset(settings, false);
  filtered_generator.setNoisedControls(filtered_state, control_sequence);
  filtered_generator.shutdown();

  NoiseGenerator raw_generator;
  raw_generator.initialize(settings, false, "raw_ns", &raw_handler);
  raw_generator.reset(settings, false);
  raw_generator.setNoisedControls(raw_state, control_sequence);
  raw_generator.shutdown();

  const double filtered_vx_diff = meanAbsTemporalDifference(filtered_state.cvx);
  const double raw_vx_diff = meanAbsTemporalDifference(raw_state.cvx);
  const double filtered_wz_diff = meanAbsTemporalDifference(filtered_state.cwz);
  const double raw_wz_diff = meanAbsTemporalDifference(raw_state.cwz);

  EXPECT_LT(filtered_vx_diff, raw_vx_diff);
  EXPECT_LT(filtered_wz_diff, raw_wz_diff);

  // Repeat in holonomic mode to verify vy smoothing path.
  filtered_state.reset(settings.batch_size, settings.time_steps);
  raw_state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator filtered_holo_generator;
  filtered_holo_generator.initialize(settings, true, "filtered_ns", &filtered_handler);
  filtered_holo_generator.reset(settings, true);
  filtered_holo_generator.setNoisedControls(filtered_state, control_sequence);
  filtered_holo_generator.shutdown();

  NoiseGenerator raw_holo_generator;
  raw_holo_generator.initialize(settings, true, "raw_ns", &raw_handler);
  raw_holo_generator.reset(settings, true);
  raw_holo_generator.setNoisedControls(raw_state, control_sequence);
  raw_holo_generator.shutdown();

  const double filtered_vy_diff = meanAbsTemporalDifference(filtered_state.cvy);
  const double raw_vy_diff = meanAbsTemporalDifference(raw_state.cvy);

  EXPECT_LT(filtered_vy_diff, raw_vy_diff);
}

TEST(NoiseGeneratorTest, LowPassFilterWarmupKeepsNearTermPerturbationAuthority)
{
  auto node = std::make_shared<nav2::LifecycleNode>("lp_warmup_node");
  node->declare_parameter("lp.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("lp.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("lp.filter_cutoff_frequency", rclcpp::ParameterValue(3.0));
  node->declare_parameter("lp.filter_order", rclcpp::ParameterValue(4));
  std::string name = "lp";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 3000;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator generator;
  generator.initialize(settings, false, "lp", &handler);
  generator.reset(settings, false);
  generator.setNoisedControls(state, control_sequence);
  generator.shutdown();

  const double first_vx_std = columnStd(state.cvx, 0);
  const double settled_vx_std = columnStd(state.cvx, 20);
  const double first_wz_std = columnStd(state.cwz, 0);
  const double settled_wz_std = columnStd(state.cwz, 20);

  EXPECT_GT(first_vx_std, settled_vx_std * 0.5);
  EXPECT_GT(first_wz_std, settled_wz_std * 0.5);
}


TEST(NoiseGeneratorTest, CutoffClampedToNyquist)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("test_name.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("test_name.filter_cutoff_frequency", rclcpp::ParameterValue(1000.0));
  node->declare_parameter("test_name.filter_order", rclcpp::ParameterValue(2));
  std::string name = "test";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 64;
  settings.time_steps = 32;
  settings.model_dt = 0.1;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  NoiseGeneratorTester generator;
  generator.initialize(settings, false, "test_name", &handler);

  const double nyquist = 1.0 / (2.0 * settings.model_dt);
  EXPECT_TRUE(generator.isFilterConfigured());
  EXPECT_GT(generator.getConfiguredCutoff(), 0.0);
  EXPECT_LT(generator.getConfiguredCutoff(), nyquist);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, LowPassFilterDisabledWithNonPositiveModelDt)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node_invalid_dt");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("test_name.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("test_name.filter_cutoff_frequency", rclcpp::ParameterValue(1.0));
  node->declare_parameter("test_name.filter_order", rclcpp::ParameterValue(2));
  std::string name = "test";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 16;
  settings.time_steps = 8;
  settings.model_dt = 0.0;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  NoiseGeneratorTester generator;
  generator.initialize(settings, false, "test_name", &handler);

  EXPECT_FALSE(generator.isFilterConfigured());
  generator.shutdown();
}

TEST(NoiseGeneratorTest, LowPassFilterDisabledWithNonPositiveCutoff)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node_invalid_cutoff");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("test_name.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("test_name.filter_cutoff_frequency", rclcpp::ParameterValue(0.0));
  node->declare_parameter("test_name.filter_order", rclcpp::ParameterValue(2));
  std::string name = "test";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 16;
  settings.time_steps = 8;
  settings.model_dt = 0.1;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  NoiseGeneratorTester generator;
  generator.initialize(settings, false, "test_name", &handler);

  EXPECT_FALSE(generator.isFilterConfigured());
  generator.shutdown();
}

TEST(NoiseGeneratorTest, LowPassFilterDisabledWithInvalidOrder)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node_invalid_order");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("test_name.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("test_name.filter_cutoff_frequency", rclcpp::ParameterValue(1.0));
  node->declare_parameter("test_name.filter_order", rclcpp::ParameterValue(0));
  std::string name = "test";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 16;
  settings.time_steps = 8;
  settings.model_dt = 0.1;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  NoiseGeneratorTester generator;
  generator.initialize(settings, false, "test_name", &handler);

  EXPECT_FALSE(generator.isFilterConfigured());
  generator.shutdown();
}

TEST(NoiseGeneratorTest, LowPassFilterDisabledWhenDesignFails)
{
  auto node = std::make_shared<nav2::LifecycleNode>("node_design_fail");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("test_name.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("test_name.filter_cutoff_frequency", rclcpp::ParameterValue(1.0));
  node->declare_parameter("test_name.filter_order", rclcpp::ParameterValue(2));
  std::string name = "test";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 16;
  settings.time_steps = 8;
  settings.model_dt = std::numeric_limits<float>::quiet_NaN();
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  NoiseGeneratorTester generator;
  generator.initialize(settings, false, "test_name", &handler);

  EXPECT_FALSE(generator.isFilterConfigured());
  generator.shutdown();
}

TEST(NoiseGeneratorTest, DesignRejectsInvalidOrderInput)
{
  NoiseGeneratorTester generator;
  generator.setFilterInputsForTest(0.1f, 1.0, 0);
  EXPECT_FALSE(generator.designLowPassForTest());
}

TEST(NoiseGeneratorTest, NoiseGeneratorMainNoRegenerate)
{
  // This time with no regeneration of noises
  auto node = std::make_shared<nav2::LifecycleNode>("node");
  node->declare_parameter("test_name.regenerate_noises", rclcpp::ParameterValue(false));
  std::string name = "test";
  ParametersHandler handler(node, name);
  NoiseGenerator generator;
  mppi::models::OptimizerSettings settings;
  settings.batch_size = 100;
  settings.time_steps = 25;
  settings.sampling_std.vx = 0.1;
  settings.sampling_std.vy = 0.1;
  settings.sampling_std.wz = 0.1;

  // Populate a potential control sequence
  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(25);
  for (unsigned int i = 0; i != control_sequence.vx.rows(); i++) {
    control_sequence.vx(i) = i;
    control_sequence.vy(i) = i;
    control_sequence.wz(i) = i;
  }

  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  // Request an update with no noise yet generated, should result in identical outputs
  generator.initialize(settings, false, "test_name", &handler);
  generator.reset(settings, false);  // sets initial sizing and zeros out noises
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // save initial state
  auto initial_cvx_0 = state.cvx(0);
  auto initial_cvy_0 = state.cvy(0);
  auto initial_cwz_0 = state.cwz(0);
  auto initial_cvx_9 = state.cvx(0, 9);
  auto initial_cvy_9 = state.cvy(0, 9);
  auto initial_cwz_9 = state.cwz(0, 9);

  EXPECT_NE(state.cvx(0), 0);
  EXPECT_EQ(state.cvy(0), 0);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0), 0);
  EXPECT_NE(state.cvx(0, 9), 9);
  EXPECT_EQ(state.cvy(0, 9), 9);  // Not populated in non-holonomic
  EXPECT_NE(state.cwz(0, 9), 9);

  EXPECT_NEAR(state.cvx(0), 0, 0.3);
  EXPECT_NEAR(state.cwz(0), 0, 0.3);
  EXPECT_NEAR(state.cvx(0, 9), 9, 0.3);
  EXPECT_NEAR(state.cwz(0, 9), 9, 0.3);

  // this doesn't work if regenerate_noises is false
  generator.generateNextNoises();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.setNoisedControls(state, control_sequence);

  // Ensure the state has changed after generating new noises
  EXPECT_EQ(state.cvx(0), initial_cvx_0);
  EXPECT_EQ(state.cvy(0), initial_cvy_0);  // Not populated in non-holonomic
  EXPECT_EQ(state.cwz(0), initial_cwz_0);
  EXPECT_EQ(state.cvx(0, 9), initial_cvx_9);
  EXPECT_EQ(state.cvy(0, 9), initial_cvy_9);  // Not populated in non-holonomic
  EXPECT_EQ(state.cwz(0, 9), initial_cwz_9);

  generator.shutdown();
}

TEST(NoiseGeneratorTest, ColoredNoiseSmoothsPerturbations)
{
  auto colored_node = std::make_shared<nav2::LifecycleNode>("colored_smooth_node");
  colored_node->declare_parameter("colored_ns.regenerate_noises", rclcpp::ParameterValue(false));
  colored_node->declare_parameter("colored_ns.use_colored_noise", rclcpp::ParameterValue(true));
  colored_node->declare_parameter(
    "colored_ns.colored_noise_exponent_vx", rclcpp::ParameterValue(3.0));
  colored_node->declare_parameter(
    "colored_ns.colored_noise_exponent_vy", rclcpp::ParameterValue(3.0));
  colored_node->declare_parameter(
    "colored_ns.colored_noise_exponent_wz", rclcpp::ParameterValue(3.0));
  std::string colored_name = "colored";
  ParametersHandler colored_handler(colored_node, colored_name);

  auto raw_node = std::make_shared<nav2::LifecycleNode>("raw_colored_node");
  raw_node->declare_parameter("raw_ns.regenerate_noises", rclcpp::ParameterValue(false));
  std::string raw_name = "raw";
  ParametersHandler raw_handler(raw_node, raw_name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 200;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State colored_state, raw_state;
  colored_state.reset(settings.batch_size, settings.time_steps);
  raw_state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator colored_generator;
  colored_generator.initialize(settings, false, "colored_ns", &colored_handler);
  colored_generator.reset(settings, false);
  colored_generator.setNoisedControls(colored_state, control_sequence);
  colored_generator.shutdown();

  NoiseGenerator raw_generator;
  raw_generator.initialize(settings, false, "raw_ns", &raw_handler);
  raw_generator.reset(settings, false);
  raw_generator.setNoisedControls(raw_state, control_sequence);
  raw_generator.shutdown();

  const double colored_vx_diff = meanAbsTemporalDifference(colored_state.cvx);
  const double raw_vx_diff = meanAbsTemporalDifference(raw_state.cvx);
  const double colored_wz_diff = meanAbsTemporalDifference(colored_state.cwz);
  const double raw_wz_diff = meanAbsTemporalDifference(raw_state.cwz);

  EXPECT_LT(colored_vx_diff, raw_vx_diff);
  EXPECT_LT(colored_wz_diff, raw_wz_diff);
}

TEST(NoiseGeneratorTest, ColoredNoiseReferenceOffsetDecaySuppressesStrideSample)
{
  auto node = std::make_shared<nav2::LifecycleNode>("colored_offset_node");
  node->declare_parameter("cn_offset.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("cn_offset.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("cn_offset.colored_noise_exponent_vx", rclcpp::ParameterValue(2.0));
  node->declare_parameter("cn_offset.colored_noise_exponent_wz", rclcpp::ParameterValue(2.0));
  std::string name = "cn_offset";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 200;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator generator;
  generator.initialize(settings, false, "cn_offset", &handler);
  generator.reset(settings, false);
  generator.setNoisedControls(state, control_sequence);
  generator.shutdown();

  const double offset_vx_std = columnStd(state.cvx, 1);
  const double later_vx_std = columnStd(state.cvx, 12);
  const double offset_wz_std = columnStd(state.cwz, 1);
  const double later_wz_std = columnStd(state.cwz, 12);

  EXPECT_LT(offset_vx_std, later_vx_std * 0.25);
  EXPECT_LT(offset_wz_std, later_wz_std * 0.25);
}

TEST(NoiseGeneratorTest, ColoredNoiseGammaZeroIsGaussian)
{
  auto colored_node = std::make_shared<nav2::LifecycleNode>("gamma_zero_colored");
  colored_node->declare_parameter("gzero_ns.regenerate_noises", rclcpp::ParameterValue(false));
  colored_node->declare_parameter("gzero_ns.use_colored_noise", rclcpp::ParameterValue(true));
  colored_node->declare_parameter(
    "gzero_ns.colored_noise_exponent_vx", rclcpp::ParameterValue(0.0));
  colored_node->declare_parameter(
    "gzero_ns.colored_noise_exponent_wz", rclcpp::ParameterValue(0.0));
  std::string colored_name = "gzero";
  ParametersHandler colored_handler(colored_node, colored_name);

  auto raw_node = std::make_shared<nav2::LifecycleNode>("gamma_zero_raw");
  raw_node->declare_parameter("raw_ns.regenerate_noises", rclcpp::ParameterValue(false));
  std::string raw_name = "raw";
  ParametersHandler raw_handler(raw_node, raw_name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 200;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State colored_state, raw_state;
  colored_state.reset(settings.batch_size, settings.time_steps);
  raw_state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator colored_generator;
  colored_generator.initialize(settings, false, "gzero_ns", &colored_handler);
  colored_generator.reset(settings, false);
  colored_generator.setNoisedControls(colored_state, control_sequence);
  colored_generator.shutdown();

  NoiseGenerator raw_generator;
  raw_generator.initialize(settings, false, "raw_ns", &raw_handler);
  raw_generator.reset(settings, false);
  raw_generator.setNoisedControls(raw_state, control_sequence);
  raw_generator.shutdown();

  // With gamma=0, colored noise should degenerate to white noise,
  // so temporal difference should be similar to raw Gaussian
  const double colored_vx_diff = meanAbsTemporalDifference(colored_state.cvx);
  const double raw_vx_diff = meanAbsTemporalDifference(raw_state.cvx);

  // Allow 50% tolerance since both are stochastic
  EXPECT_NEAR(colored_vx_diff, raw_vx_diff, raw_vx_diff * 0.5);
}

TEST(NoiseGeneratorTest, ColoredNoiseHolonomicSmooths)
{
  auto node = std::make_shared<nav2::LifecycleNode>("colored_holo_node");
  node->declare_parameter("cn_holo.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("cn_holo.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("cn_holo.colored_noise_exponent_vx", rclcpp::ParameterValue(3.0));
  node->declare_parameter("cn_holo.colored_noise_exponent_vy", rclcpp::ParameterValue(3.0));
  node->declare_parameter("cn_holo.colored_noise_exponent_wz", rclcpp::ParameterValue(3.0));
  std::string cn_name = "cn_holo";
  ParametersHandler cn_handler(node, cn_name);

  auto raw_node = std::make_shared<nav2::LifecycleNode>("raw_holo_node");
  raw_node->declare_parameter("raw_holo.regenerate_noises", rclcpp::ParameterValue(false));
  std::string raw_name = "raw_holo";
  ParametersHandler raw_handler(raw_node, raw_name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 200;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State cn_state, raw_state;
  cn_state.reset(settings.batch_size, settings.time_steps);
  raw_state.reset(settings.batch_size, settings.time_steps);

  // Holonomic mode
  NoiseGenerator cn_gen;
  cn_gen.initialize(settings, true, "cn_holo", &cn_handler);
  cn_gen.reset(settings, true);
  cn_gen.setNoisedControls(cn_state, control_sequence);
  cn_gen.shutdown();

  NoiseGenerator raw_gen;
  raw_gen.initialize(settings, true, "raw_holo", &raw_handler);
  raw_gen.reset(settings, true);
  raw_gen.setNoisedControls(raw_state, control_sequence);
  raw_gen.shutdown();

  // vy should also be smoother
  const double cn_vy_diff = meanAbsTemporalDifference(cn_state.cvy);
  const double raw_vy_diff = meanAbsTemporalDifference(raw_state.cvy);
  EXPECT_LT(cn_vy_diff, raw_vy_diff);

  // vx and wz should still be smoother
  EXPECT_LT(meanAbsTemporalDifference(cn_state.cvx), meanAbsTemporalDifference(raw_state.cvx));
  EXPECT_LT(meanAbsTemporalDifference(cn_state.cwz), meanAbsTemporalDifference(raw_state.cwz));
}

TEST(NoiseGeneratorTest, ColoredNoiseEdgeCases)
{
  // Test with minimal T=2 and batch_size=1; should not crash.
  auto node = std::make_shared<nav2::LifecycleNode>("edge_node");
  node->declare_parameter("edge.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("edge.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("edge.colored_noise_exponent_vx", rclcpp::ParameterValue(2.0));
  node->declare_parameter("edge.colored_noise_exponent_wz", rclcpp::ParameterValue(2.0));
  std::string name = "edge";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 1;
  settings.time_steps = 2;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator gen;
  gen.initialize(settings, false, "edge", &handler);
  gen.reset(settings, false);
  EXPECT_NO_THROW(gen.setNoisedControls(state, control_sequence));
  gen.shutdown();

  // Test with odd T=3; exercises the alternate iFFT last-bin path.
  settings.batch_size = 4;
  settings.time_steps = 3;
  state.reset(settings.batch_size, settings.time_steps);
  control_sequence.reset(settings.time_steps);

  NoiseGenerator gen2;
  gen2.initialize(settings, false, "edge", &handler);
  gen2.reset(settings, false);
  EXPECT_NO_THROW(gen2.setNoisedControls(state, control_sequence));
  gen2.shutdown();
}

TEST(NoiseGeneratorTest, ColoredNoiseMonotonicity)
{
  // Higher gamma should produce smoother (lower temporal diff) noise.
  auto make_generator = [](const std::string & ns_suffix, double gamma) {
      auto node = std::make_shared<nav2::LifecycleNode>("mono_" + ns_suffix);
      std::string ns = "mono_" + ns_suffix;
      node->declare_parameter(ns + ".regenerate_noises", rclcpp::ParameterValue(false));
      node->declare_parameter(ns + ".use_colored_noise", rclcpp::ParameterValue(true));
      node->declare_parameter(ns + ".colored_noise_exponent_vx", rclcpp::ParameterValue(gamma));
      node->declare_parameter(ns + ".colored_noise_exponent_wz", rclcpp::ParameterValue(gamma));
      return std::make_pair(node, ns);
    };

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 300;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  auto run_with_gamma = [&](double gamma, std::string suffix) -> double {
      auto [node, ns] = make_generator(suffix, gamma);
      ParametersHandler handler(node, suffix);
      mppi::models::ControlSequence cs;
      cs.reset(settings.time_steps);
      mppi::models::State state;
      state.reset(settings.batch_size, settings.time_steps);
      NoiseGenerator gen;
      gen.initialize(settings, false, ns, &handler);
      gen.reset(settings, false);
      gen.setNoisedControls(state, cs);
      gen.shutdown();
      return meanAbsTemporalDifference(state.cvx);
    };

  double diff_gamma1 = run_with_gamma(1.0, "g1");
  double diff_gamma4 = run_with_gamma(4.0, "g4");

  // gamma=4 should be significantly smoother than gamma=1.
  EXPECT_LT(diff_gamma4, diff_gamma1);
}

TEST(NoiseGeneratorTest, ColoredNoiseMutualExclusionWithLPFilter)
{
  // When both use_colored_noise and use_low_pass_filter are true,
  // colored noise should win and LP filter should be disabled
  auto node = std::make_shared<nav2::LifecycleNode>("mutex_node");
  node->declare_parameter("mutex_ns.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("mutex_ns.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("mutex_ns.use_low_pass_filter", rclcpp::ParameterValue(true));
  node->declare_parameter("mutex_ns.filter_cutoff_frequency", rclcpp::ParameterValue(1.0));
  node->declare_parameter("mutex_ns.filter_order", rclcpp::ParameterValue(2));
  node->declare_parameter("mutex_ns.colored_noise_exponent_vx", rclcpp::ParameterValue(2.0));
  node->declare_parameter("mutex_ns.colored_noise_exponent_wz", rclcpp::ParameterValue(2.0));
  std::string name = "mutex";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 16;
  settings.time_steps = 32;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.2;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = 0.2;

  NoiseGeneratorTester gen;
  gen.initialize(settings, false, "mutex_ns", &handler);

  // Colored noise should be enabled, LP filter disabled
  EXPECT_TRUE(gen.isColoredNoiseEnabled());
  EXPECT_FALSE(gen.isFilterConfigured());

  gen.shutdown();
}


// === Math-verifying tests ===
// These tests verify the correctness of the colored noise algorithm beyond
// simple smoothness checks: variance/std normalization, spectral PSD shape,
// and output validity (finite, zero-mean).

TEST(NoiseGeneratorTest, ColoredNoiseVarianceNormalization)
{
  // The output std should match the requested sigma within ~20%
  // when averaged over many trajectories.
  auto node = std::make_shared<nav2::LifecycleNode>("var_norm_node");
  node->declare_parameter("vn.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("vn.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("vn.colored_noise_exponent_vx", rclcpp::ParameterValue(2.0));
  node->declare_parameter("vn.colored_noise_exponent_wz", rclcpp::ParameterValue(2.0));
  std::string name = "vn";
  ParametersHandler handler(node, name);

  const float target_std_vx = 0.3f;
  const float target_std_wz = 0.5f;

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 1000;  // Large batch for statistical accuracy
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = target_std_vx;
  settings.sampling_std.vy = 0.2;
  settings.sampling_std.wz = target_std_wz;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator gen;
  gen.initialize(settings, false, "vn", &handler);
  gen.reset(settings, false);
  gen.setNoisedControls(state, control_sequence);
  gen.shutdown();

  // Compute empirical std across all elements (noise + 0 control = noise)
  // state.cvx is [batch, time_steps]
  double sum_vx = 0.0, sum_sq_vx = 0.0;
  double sum_wz = 0.0, sum_sq_wz = 0.0;
  int n = settings.batch_size * settings.time_steps;
  for (unsigned int r = 0; r < settings.batch_size; ++r) {
    for (unsigned int c = 0; c < settings.time_steps; ++c) {
      double vx = static_cast<double>(state.cvx(r, c));
      double wz = static_cast<double>(state.cwz(r, c));
      sum_vx += vx;
      sum_sq_vx += vx * vx;
      sum_wz += wz;
      sum_sq_wz += wz * wz;
    }
  }
  double mean_vx = sum_vx / n;
  double mean_wz = sum_wz / n;
  double empirical_std_vx = std::sqrt(sum_sq_vx / n - mean_vx * mean_vx);
  double empirical_std_wz = std::sqrt(sum_sq_wz / n - mean_wz * mean_wz);

  // Within 20% of requested std (stochastic, but 1000 trajectories is enough)
  EXPECT_NEAR(empirical_std_vx, target_std_vx, target_std_vx * 0.20);
  EXPECT_NEAR(empirical_std_wz, target_std_wz, target_std_wz * 0.20);
}

TEST(NoiseGeneratorTest, ColoredNoiseSpectralPSDShape)
{
  // Verify that the frequency-domain power spectrum of the generated noise
  // actually decreases with frequency for gamma > 0.
  // We compute the DFT of each trajectory, average the power spectrum,
  // and check that the low-frequency band has more power than the high band.
  auto node = std::make_shared<nav2::LifecycleNode>("psd_node");
  node->declare_parameter("psd.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("psd.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("psd.colored_noise_exponent_vx", rclcpp::ParameterValue(3.0));
  node->declare_parameter("psd.colored_noise_exponent_wz", rclcpp::ParameterValue(3.0));
  std::string name = "psd";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 500;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator gen;
  gen.initialize(settings, false, "psd", &handler);
  gen.reset(settings, false);
  gen.setNoisedControls(state, control_sequence);
  gen.shutdown();

  // Compute the average power spectrum via DFT
  const unsigned int T = settings.time_steps;
  const unsigned int N = T / 2 + 1;
  std::vector<double> avg_psd(N, 0.0);

  for (unsigned int row = 0; row < settings.batch_size; ++row) {
    for (unsigned int k = 0; k < N; ++k) {
      double re = 0.0, im = 0.0;
      for (unsigned int t = 0; t < T; ++t) {
        double angle = 2.0 * M_PI * k * t / T;
        double val = static_cast<double>(state.cvx(row, t));
        re += val * std::cos(angle);
        im -= val * std::sin(angle);
      }
      avg_psd[k] += (re * re + im * im) / (T * T);
    }
  }
  for (unsigned int k = 0; k < N; ++k) {
    avg_psd[k] /= settings.batch_size;
  }

  // Split into low-freq band (bins 1..N/4) and high-freq band (bins 3N/4..N-1)
  // Skip DC bin (k=0) as it's just the mean
  double low_power = 0.0, high_power = 0.0;
  int low_count = 0, high_count = 0;
  for (unsigned int k = 1; k <= N / 4; ++k) {
    low_power += avg_psd[k];
    low_count++;
  }
  for (unsigned int k = 3 * N / 4; k < N; ++k) {
    high_power += avg_psd[k];
    high_count++;
  }

  double avg_low = low_power / std::max(low_count, 1);
  double avg_high = high_power / std::max(high_count, 1);

  // For gamma=3, the PSD ratio between low and high bands should be large.
  // f_low ~ 1/(4*56), f_high ~ 3/(4*56), ratio ~ (f_high/f_low)^3 = 3^3 = 27x
  // In practice, with averaging, we expect at least 5x power difference.
  EXPECT_GT(avg_low, avg_high * 5.0)
    << "Low-freq power (" << avg_low
    << ") should be much larger than high-freq power (" << avg_high
    << ") for gamma=3 colored noise";
}

TEST(NoiseGeneratorTest, ColoredNoiseOutputIsFiniteAndZeroMean)
{
  // Verify that the iFFT output contains no NaN/Inf values (real-valued check)
  // and that the mean is near zero (unbiased noise).
  auto node = std::make_shared<nav2::LifecycleNode>("finite_node");
  node->declare_parameter("fi.regenerate_noises", rclcpp::ParameterValue(false));
  node->declare_parameter("fi.use_colored_noise", rclcpp::ParameterValue(true));
  node->declare_parameter("fi.colored_noise_exponent_vx", rclcpp::ParameterValue(4.0));
  node->declare_parameter("fi.colored_noise_exponent_wz", rclcpp::ParameterValue(4.0));
  std::string name = "fi";
  ParametersHandler handler(node, name);

  mppi::models::OptimizerSettings settings;
  settings.batch_size = 200;
  settings.time_steps = 56;
  settings.model_dt = 0.05;
  settings.sampling_std.vx = 0.25;
  settings.sampling_std.vy = 0.25;
  settings.sampling_std.wz = 0.25;

  mppi::models::ControlSequence control_sequence;
  control_sequence.reset(settings.time_steps);
  mppi::models::State state;
  state.reset(settings.batch_size, settings.time_steps);

  NoiseGenerator gen;
  gen.initialize(settings, false, "fi", &handler);
  gen.reset(settings, false);
  gen.setNoisedControls(state, control_sequence);
  gen.shutdown();

  // Check all values are finite (no NaN or Inf from iFFT)
  double sum_vx = 0.0;
  int n = settings.batch_size * settings.time_steps;
  for (unsigned int r = 0; r < settings.batch_size; ++r) {
    for (unsigned int c = 0; c < settings.time_steps; ++c) {
      float vx = state.cvx(r, c);
      EXPECT_TRUE(std::isfinite(vx))
        << "Non-finite value at row=" << r << " col=" << c << ": " << vx;
      sum_vx += static_cast<double>(vx);
    }
  }

  // Mean should be near zero (unbiased noise)
  double mean_vx = sum_vx / n;
  EXPECT_NEAR(mean_vx, 0.0, 0.05)  // Within 0.05 of zero
    << "Mean of colored noise should be near zero, got " << mean_vx;
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  rclcpp::init(0, nullptr);

  int result = RUN_ALL_TESTS();

  rclcpp::shutdown();

  return result;
}
