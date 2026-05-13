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

#include "nav2_mppi_controller/tools/noise_generator.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

namespace mppi
{

void NoiseGenerator::initialize(
  mppi::models::OptimizerSettings & settings, bool is_holonomic,
  const std::string & name, ParametersHandler * param_handler)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  active_ = true;

  ndistribution_vx_ = std::normal_distribution(0.0f, settings_.sampling_std.vx);
  ndistribution_vy_ = std::normal_distribution(0.0f, settings_.sampling_std.vy);
  ndistribution_wz_ = std::normal_distribution(0.0f, settings_.sampling_std.wz);

  auto getParam = param_handler->getParamGetter(name);
  getParam(regenerate_noises_, "regenerate_noises", false);
  getParam(use_low_pass_filter_, "use_low_pass_filter", false, ParameterType::Static);
  getParam(
    filter_cutoff_frequency_, "filter_cutoff_frequency", 2.0,
    ParameterType::Static);
  getParam(filter_order_, "filter_order", 2, ParameterType::Static);

  bool use_colored_noise_param = false;
  double cn_exp_vx = 2.0, cn_exp_vy = 2.0, cn_exp_wz = 2.0;
  getParam(use_colored_noise_param, "use_colored_noise", false, ParameterType::Static);
  getParam(cn_exp_vx, "colored_noise_exponent_vx", 2.0, ParameterType::Static);
  getParam(cn_exp_vy, "colored_noise_exponent_vy", 2.0, ParameterType::Static);
  getParam(cn_exp_wz, "colored_noise_exponent_wz", 2.0, ParameterType::Static);
  colored_noise_exponent_vx_ = static_cast<float>(cn_exp_vx);
  colored_noise_exponent_vy_ = static_cast<float>(cn_exp_vy);
  colored_noise_exponent_wz_ = static_cast<float>(cn_exp_wz);

  if (use_colored_noise_param && use_low_pass_filter_) {
    RCLCPP_WARN(
      logger_,
      "Both use_colored_noise and use_low_pass_filter are true. "
      "Using colored noise; LP filter will be disabled.");
    use_low_pass_filter_ = false;
  }
  use_colored_noise_ = use_colored_noise_param;
  generator_ = std::default_random_engine();

  configureLowPassFilter();

  if (regenerate_noises_) {
    noise_thread_ = std::thread(std::bind(&NoiseGenerator::noiseThread, this));
  } else {
    generateNoisedControls();
  }
}

void NoiseGenerator::shutdown()
{
  active_ = false;
  ready_ = true;
  noise_cond_.notify_all();
  if (noise_thread_.joinable()) {
    noise_thread_.join();
  }
}

void NoiseGenerator::generateNextNoises()
{
  // Trigger the thread to run in parallel to this iteration
  // to generate the next iteration's noises (if applicable).
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    ready_ = true;
  }
  noise_cond_.notify_all();
}

void NoiseGenerator::setNoisedControls(
  models::State & state,
  const models::ControlSequence & control_sequence)
{
  std::unique_lock<std::mutex> guard(noise_lock_);
  if (regenerate_noises_) {
    noise_cond_.wait(guard, [this]() {return !ready_;});
  }

  state.cvx = noises_vx_.rowwise() + control_sequence.vx.transpose();
  state.cvy = noises_vy_.rowwise() + control_sequence.vy.transpose();
  state.cwz = noises_wz_.rowwise() + control_sequence.wz.transpose();
}

void NoiseGenerator::reset(mppi::models::OptimizerSettings & settings, bool is_holonomic)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  ndistribution_vx_ = std::normal_distribution(0.0f, settings_.sampling_std.vx);
  ndistribution_vy_ = std::normal_distribution(0.0f, settings_.sampling_std.vy);
  ndistribution_wz_ = std::normal_distribution(0.0f, settings_.sampling_std.wz);
  configureLowPassFilter();

  // Recompute the noises on reset, initialization, and fallback
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noises_vx_.setZero(settings_.batch_size, settings_.time_steps);
    noises_vy_.setZero(settings_.batch_size, settings_.time_steps);
    noises_wz_.setZero(settings_.batch_size, settings_.time_steps);
    ready_ = false;
    generateNoisedControls();
  }
  noise_cond_.notify_all();
}

void NoiseGenerator::noiseThread()
{
  while (true) {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noise_cond_.wait(guard, [this]() {return ready_ || !active_;});
    if (!active_) {
      break;
    }
    ready_ = false;
    generateNoisedControls();
    guard.unlock();
    noise_cond_.notify_all();
  }
}

void NoiseGenerator::generateNoisedControls()
{
  auto & s = settings_;

  if (use_colored_noise_) {
    noises_vx_.resize(s.batch_size, s.time_steps);
    noises_wz_.resize(s.batch_size, s.time_steps);
    generateColoredNoise(noises_vx_, colored_noise_exponent_vx_, s.sampling_std.vx);
    generateColoredNoise(noises_wz_, colored_noise_exponent_wz_, s.sampling_std.wz);
    if (is_holonomic_) {
      noises_vy_.resize(s.batch_size, s.time_steps);
      generateColoredNoise(noises_vy_, colored_noise_exponent_vy_, s.sampling_std.vy);
    }
    return;
  }

  int low_pass_warmup_steps = 0;
  if (lpf_configured_ && s.model_dt > 0.0f && filter_cutoff_frequency_ > 0.0) {
    const double sample_rate = 1.0 / static_cast<double>(s.model_dt);
    const int cutoff_response_steps = static_cast<int>(
      std::ceil(4.0 * sample_rate / filter_cutoff_frequency_));
    const int order_response_steps = std::max(1, filter_order_) * 8;
    const int max_warmup_steps = std::max(1, static_cast<int>(s.time_steps) * 2);
    low_pass_warmup_steps = std::clamp(
      std::max(cutoff_response_steps, order_response_steps), 1, max_warmup_steps);
  }

  auto generate_white_noise =
    [&](std::normal_distribution<float> & distribution) -> Eigen::ArrayXXf
    {
      Eigen::ArrayXXf noise = Eigen::ArrayXXf::NullaryExpr(
        s.batch_size, s.time_steps + low_pass_warmup_steps,
        [&]() {return distribution(generator_);});

      if (lpf_configured_) {
        applyLowPassFilter(noise);
        return noise.rightCols(s.time_steps).eval();
      }

      return noise;
    };

  noises_vx_ = generate_white_noise(ndistribution_vx_);
  noises_wz_ = generate_white_noise(ndistribution_wz_);
  if (is_holonomic_) {
    noises_vy_ = generate_white_noise(ndistribution_vy_);
  }
}

void NoiseGenerator::configureLowPassFilter()
{
  lpf_configured_ = false;
  lpf_a_.clear();
  lpf_b_.clear();

  if (!use_low_pass_filter_) {
    return;
  }

  if (settings_.model_dt <= 0.0f) {
    RCLCPP_WARN(
      logger_,
      "Low-pass filter disabled: model_dt must be > 0, got %.6f",
      settings_.model_dt);
    return;
  }

  const double fs = 1.0 / static_cast<double>(settings_.model_dt);
  const double nyquist = 0.5 * fs;

  if (filter_cutoff_frequency_ <= 0.0) {
    RCLCPP_WARN(
      logger_,
      "Low-pass filter disabled: filter_cutoff_frequency must be > 0, got %.6f",
      filter_cutoff_frequency_);
    return;
  }

  if (filter_order_ < 1) {
    RCLCPP_WARN(
      logger_,
      "Low-pass filter disabled: filter_order must be >= 1, got %d",
      filter_order_);
    return;
  }

  if (filter_cutoff_frequency_ >= nyquist) {
    const double clamped = std::max(
      std::nextafter(nyquist, 0.0),
      nyquist * 0.99);
    RCLCPP_WARN(
      logger_,
      "filter_cutoff_frequency %.6f Hz is >= Nyquist %.6f Hz. Clamping to %.6f Hz.",
      filter_cutoff_frequency_, nyquist, clamped);
    filter_cutoff_frequency_ = clamped;
  }

  if (!designButterworthLowPass()) {
    RCLCPP_WARN(logger_, "Low-pass filter disabled: failed to design coefficients.");
    return;
  }

  lpf_configured_ = true;
}

bool NoiseGenerator::designButterworthLowPass()
{
  const int order_i = filter_order_;
  if (order_i < 1) {
    return false;
  }
  const size_t order = static_cast<size_t>(order_i);

  const double fs = 1.0 / static_cast<double>(settings_.model_dt);
  const double pi = std::acos(-1.0);
  const double omega_c = 2.0 * fs * std::tan(pi * filter_cutoff_frequency_ / fs);

  if (!std::isfinite(omega_c) || omega_c <= 0.0) {
    return false;
  }

  std::vector<std::complex<double>> z_poles;
  z_poles.reserve(order);

  for (int k = 0; k < order_i; ++k) {
    const double theta = pi * (2.0 * static_cast<double>(k) + 1.0 + order_i) /
      (2.0 * static_cast<double>(order_i));
    const std::complex<double> s_pole = omega_c * std::exp(std::complex<double>(0.0, theta));
    const std::complex<double> z_pole = (2.0 * fs + s_pole) / (2.0 * fs - s_pole);
    z_poles.push_back(z_pole);
  }

  // Denominator A(z^-1) = Π (1 - z_pole * z^-1)
  std::vector<std::complex<double>> a_poly(1, std::complex<double>(1.0, 0.0));
  for (const auto & pole : z_poles) {
    std::vector<std::complex<double>> next(a_poly.size() + 1, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < a_poly.size(); ++i) {
      next[i] += a_poly[i];
      next[i + 1] -= a_poly[i] * pole;
    }
    a_poly = std::move(next);
  }

  lpf_a_.resize(a_poly.size());
  for (size_t i = 0; i < a_poly.size(); ++i) {
    lpf_a_[i] = a_poly[i].real();
  }

  // Numerator B(z^-1) = K * (1 + z^-1)^order
  std::vector<double> b_poly(1, 1.0);
  b_poly.resize(order + 1, 0.0);
  for (int i = 1; i <= order_i; ++i) {
    for (int j = i; j > 0; --j) {
      b_poly[static_cast<size_t>(j)] += b_poly[static_cast<size_t>(j - 1)];
    }
  }

  const double sum_a = std::accumulate(lpf_a_.begin(), lpf_a_.end(), 0.0);
  const double sum_b = std::accumulate(b_poly.begin(), b_poly.end(), 0.0);
  if (std::abs(sum_b) < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double dc_gain = sum_a / sum_b;
  lpf_b_.resize(b_poly.size());
  for (size_t i = 0; i < b_poly.size(); ++i) {
    lpf_b_[i] = b_poly[i] * dc_gain;
  }

  if (std::abs(lpf_a_.front()) < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  // Normalize by a0 for direct-form implementation.
  const double a0 = lpf_a_.front();
  for (auto & a : lpf_a_) {
    a /= a0;
  }
  for (auto & b : lpf_b_) {
    b /= a0;
  }

  return true;
}

void NoiseGenerator::applyLowPassFilter(Eigen::ArrayXXf & signal) const
{
  if (!lpf_configured_) {
    return;
  }

  const size_t order = static_cast<size_t>(filter_order_);
  std::vector<double> x_hist(order + 1, 0.0);
  std::vector<double> y_hist(order + 1, 0.0);

  for (int row = 0; row < signal.rows(); ++row) {
    std::fill(x_hist.begin(), x_hist.end(), 0.0);
    std::fill(y_hist.begin(), y_hist.end(), 0.0);

    for (int col = 0; col < signal.cols(); ++col) {
      for (size_t idx = order; idx > 0; --idx) {
        x_hist[idx] = x_hist[idx - 1];
        y_hist[idx] = y_hist[idx - 1];
      }

      x_hist[0] = static_cast<double>(signal(row, col));

      double y = 0.0;
      for (size_t idx = 0; idx <= order; ++idx) {
        y += lpf_b_[idx] * x_hist[idx];
      }
      for (size_t idx = 1; idx <= order; ++idx) {
        y -= lpf_a_[idx] * y_hist[idx];
      }

      y_hist[0] = y;
      signal(row, col) = static_cast<float>(y);
    }
  }
}

// Implements Vlahov et al., "Low Frequency Sampling in MPPI Control", RA-L 2024.
// Algorithm: sample Gaussians in frequency domain with PSD proportional to 1/f^gamma,
// then iFFT to get time-correlated noise emphasising low frequencies.
void NoiseGenerator::generateColoredNoise(
  Eigen::ArrayXXf & noise, float exponent, float sigma)
{
  const int T = settings_.time_steps;
  const int batch = settings_.batch_size;
  if (T <= 0 || batch <= 0 || sigma <= 0.0f) {
    noise.setZero(batch, T);
    return;
  }

  const int sample_T = T * 2;

  // TODO(anovate): For T > 128, consider switching to an FFT library (FFTW / Eigen FFT)
  // for O(T log T) instead of the current O(T^2) direct summation.
  if (sample_T > 128) {
    RCLCPP_WARN_ONCE(
      logger_,
      "Colored noise iFFT uses direct O(T^2) summation. With doubled T=%d, "
      "consider adding an FFT library for better performance.", sample_T);
  }

  // Number of unique frequency components for Hermitian-symmetric spectrum
  const int N = sample_T / 2 + 1;
  const float fmin = 1.0f / static_cast<float>(sample_T);

  // Build frequency-domain PSD scaling: s_scale[k] = f[k]^(-exponent/2)
  // with fmin cutoff so DC and very low bins don't blow up
  Eigen::ArrayXf s_scale(N);
  for (int k = 0; k < N; ++k) {
    float f_k = static_cast<float>(k) / static_cast<float>(sample_T);
    f_k = std::max(f_k, fmin);
    s_scale(k) = std::pow(f_k, -exponent / 2.0f);
  }

  // Compute normalization sigma so that time-domain variance = 1
  // sigma = 2 * sqrt(sum(w^2)) / T, where w = s_scale[1:end-1]
  // with correction for the last frequency bin
  double sum_sq = 0.0;
  for (int k = 1; k < N - 1; ++k) {
    sum_sq += static_cast<double>(s_scale(k)) * static_cast<double>(s_scale(k));
  }
  // Match ACDSLab MPPI-Generic's variance correction for the final frequency bin.
  const float last_factor = (1.0f + static_cast<float>(sample_T % 2)) / 2.0f;
  double last_val = static_cast<double>(s_scale(N - 1)) *
    static_cast<double>(last_factor);
  sum_sq += last_val * last_val;
  double sigma_norm = 2.0 * std::sqrt(sum_sq) / static_cast<double>(sample_T);
  if (sigma_norm < 1e-12) {
    noise.setZero(batch, T);
    return;
  }

  // Precompute iFFT transform matrix M (T x (2N-2) or so)
  // z(t) = (1/T) * [Z_real[0] + sum_k 2*Z_real[k]*cos(2*pi*k*t/T)
  //                  - 2*Z_imag[k]*sin(2*pi*k*t/T) ...]
  // Build this as a matrix M of shape [T, 2*N] where columns are:
  //   [Z_real[0], Z_real[1]..Z_real[N-1], Z_imag[1]..Z_imag[N-2]]
  // But for simplicity and small T (~56), we process per-row using direct summation.

  const float inv_sample_T = 1.0f / static_cast<float>(sample_T);
  const float unit_scale_factor = 1.0f / static_cast<float>(sigma_norm);
  const float two_pi_over_sample_T =
    2.0f * static_cast<float>(M_PI) / static_cast<float>(sample_T);
  constexpr int offset_t = 1;
  constexpr float offset_decay_rate = 0.97f;

  std::normal_distribution<float> ndist(0.0f, 1.0f);

  for (int row = 0; row < batch; ++row) {
    // Sample frequency-domain Gaussian with PSD scaling
    // Z_real[k] ~ N(0, s_scale[k]), Z_imag[k] ~ N(0, s_scale[k])
    // Hermitian constraints: Z_imag[0] = 0
    //                        Z_imag[N-1] = 0 if T is even
    Eigen::ArrayXf z_real(N), z_imag(N);
    for (int k = 0; k < N; ++k) {
      z_real(k) = ndist(generator_) * s_scale(k);
      z_imag(k) = ndist(generator_) * s_scale(k);
    }
    // DC component must be real
    z_imag(0) = 0.0f;
    z_real(0) *= std::sqrt(2.0f);  // magnitude fix for DC

    // If T is even, Nyquist component must be real
    if (sample_T % 2 == 0) {
      z_imag(N - 1) = 0.0f;
      z_real(N - 1) *= std::sqrt(2.0f);  // magnitude fix
    }

    // iFFT via direct summation (Eq 4 from paper):
    // z(t) = (1/T) * [ z_real[0] + 2*sum_{k=1}^{N-2}(z_real[k]*cos - z_imag[k]*sin)
    //                   + z_real[N-1]*cos ]   (last term factor 1 if T even, 2 if odd)
    Eigen::ArrayXf doubled_horizon_noise(sample_T);
    for (int t = 0; t < sample_T; ++t) {
      float val = z_real(0);

      for (int k = 1; k < N - 1; ++k) {
        float angle = two_pi_over_sample_T * static_cast<float>(k) * static_cast<float>(t);
        val += 2.0f * (z_real(k) * std::cos(angle) - z_imag(k) * std::sin(angle));
      }

      // Last frequency bin
      if (N > 1) {
        float angle_last =
          two_pi_over_sample_T * static_cast<float>(N - 1) * static_cast<float>(t);
        if (sample_T % 2 == 0) {
          // Nyquist bin: real only, factor 1
          val += z_real(N - 1) * std::cos(angle_last);
        } else {
          // Odd T: last bin is a normal complex bin with factor 2
          val += 2.0f * (z_real(N - 1) * std::cos(angle_last) -
            z_imag(N - 1) * std::sin(angle_last));
        }
      }

      doubled_horizon_noise(t) = val * inv_sample_T * unit_scale_factor;
    }

    const int clamped_offset_t = std::min(offset_t, sample_T - 1);
    const float offset_value = doubled_horizon_noise(clamped_offset_t);
    float decay = 1.0f;
    for (int t = 0; t < T; ++t) {
      noise(row, t) = sigma * (doubled_horizon_noise(t) - offset_value * decay);
      decay *= offset_decay_rate;
    }
  }
}

}  // namespace mppi
