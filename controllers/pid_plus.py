import os
import sys

try:
  from . import BaseController
except ImportError:
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from controllers import BaseController


class Controller(BaseController):
  """
  Minimal PID+ controller:
  - Keeps the original PID gains close to baseline.
  - Adds anti-windup on integral term.
  - Adds derivative smoothing.
  - Adds action rate limiting to reduce jerk.
  """

  def __init__(self):
    self.p = 0.21
    self.i = 0.085
    self.d = -0.060

    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_derivative = 0.0
    self.prev_action = 0.0

    self.integral_limit = 18.0
    self.derivative_alpha = 0.80
    self.max_action_delta_low_speed = 0.15
    self.max_action_delta_high_speed = 0.07

    self.roll_ff_gain = 0.055
    self.lookahead_weights = [1.0, 0.70, 0.45, 0.25]

  def _blended_target(self, target_lataccel, future_plan):
    if future_plan is None or len(future_plan.lataccel) == 0:
      return float(target_lataccel)

    values = [target_lataccel] + future_plan.lataccel[:len(self.lookahead_weights) - 1]
    weights = self.lookahead_weights[:len(values)]
    weighted_sum = 0.0
    weight_total = 0.0
    for value, weight in zip(values, weights):
      weighted_sum += weight * value
      weight_total += weight
    return weighted_sum / weight_total

  def _max_action_delta(self, v_ego):
    speed = max(0.0, float(v_ego))
    speed_norm = min(speed / 30.0, 1.0)
    return (
      self.max_action_delta_low_speed * (1.0 - speed_norm)
      + self.max_action_delta_high_speed * speed_norm
    )

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    blended_target = self._blended_target(target_lataccel, future_plan)
    error = blended_target - current_lataccel

    self.error_integral += error
    if self.error_integral > self.integral_limit:
      self.error_integral = self.integral_limit
    elif self.error_integral < -self.integral_limit:
      self.error_integral = -self.integral_limit

    derivative_raw = error - self.prev_error
    derivative = self.derivative_alpha * self.prev_derivative + (1.0 - self.derivative_alpha) * derivative_raw
    self.prev_error = error
    self.prev_derivative = derivative

    roll_ff = -self.roll_ff_gain * float(state.roll_lataccel)
    action_unclamped = self.p * error + self.i * self.error_integral + self.d * derivative + roll_ff

    delta = action_unclamped - self.prev_action
    max_action_delta = self._max_action_delta(state.v_ego)
    if delta > max_action_delta:
      action = self.prev_action + max_action_delta
    elif delta < -max_action_delta:
      action = self.prev_action - max_action_delta
    else:
      action = action_unclamped

    self.prev_action = action
    return action
