import os
import sys

try:
  from . import BaseController
except ImportError:
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from controllers import BaseController
import numpy as np

class Controller(BaseController):
  """
  A controller that always outputs zero
  """
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return 0.0
