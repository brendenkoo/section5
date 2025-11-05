#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class HeadingController(BaseHeadingController):
    """Class HeadingController that inherits from BaseHeadingController"""
    def __init__(self) -> None:
        super().__init__()
        self.kp = 2.0

    def compute_control_with_goal(
        self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:

        u = TurtleBotControl()
        # calculate the heading error (∈ [−π, π]) as the wrapped difference between the
        # goal’s theta and the state’s theta
        heading_error = wrap_angle(goal.theta - state.theta)

        # proportional control formula, ω = kp · err 
        # set its omega attribute to the computed angular velocity,
        # and return it
        u.omega = self.kp * heading_error

        return u
        
if __name__ == "__main__":
    rclpy.init()
    node = HeadingController()
    rclpy.spin(node)
    rclpy.shutdown()