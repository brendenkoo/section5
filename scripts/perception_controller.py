#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from std_msgs.msg import Bool
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class PerceptionController(BaseHeadingController):
    """Class PerceptionController that inherits from BaseHeadingController"""
    def __init__(self) -> None:
        super().__init__("perception_controller")
        self.kp = 2.0
        # self._active = True
        self.declare_parameter("active", True)
        self.image_detected = False
        self.sub = self.create_subscription(Bool, "/detector_bool", self.image_callback, 10)

    def image_callback(self, msg):
        if msg.data:
            self.get_logger().info("Image detected!")
            self.image_detected = True
        else:
            self.get_logger().info("No image detected.")
            self.image_detected = False

    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
    
    def compute_control_with_goal(
        self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:

        u = TurtleBotControl()
        # calculate the heading error (∈ [−π, π]) as the wrapped difference between the
        # goal’s theta and the state’s theta
        heading_error = wrap_angle(goal.theta - state.theta)
        self.get_logger().info(f"image detected: {self.image_detected}")
        if self.image_detected:
            u.omega = 0.0
        else:
            u.omega = 0.2

        # proportional control formula, ω = kp · err 
        # set its omega attribute to the computed angular velocity,
        # and return it
        # u.omega = self.kp * heading_error
        # u.omega = 0.2
        return u
        
if __name__ == "__main__":
    rclpy.init()
    perception_controller = PerceptionController()
    rclpy.spin(perception_controller)
    rclpy.shutdown()