#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from scipy.signal import convolve2d
import time


class FrontierExploration(Node):
    """
    ROS2 node for autonomous frontier-based exploration.
    
    This node subscribes to the map and robot state, computes frontier points
    (boundaries between explored and unexplored regions), and publishes navigation
    goals to systematically explore the environment.
    """
    
    def __init__(self):
        super().__init__("frontier_exploration")
        
        
        # Publishers
        self.nav_goal_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
        # Subscribers with appropriate QoS
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )
        self.state_sub = self.create_subscription(
            TurtleBotState, "/state", self.state_callback, 10
        )
        self.nav_success_sub = self.create_subscription(
            Bool, "/nav_success", self.nav_success_callback, 10
        )
        self.sub = self.create_subscription(Bool, "/detector_bool", self.image_callback, 10)

        
        # State variables
        self.current_state = None
        self.occupancy = None
        self.map_msg = None
        self.waiting_for_nav = False
        self.exploration_complete = False
        self.no_frontier_count = 0
        
        # Parameters
        self.frontier_threshold = 5  # Minimum number of frontier cells to consider
        self.min_frontier_distance = 0.3  # Minimum distance from robot to frontier (meters)
        self.max_no_frontier_attempts = 3  # Max attempts before declaring completion
        
        # Create a timer to periodically check and start exploration
        self.startup_timer = self.create_timer(2.0, self.startup_check_callback)
        self.startup_complete = False
        
        self.get_logger().info("Frontier Exploration Node initialized!")

    def image_callback(self, msg):
        if msg.data:
            self.get_logger().info("Image detected!")
            self.image_detected = True
            # if stop sign, detected, sleep for 5 seconds
            time.sleep(5)
            self.image_detected = False
            self.compute_and_send_next_goal()

        else:
            self.get_logger().info("No image detected.")
            self.image_detected = False

        
    def state_callback(self, msg: TurtleBotState):
        """Update current robot state."""
        self.current_state = msg
    
    def startup_check_callback(self):
        """Periodically check if we have received initial data and can start exploration."""
        if self.startup_complete:
            return
        
        if self.current_state is None:
            return
        
        if self.map_msg is None or self.occupancy is None:
            return
        
        # We have both state and map, start exploration!
        self.get_logger().info("Starting autonomous exploration!")
        self.startup_complete = True
        self.startup_timer.cancel()  # Stop the timer
        
        # Start exploration
        self.start_exploration()
        
    def map_callback(self, msg: OccupancyGrid):
        """
        Process incoming map messages and convert to StochOccupancyGrid2D.
        
        OccupancyGrid values: -1 = unknown, 0 = free, 100 = occupied
        """
        self.map_msg = msg
        
        # Extract map parameters
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        
        # Reshape map data (row-major order)
        # Note: occupancy values are -1 (unknown), 0-100 (occupied probability)
        map_data = np.array(msg.data, dtype=float).reshape((height, width))
        
        # Create StochOccupancyGrid2D
        # Keep values as: -1 for unknown, 0-1 for probability
        probs = map_data.copy()
        probs[probs >= 0] = probs[probs >= 0] / 100.0  # Convert 0-100 to 0-1
        
        self.occupancy = StochOccupancyGrid2D(
            resolution=resolution,
            size_xy=np.array([width, height]),
            origin_xy=np.array([origin_x, origin_y]),
            window_size=7,  # Window for is_free checks
            probs=probs,
            thresh=0.5
        )
        
    def nav_success_callback(self, msg: Bool):
        """
        Handle navigation success/failure messages.
        
        Args:
            msg: Bool message - True if navigation succeeded, False if it failed
        """
        if not self.waiting_for_nav:
            return
            
        self.waiting_for_nav = False
        
        if msg.data:
            self.get_logger().info("Navigation succeeded! Finding next frontier...")
        else:
            self.get_logger().warn("Navigation failed! Replanning...")
        
        # Whether success or failure, compute and send next goal
        self.compute_and_send_next_goal()
        
    def compute_and_send_next_goal(self):
        """
        Main exploration logic: find frontiers and send navigation goal.
        """
        if self.exploration_complete:
            self.get_logger().info("Exploration complete!")
            return
            
        if self.current_state is None or self.map_msg is None or self.occupancy is None:
            return
        
        # Find frontier points
        frontiers = self.find_frontiers()
        
        if len(frontiers) == 0:
            self.no_frontier_count += 1
            self.get_logger().warn(
                f"No frontiers found! (attempt {self.no_frontier_count}/{self.max_no_frontier_attempts})"
            )
            
            if self.no_frontier_count >= self.max_no_frontier_attempts:
                self.get_logger().info("Exploration complete! No more frontiers to explore.")
                self.exploration_complete = True
            return
        
        # Reset counter if we found frontiers
        self.no_frontier_count = 0
        
        # Select best frontier
        best_frontier = self.select_best_frontier(frontiers)
        
        if best_frontier is None:
            self.get_logger().warn("Could not select valid frontier!")
            return
        
        # Create and publish navigation goal
        goal_msg = TurtleBotState()
        goal_msg.x = float(best_frontier[0])
        goal_msg.y = float(best_frontier[1])
        goal_msg.theta = 0.0  # Heading doesn't matter for navigation
        
        self.get_logger().info(
            f"Sending navigation goal to frontier at ({goal_msg.x:.2f}, {goal_msg.y:.2f})"
        )
        
        self.nav_goal_pub.publish(goal_msg)
        self.waiting_for_nav = True
        
    def find_frontiers(self):
        """
        Find frontier cells in the map using convolution-based approach from Problem 2.
        
        A frontier cell is a free cell that:
        1. Has >= 20% unknown cells in surrounding window
        2. Has 0 occupied cells in surrounding window
        3. Has >= 30% free cells in surrounding window
        
        Returns:
            List of (x, y) coordinates of frontier cells in world frame
        """
        if self.occupancy is None:
            return []
        
        # Window size for neighborhood analysis
        window_size = 13
        
        # Get the probability grid from occupancy
        # probs: -1 = unknown, 0-1 = probability of occupancy
        probs = self.occupancy.probs
        
        # Create convolution kernel (all ones)
        kernel = np.ones((window_size, window_size))
        total_cells = window_size * window_size
        
        # Compute heuristics using convolution
        # 1. Count unknown cells (where probs < 0)
        unknown_mask = (probs < 0).astype(float)
        unknown_count = convolve2d(unknown_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        unknown_percentage = unknown_count / total_cells
        
        # 2. Count occupied cells (where probs > threshold, e.g., 0.5)
        occupied_mask = (probs > 0.5).astype(float)
        occupied_count = convolve2d(occupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        
        # 3. Count free cells (where 0 <= probs <= 0.5)
        free_mask = ((probs >= 0) & (probs <= 0.5)).astype(float)
        free_count = convolve2d(free_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        free_percentage = free_count / total_cells
        
        # Apply frontier criteria
        # The cell itself should be free (known and unoccupied)
        is_cell_free = (probs >= 0) & (probs <= 0.5)
        
        # Apply the three heuristics
        frontier_mask = (
            is_cell_free &
            (unknown_percentage >= 0.20) &  # At least 20% unknown neighbors
            (occupied_count == 0) &          # No occupied neighbors
            (free_percentage >= 0.30)        # At least 30% free neighbors
        )
        
        # Get grid coordinates of frontier cells
        frontier_indices = np.argwhere(frontier_mask)
        
        if len(frontier_indices) == 0:
            self.get_logger().info("No frontier cells found")
            return []
        
        # Convert grid indices to world coordinates
        # In the occupancy grid: rows correspond to y, columns to x
        frontiers_world = []
        for idx in frontier_indices:
            grid_xy = np.array([idx[1], idx[0]])  # [column, row] = [x, y]
            state_xy = self.occupancy.grid2state(grid_xy)
            
            # Filter out frontiers too close to robot
            if self.current_state is not None:
                dist = np.sqrt(
                    (state_xy[0] - self.current_state.x)**2 + 
                    (state_xy[1] - self.current_state.y)**2
                )
                if dist >= self.min_frontier_distance:
                    frontiers_world.append((float(state_xy[0]), float(state_xy[1])))
        
        self.get_logger().info(f"Found {len(frontiers_world)} frontier points")
        return frontiers_world
    
    def select_best_frontier(self, frontiers):
        """
        Select the best frontier to explore next.
        
        Strategy: Choose the closest frontier to the robot.
        You can modify this to use different heuristics (e.g., information gain).
        
        Args:
            frontiers: List of (x, y) frontier coordinates
            
        Returns:
            (x, y) coordinates of selected frontier, or None if none valid
        """
        if len(frontiers) == 0 or self.current_state is None:
            return None
        
        robot_pos = np.array([self.current_state.x, self.current_state.y])
        
        # Find closest frontier
        min_dist = float('inf')
        best_frontier = None
        
        for frontier in frontiers:
            frontier_pos = np.array(frontier)
            dist = np.linalg.norm(frontier_pos - robot_pos)
            
            # Check if this frontier is reachable (free space)
            if self.occupancy is not None:
                if self.occupancy.is_free(frontier_pos):
                    if dist < min_dist:
                        min_dist = dist
                        best_frontier = frontier
        
        return best_frontier
    
    def start_exploration(self):
        """
        Start the exploration process by finding and navigating to first frontier.
        Call this after a short delay to ensure map and state are initialized.
        """
        self.get_logger().info("Starting autonomous exploration!")
        self.compute_and_send_next_goal()



if __name__ == "__main__":
    rclpy.init()
    node = FrontierExploration()    
    rclpy.spin(node)
    rclpy.shutdown()
