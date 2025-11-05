#!/usr/bin/env python3
import numpy as np
from numpy import linalg
import typing as T

import rclpy
from rclpy.node import Node

import matplotlib.pyplot as plt

from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.navigation import TrajectoryPlan
from asl_tb3_lib.grids import StochOccupancyGrid2D

from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

from scipy.interpolate import splrep, splev
from utils import plot_line_segments

V_PREV_THRES = 0.0001

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        # determine if x[0] and x[1] is in the bounds provided by the state space
        x = np.asarray(x)
        if not (self.statespace_lo[0] <= x[0] <= self.statespace_hi[0] and 
                self.statespace_lo[1] <= x[1] <= self.statespace_hi[1]):
            return False
        # as long as x is in bounds, check occupancy 
        if not self.occupancy.is_free(np.array(x)):
            return False
        return True

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        # compute and return the Euclidean distance between x1 and x2
        # return (np.sqrt(np.sum((np.array(x1) - np.array(x2))**2)))

        # alternative: Instead of the Euclidean (L2) norm,
        # implement the L1 and L∞ norms

        # L1 Norm
        # return np.sum(np.abs(np.array(x1) - np.array(x2)))

        # L∞ Norm
        # return np.max(np.abs(np.array(x1) - np.array(x2)))

        # Original Euclidean (L2) Norm
        return (np.sqrt(np.sum((np.array(x1) - np.array(x2))**2)))

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        # calculate all of the possible moves (up, down, left, right, diagonals)
        possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        # see if theself is on the grid
        grid_self = self.snap_to_grid(x)
        
        for possible_move in possible_moves:
            # check each possible move to see if it is a neighbor
            neighbor = (grid_self[0] + possible_move[0] * self.resolution,
                        grid_self[1] + possible_move[1] * self.resolution)
            neighbor = self.snap_to_grid(neighbor)
            # if neighbor is free, add to neighbors list
            if self.is_free(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        while self.open_set:
            # use find best est cost helper function
            current = self.find_best_est_cost_through()

            # if currently at the goal, make a path from origin to here, return
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            # if not at goal, remove the current from the open 
            # set and add to closed set
            self.open_set.remove(current)
            self.closed_set.add(current)

            # for each of the neighbors, find the cost to arrive
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue
                
                tentative_cost_to_arrive = self.cost_to_arrive[current] + self.distance(current, neighbor)

                # if neighbor not in open set, add it
                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                # if neighbor cost to arrive is not better, skip
                elif tentative_cost_to_arrive >= self.cost_to_arrive.get(neighbor, float('inf')):
                    continue

                # this path is the best until now
                self.came_from[neighbor] = current
                self.cost_to_arrive[neighbor] = tentative_cost_to_arrive
                self.est_cost_through[neighbor] = tentative_cost_to_arrive + self.distance(neighbor, self.x_goal)
        # if you exit the while loop, no path was found
        return False

class Navigator(BaseNavigator):
    """Class Navigator that inherits from BaseNavigator"""
    def __init__(self):
        super().__init__()
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                  # returned by the differential flatness code
        self.kp = 2.0

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        u = TurtleBotControl()
        # calculate the heading error (∈ [−π, π]) as the wrapped difference between the
        # goal’s theta and the state’s theta
        heading_error = wrap_angle(goal.theta - state.theta)

        # proportional control formula, ω = kp · err 
        # set its omega attribute to the computed angular velocity,
        # and return it
        u.omega = float(self.kp * heading_error)
        u.v = 0.0

        return u

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t) -> TurtleBotControl:
        """ 
        Migrate and restructure the code from compute_control
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        ctrl_msg = TurtleBotControl()

        x = state.x
        y = state.y
        th = state.theta

        dt = t - self.t_prev
        # x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)
        x_d   = float(splev(t, plan.path_x_spline, der=0))
        xd_d  = float(splev(t, plan.path_x_spline, der=1))
        xdd_d = float(splev(t, plan.path_x_spline, der=2))

        y_d   = float(splev(t, plan.path_y_spline, der=0))
        yd_d  = float(splev(t, plan.path_y_spline, der=1))
        ydd_d = float(splev(t, plan.path_y_spline, der=2))

        # avoid singularity
        if abs(self.V_prev) < V_PREV_THRES:
            self.V_prev = V_PREV_THRES

        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d - x) + self.kdx*(xd_d - xd),
                      ydd_d + self.kpy*(y_d - y) + self.kdy*(yd_d - yd)], dtype=float)

        # compute real controls (Jacobian)
        J = np.array([[np.cos(th), -self.V_prev * np.sin(th)],
                      [np.sin(th),  self.V_prev * np.cos(th)]], dtype=float)
        

        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = float(V)
        self.om_prev = float(om)

        ctrl_msg.v = float(V)
        ctrl_msg.omega = float(om)

        # safety: log if non-finite
        if not np.isfinite(ctrl_msg.v) or not np.isfinite(ctrl_msg.omega):
            self.get_logger().error(f"Non-finite control produced: v={ctrl_msg.v}, omega={ctrl_msg.omega}, u={u}, J={J}")

        return ctrl_msg
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy:StochOccupancyGrid2D, resolution: float, horizon:float) -> T.Optional[TrajectoryPlan]:
        self.get_logger().info(f"Computing trajectory plan from {state.x, state.y} to {goal.x, goal.y}")
        astar = AStar(
            (-horizon + state.x, -horizon + state.y),
            (horizon + state.x, horizon + state.y),
            (state.x, state.y),
            (goal.x, goal.y),
            occupancy,
            resolution=resolution,
        )

        success = astar.solve()
        # raise error if path is not sufficient
        if not success or astar.path is None or (len(astar.path) < 4):
            self.get_logger().warn("A* failed to find a path.")
            return None
        
        # reset the controller state
        self.reset()

        # Compute planned time stamps using constant velocity heuristics
        astar.path = [np.asarray(p, dtype=float) for p in astar.path]
        path_np = np.vstack(astar.path)

        distances = np.sqrt(np.sum(np.diff(path_np, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        if total_distance == 0:
            times = np.array([0.0 for _ in range(path_np.shape[0])])
        else:
            avg_speed = total_distance / horizon
            times = np.insert(np.cumsum(distances / avg_speed), 0, 0.0)
            if times[-1] > 0:
                times *= horizon / times[-1]
        
        # Fit splines to path
        splinex = splrep(times, path_np[:,0], k=3)
        spliney = splrep(times, path_np[:,1], k= 3)
    
        # path = [np.array(p) for p in path]

        return TrajectoryPlan(
            path = astar.path,
            path_x_spline = splinex,
            path_y_spline = spliney,
            duration= times[-1]
        )

if __name__ == "__main__":
    rclpy.init()
    navigator = Navigator()
    rclpy.spin(navigator)
    rclpy.shutdown()

