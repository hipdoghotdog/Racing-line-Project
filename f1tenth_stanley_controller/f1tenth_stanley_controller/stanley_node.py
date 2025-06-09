#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import time
import csv

class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        # Parameters
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering_angle', 0.34)
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value

        self.centerline_params = {
            'base_lookahead': 0.2,
            'lookahead_gain': 0.25,
            'k': 0.2
        }
        self.racingline_params = {
            'base_lookahead': 0.7,
            'lookahead_gain': 0.35,
            'k': 0.2
        }

        self.lap_count = 0
        self.max_laps = 20
        self.path_type = 'centerline'

        self.results_csv_path = os.path.expanduser(
            '/sim_ws/src/f1tenth_stanley_controller/f1tenth_stanley_controller/lap_results.csv'
        )

        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.init_pose_timer = self.create_timer(1.0, self.set_initial_pose)

        self.path = np.load('/sim_ws/src/f1tenth_stanley_controller/f1tenth_stanley_controller/center_line.npy')
        self.apply_parameters(self.centerline_params)

        self.started = False
        self.start_time = None
        self.finished = False
        self.log_pos = []

        self.current_pose = None
        self.current_speed = 0.0

        self.odom_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.timer = self.create_timer(0.05, self.control_loop)

    def apply_parameters(self, param_dict):
        self.base_lookahead = param_dict['base_lookahead']
        self.lookahead_gain = param_dict['lookahead_gain']
        self.k = param_dict['k']
        self.get_logger().info(f"Parameters updated: base_lookahead={self.base_lookahead}, "
                               f"lookahead_gain={self.lookahead_gain}, k={self.k}")

    def set_initial_pose(self):
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.pose.pose.position.x = 0.0
        initial_pose.pose.pose.position.y = 0.0
        initial_pose.pose.pose.position.z = 0.0

        direction = self.path[10] - self.path[1]
        yaw = np.arctan2(direction[1], direction[0])
        quat = R.from_euler('z', yaw).as_quat()
        initial_pose.pose.pose.orientation.x = quat[0]
        initial_pose.pose.pose.orientation.y = quat[1]
        initial_pose.pose.pose.orientation.z = quat[2]
        initial_pose.pose.pose.orientation.w = quat[3]

        initial_pose.pose.covariance[0] = 0.25
        initial_pose.pose.covariance[7] = 0.25
        initial_pose.pose.covariance[35] = 0.0685

        self.init_pose_pub.publish(initial_pose)
        self.get_logger().info("Published initial pose at (0, 0)")
        self.init_pose_timer.cancel()
        self.started = True

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        roll, pitch, yaw = r.as_euler('xyz')
        self.current_pose = np.array([pos.x, pos.y, yaw])
        self.current_speed = msg.twist.twist.linear.x

    def compute_dynamic_lookahead(self):
        return self.base_lookahead + self.lookahead_gain * self.current_speed

    def find_lookahead_point(self, pose, lookahead_distance):
        distances = np.linalg.norm(self.path - pose[:2], axis=1)
        closest_index = np.argmin(distances)
        for i in range(closest_index, len(self.path)):
            dist = np.linalg.norm(self.path[i] - pose[:2])
            if dist > lookahead_distance:
                return self.path[i], i
        return self.path[-1], len(self.path) - 1

    def control_loop(self):
        if self.current_pose is None or not self.started:
            return

        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info("Started timing lap")

        end_dist = np.linalg.norm(self.current_pose[:2] - [0, 0])
        if end_dist < 1 and time.time() - self.start_time > 10 and not self.finished:
            self.finished = True
            elapsed = time.time() - self.start_time
            total_distance = sum(
                np.linalg.norm(np.array(self.log_pos[i + 1]) - np.array(self.log_pos[i]))
                for i in range(len(self.log_pos) - 1)
            )
            header_needed = not os.path.exists(self.results_csv_path)
            with open(self.results_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(['Lap', 'Path Type', 'Time (s)', 'Distance (m)'])
                writer.writerow([self.lap_count + 1, self.path_type, round(elapsed, 2), round(total_distance, 2)])

            self.get_logger().info(f"Lap {self.lap_count + 1} complete: {self.path_type}, Time: {elapsed:.2f}s, Distance: {total_distance:.2f}m")
            self.switch_path_and_reset()
            return

        lookahead = self.compute_dynamic_lookahead()
        self.log_pos.append(self.current_pose[:2].copy())
        target_point, index = self.find_lookahead_point(self.current_pose, lookahead)

        heading_to_point = np.arctan2(target_point[1] - self.current_pose[1],
                                      target_point[0] - self.current_pose[0])
        heading_error = heading_to_point - self.current_pose[2]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        dx = target_point[0] - self.current_pose[0]
        dy = target_point[1] - self.current_pose[1]
        cte = np.sin(self.current_pose[2]) * dy - np.cos(self.current_pose[2]) * dx

        alpha = 1.0
        cte_term = np.arctan2(self.k * cte, self.current_speed + 1e-5)
        steer = alpha * heading_error + (1 - alpha) * cte_term
        steer = np.clip(steer, -self.max_steering_angle, self.max_steering_angle)

        yaw = self.current_pose[2]
        x_veh = np.cos(yaw) * dx + np.sin(yaw) * dy
        y_veh = -np.sin(yaw) * dx + np.cos(yaw) * dy

        if x_veh <= 0.1:
            x_veh = 0.1

        curvature = 2.0 * y_veh / (lookahead ** 2)
        curvature_gain = 4.0
        speed = max(0.1, min(5.0, 5.0 / (1 + curvature_gain * abs(curvature))))

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

        self.get_logger().info(f'Steer: {steer:.2f}, Speed: {speed:.2f}, CTE: {cte:.2f}, Lookahead: {lookahead:.2f}')

    def switch_path_and_reset(self):
        self.lap_count += 1

        if self.lap_count >= self.max_laps * 2:
            self.get_logger().info("All laps completed. Shutting down.")
            self.stop_robot()
            rclpy.shutdown()
            return

        if self.lap_count % self.max_laps == 0:
            if self.path_type == 'centerline':
                self.path_type = 'racing'
                self.path = np.load('/sim_ws/src/f1tenth_stanley_controller/f1tenth_stanley_controller/racing_line.npy')
                self.apply_parameters(self.racingline_params)
            else:
                self.path_type = 'centerline'
                self.path = np.load('/sim_ws/src/f1tenth_stanley_controller/f1tenth_stanley_controller/center_line.npy')
                self.apply_parameters(self.centerline_params)

        self.start_time = None
        self.finished = False
        self.log_pos = []
        self.set_initial_pose()

    def stop_robot(self):
        self.get_logger().info('Node shutting down. Stopping the robot.')
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_msg)
        time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt - stopping robot.')
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
