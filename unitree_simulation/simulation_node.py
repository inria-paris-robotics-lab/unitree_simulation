import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from unitree_simulation.abstract_wrapper import AbstractSimulatorWrapper
from unitree_simulation.robots_configuration import RobotConfigurationAbstract


class UnitreeSimulation(Node):
    def __init__(self):
        super().__init__("unitree_simulation")
        simulator_name = self.declare_parameter("simulator", rclpy.Parameter.Type.STRING).value
        robot_name = self.declare_parameter("robot", rclpy.Parameter.Type.STRING).value
        self.unlock_base_default = self.declare_parameter("unlock_base", rclpy.Parameter.Type.BOOL).value

        ########################## Robot configuration
        if robot_name is None:
            self.get_logger().error("No robot type provided, please set parameter to 'robot'.")
            exit()

        self.robot: RobotConfigurationAbstract = None
        if robot_name.lower() == "g1":
            from unitree_simulation.robots_configuration import G1Configuration

            self.robot = G1Configuration()
        elif robot_name.lower() == "go2":
            from unitree_simulation.robots_configuration import Go2Configuration

            self.robot = Go2Configuration()
        else:
            self.get_logger().error("Robot name not recognized, please set parameter to 'g1' or 'go2'.")
            exit()

        ########################## Simulator
        self.simulator: AbstractSimulatorWrapper = None
        if simulator_name == "simple":
            from unitree_simulation.simple_wrapper import SimpleWrapper

            self.simulator = SimpleWrapper(self.robot)
        elif simulator_name == "pybullet":
            from unitree_simulation.bullet_wrapper import BulletWrapper

            self.simulator = BulletWrapper(self.robot)
        else:
            self.get_logger().error("Simulation tool not recognized, please set parameter to 'simple' or 'pybullet'.")
            exit()

        ########################## Initial state
        self.q_current = np.zeros(7 + self.robot.n_dof)
        self.v_current = np.zeros(6 + self.robot.n_dof)
        self.a_current = np.zeros(6 + self.robot.n_dof)
        self.f_current = np.zeros(len(self.robot.feet_sensors_names))

        ########################### State publisher
        self.lowstate_publisher = self.create_publisher(self.robot.lowstate_msgs_type, "/lowstate", 10)
        self.odometry_publisher = self.create_publisher(Odometry, "/odometry/filtered", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        ########################## Cmd listener
        self.create_subscription(self.robot.lowcmd_msgs_type, "/lowcmd", self.receive_cmd_cb, 10)
        self.last_cmd_msg = self.robot.lowcmd_msgs_type()

        ########################## Unlock base
        if self.unlock_base_default is None:
            self.get_logger().error("Parameter 'unlock_base' not set!")
            exit()

        if self.unlock_base_default:
            self.unlock_base()
        else:
            self.create_subscription(Empty, "/unlock_base", lambda msg: self.unlock_base(), 1)

        self.create_subscription(Empty, "/reset", lambda msg: self.reset(), 1)

        ########################## Update loop
        self.timer = self.create_timer(self.robot.high_level_period, self.update)

    def update(self):
        ## Control robot
        q_des = np.array([self.last_cmd_msg.motor_cmd[i].q for i in range(self.robot.n_dof)])
        v_des = np.array([self.last_cmd_msg.motor_cmd[i].dq for i in range(self.robot.n_dof)])
        tau_des = np.array([self.last_cmd_msg.motor_cmd[i].tau for i in range(self.robot.n_dof)])
        kp_des = np.array([self.last_cmd_msg.motor_cmd[i].kp for i in range(self.robot.n_dof)])
        kd_des = np.array([self.last_cmd_msg.motor_cmd[i].kd for i in range(self.robot.n_dof)])

        for _ in range(self.robot.low_level_sub_step):
            # Iterate to simulate motor internal controller
            tau_cmd = (
                tau_des
                - np.multiply(self.q_current[7:] - q_des, kp_des)
                - np.multiply(self.v_current[6:] - v_des, kd_des)
            )
            # Simulator outputs base velocity and acceleration in local frame
            self.q_current, self.v_current, self.a_current, self.f_current = self.simulator.step(tau_cmd)

        ## Send proprioceptive measures (LowState)
        low_msg = self.robot.lowstate_msgs_type()
        odometry_msg = Odometry()
        transform_msg = TransformStamped()

        timestamp = self.get_clock().now().to_msg()

        # Format motor readings
        for joint_idx in range(self.robot.n_dof):
            low_msg.motor_state[joint_idx].mode = 1
            low_msg.motor_state[joint_idx].q = self.q_current[7 + joint_idx]
            low_msg.motor_state[joint_idx].dq = self.v_current[6 + joint_idx]

        # Contact sensors reading
        if len(self.robot.feet_sensors_names) > 0:
            low_msg.foot_force = [int(self.robot.foot_force_to_val(force)) for force in self.f_current]

        # Format IMU
        quat_xyzw = self.q_current[3:7].tolist()
        l_angular_vel = self.v_current[3:6]  # In local frame
        l_linear_acc = self.a_current[0:3]  # In local frame

        # Rearrange quaternion
        quat_wxyz = quat_xyzw[-1:] + quat_xyzw[:-1]
        low_msg.imu_state.quaternion = quat_wxyz

        # Convert gravity from world to local frame
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        gravity = rot_mat @ np.array(
            [0, 0, 9.81]
        )  # This seems wrong. Proper computation should be done between base and imu frame

        imu_acc = l_linear_acc + gravity

        low_msg.imu_state.gyroscope = l_angular_vel.astype(np.float32)
        low_msg.imu_state.accelerometer = imu_acc.astype(np.float32)

        # Publish message
        self.lowstate_publisher.publish(low_msg)

        ## Send robot pose
        # Odometry / state estimation
        odometry_msg.header.stamp = timestamp
        odometry_msg.header.frame_id = "odom"
        odometry_msg.child_frame_id = "base"
        odometry_msg.pose.pose.position.x = self.q_current[0]
        odometry_msg.pose.pose.position.y = self.q_current[1]
        odometry_msg.pose.pose.position.z = self.q_current[2]
        odometry_msg.pose.pose.orientation.x = self.q_current[3]
        odometry_msg.pose.pose.orientation.y = self.q_current[4]
        odometry_msg.pose.pose.orientation.z = self.q_current[5]
        odometry_msg.pose.pose.orientation.w = self.q_current[6]
        odometry_msg.twist.twist.linear.x = self.v_current[0]
        odometry_msg.twist.twist.linear.y = self.v_current[1]
        odometry_msg.twist.twist.linear.z = self.v_current[2]
        odometry_msg.twist.twist.angular.x = self.v_current[3]
        odometry_msg.twist.twist.angular.y = self.v_current[4]
        odometry_msg.twist.twist.angular.z = self.v_current[5]
        self.odometry_publisher.publish(odometry_msg)

        # Forwar odometry on tf
        transform_msg.header.stamp = timestamp
        transform_msg.header.frame_id = "odom"
        transform_msg.child_frame_id = "base"
        transform_msg.transform.translation.x = self.q_current[0]
        transform_msg.transform.translation.y = self.q_current[1]
        transform_msg.transform.translation.z = self.q_current[2]
        transform_msg.transform.rotation.x = self.q_current[3]
        transform_msg.transform.rotation.y = self.q_current[4]
        transform_msg.transform.rotation.z = self.q_current[5]
        transform_msg.transform.rotation.w = self.q_current[6]
        self.tf_broadcaster.sendTransform(transform_msg)

        # Check that the simulator is on time
        if self.timer.time_until_next_call() < 0:
            ratio = 1.0 - self.timer.time_until_next_call() * 1e-9 / self.robot.high_level_period
            self.get_logger().warn(
                "Simulator running slower than real time! Real time ratio : %.2f " % ratio, throttle_duration_sec=0.1
            )

    def receive_cmd_cb(self, msg):
        self.last_cmd_msg = msg

    def reset(self):
        self.simulator.reset()
        if self.unlock_base_default:
            self.simulator.unlock_base()

    def unlock_base(self):
        self.get_logger().info("Unlocking robot base")
        self.simulator.unlock_base()


def main(args=None):
    rclpy.init(args=args)
    try:
        unitree_simulation = UnitreeSimulation()
        rclpy.spin(unitree_simulation)
    except rclpy.exceptions.ROSInterruptException:
        pass

    unitree_simulation.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
