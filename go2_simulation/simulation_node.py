import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_go.msg import LowState, LowCmd
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from go2_simulation.abstract_wrapper import AbstractSimulatorWrapper
from cv_bridge import CvBridge

class Go2Simulation(Node):
    def __init__(self):
        super().__init__("go2_simulation")
        simulator_name = self.declare_parameter("simulator", rclpy.Parameter.Type.STRING).value
        simulator_name = "pybullet" if simulator_name is None else simulator_name

        ########################### State publisher
        self.lowstate_publisher = self.create_publisher(LowState, "/lowstate", 10)
        self.odometry_publisher = self.create_publisher(Odometry, "/odometry/filtered", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.depth_publisher = self.create_publisher(Image, "/camera/depth", 10)

        # Timer to publish periodically
        self.high_level_period = 1.0 / 500  # seconds
        self.low_level_sub_step = 4
        self.timer = self.create_timer(self.high_level_period, self.update)

        ########################## Camera
        self.camera_period = 1.0 / 10 # seconds
        self.camera_decimation = int(self.camera_period / self.high_level_period)
        #self.camera_timer = self.create_timer(self.camera_period, self.camera_update)
        ########################## Cmd listener
        self.create_subscription(LowCmd, "/lowcmd", self.receive_cmd_cb, 10)
        self.last_cmd_msg = LowCmd()

        ########################## Simulator
        self.get_logger().info("go2_simulator::loading simulator")
        timestep = self.high_level_period / self.low_level_sub_step

        self.simulator: AbstractSimulatorWrapper = None
        if simulator_name == "simple":
            from go2_simulation.simple_wrapper import SimpleWrapper

            self.simulator = SimpleWrapper(self, timestep)
        elif simulator_name == "pybullet":
            from go2_simulation.bullet_wrapper import BulletWrapper

            self.simulator = BulletWrapper(self, timestep)
            self.bridge = CvBridge()
        else:
            self.get_logger().error("Simulation tool not recognized")

        self.simulator_name = simulator_name
        self.get_logger().info(f"go2_simulator::simulator {simulator_name} loaded")

        ########################## Initial state
        self.q_current = np.zeros(7 + 12)
        self.v_current = np.zeros(6 + 12)
        self.a_current = np.zeros(6 + 12)
        self.f_current = np.zeros(4)

        self.i = 0

    def update(self):
        ## Control robot
        q_des = np.array([self.last_cmd_msg.motor_cmd[i].q for i in range(12)])
        v_des = np.array([self.last_cmd_msg.motor_cmd[i].dq for i in range(12)])
        tau_des = np.array([self.last_cmd_msg.motor_cmd[i].tau for i in range(12)])
        kp_des = np.array([self.last_cmd_msg.motor_cmd[i].kp for i in range(12)])
        kd_des = np.array([self.last_cmd_msg.motor_cmd[i].kd for i in range(12)])

        for _ in range(self.low_level_sub_step):
            # Iterate to simulate motor internal controller
            tau_cmd = (
                tau_des
                - np.multiply(self.q_current[7:] - q_des, kp_des)
                - np.multiply(self.v_current[6:] - v_des, kd_des)
            )
            # Simulator outputs base velocity and acceleration in local frame
            self.q_current, self.v_current, self.a_current, self.f_current = self.simulator.step(tau_cmd)

        ## Send proprioceptive measures (LowState)
        low_msg = LowState()
        odometry_msg = Odometry()
        transform_msg = TransformStamped()

        timestamp = self.get_clock().now().to_msg()

        # Format motor readings
        for joint_idx in range(12):
            low_msg.motor_state[joint_idx].mode = 1
            low_msg.motor_state[joint_idx].q = self.q_current[7 + joint_idx]
            low_msg.motor_state[joint_idx].dq = self.v_current[6 + joint_idx]

        # Contact sensors reading
        ## See https://github.com/inria-paris-robotics-lab/go2_simulation/issues/6
        low_msg.foot_force = (14.2 * np.ones(4) + 0.562 * self.f_current).astype(np.int32).tolist()

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

        # Camera update
        if self.i % self.camera_decimation == 0:
            self.camera_update()

        # Check that the simulator is on time
        if self.timer.time_until_next_call() < 0 and self.i % self.camera_decimation != 0:
            ratio = 1.0 - self.timer.time_until_next_call() * 1e-9 / self.high_level_period
            self.get_logger().warn(
                "Simulator running slower than real time! Real time ratio : %.2f " % ratio, throttle_duration_sec=0.1
            )
        self.i += 1

    def camera_update(self):
        if self.simulator_name == "pybullet":
            im = self.simulator.get_camera_image()
        else:
            self.get_logger().warn(f"Camera not implemented for this simulator: {self.simulator_name}")

        if im is not None:
            img_msg = self.bridge.cv2_to_imgmsg(im, encoding="16UC1")
            self.depth_publisher.publish(img_msg)

    def receive_cmd_cb(self, msg):
        self.last_cmd_msg = msg


def main(args=None):
    rclpy.init(args=args)
    try:
        go2_simulation = Go2Simulation()
        rclpy.spin(go2_simulation)
    except rclpy.exceptions.ROSInterruptException:
        pass

    go2_simulation.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
