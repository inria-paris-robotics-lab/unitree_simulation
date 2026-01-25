import rclpy
import typing
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_go.msg import LowState, LowCmd
from nav_msgs.msg import Odometry
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from go2_simulation.abstract_wrapper import AbstractSimulatorWrapper
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from rclpy.time import Time
from rclpy.duration import Duration

import onnxruntime as rt
from collections import deque

def euler_from_quaternion(quat_angle):
    """
    NOTE: This was copied from extreme-parkour repo

    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    w, x, y, z = quat_angle
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

class Actor:
    def __init__(self):
        onnx_path = "./models/wall.onnx"
        onnx_path = "/home/hamlet/Workspace/reinforcement-learning/inference/" + onnx_path
        self.onnx_session = rt.InferenceSession(onnx_path)

        self.w_T_b = np.eye(4)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.joint_pos_policy = np.zeros(12)
        self.joint_vel_policy = np.zeros(12)

        self.q0 = np.array([-0.1,  0.8, -1.5, 0.1,  0.8, -1.5,  -0.1,  1., -1.5, 0.1,  1., -1.5])

        # First two elements are 0, third is the forward speed
        forward_speed = 0.37
        self.vel_cmd = np.array([0., 0., forward_speed])
        self.env_class = np.array([1, 0])

        self.action_buffer = deque(maxlen=2)
        self.depth_buffer = deque(maxlen=2)

        self.depth_latent = np.zeros((1, 32), dtype=np.float32)
        self.vobs = np.zeros((1, 58, 87), dtype=np.float32)
        self.yaws = np.zeros((1, 2), dtype=np.float32)
        self.obs = np.zeros((1, 53), dtype=np.float32)
        self.obs_history = np.zeros((1, 10, 53), dtype=np.float32)
        self.rnn_hidden_in = np.zeros((1, 1, 512), dtype=np.float32)
        self.update_depth = np.zeros((1,1), dtype=np.float32)
        self.update_yaw = np.ones((1,1), dtype=np.float32)
        self.step_counter = np.zeros((1,), dtype=np.float32)

        self.actions = np.zeros((1, 12), dtype=np.float32)
        self.policy_step = 0
        
        # LowCmd
        self.lowcmd = LowCmd()

    def forward(self, lowstate: LowState, im: typing.Optional[Image] = None):
        if im:
            im = np.array(im.data).reshape(im.height,im.width)
            self.vobs[:] = (im / 255.) - 0.5
            self.update_yaw[:] = 1.0
            self.update_depth[:] = 1.0
        else:
            self.update_yaw[:] = 0.0
            self.update_depth[:] = 0.0

        contact_states = lowstate.foot_force > 20

        quat = lowstate.imu_state.quaternion
        roll, pitch, yaw = euler_from_quaternion(quat)
        imu_obs = np.array([roll, pitch])

        q = np.array([ms.q for ms in lowstate.motor_state])[:12]
        q -= self.q0

        self.joint_vel[:] = (q - self.joint_pos) * 50. 
        self.joint_pos[:] = q

        obs_data = [
            1 * lowstate.imu_state.gyroscope * 0.25, # 3
            1 * imu_obs, # 2
            [0.0],
            1 * self.yaws.squeeze(),
            1 * self.vel_cmd, # 3
            1 * self.env_class, # 2
            1 * self.joint_pos, 
            1 * (self.joint_vel * 0.05),
            1 * (self.actions.squeeze()),
            1 * (contact_states - 0.5)
        ]

        clip = lambda a: np.clip(a, -100.0, 100.0)
        self.obs[:] = (
            np.concatenate(obs_data).reshape(1, 53).astype(np.float32)
        )
        self.obs[:] = clip(self.obs)
        self.step_counter[:] = self.policy_step
        self.policy_step += 1

        # Policy module
        inputs = {
            "depth": clip(self.vobs),
            "depth_latent_in": self.depth_latent,
            "yaw_in": clip(self.yaws),
            "obs_proprio": clip(self.obs),
            "obs_history_in": clip(self.obs_history),
            "update_depth": self.update_depth,
            "update_yaw": self.update_yaw,
            "hidden_states_in": self.rnn_hidden_in,
            "step_counter": self.step_counter
        }

        nn_actions, depth_latent, yaws, obs_history, _ = self.onnx_session.run(
            ['actions', 'depth_latent_out', 'yaw_out', 'obs_history_out', 'hidden_states_out'], inputs
        )
        self.actions[:] = nn_actions.astype(np.float32)
        self.depth_latent[:] = depth_latent
        self.yaws[:] = yaws
        self.obs_history[:] = obs_history
        # self.rnn_hidden_in[:] = hidden_states_out

        return self.q0 + (np.clip(self.actions.squeeze(), -4.8, 4.8) * .25)


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
        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)

        # Timer to publish periodically
        self.high_level_period = 1.0 / 50  # seconds
        self.low_level_sub_step = 24
        self.timer = self.create_timer(self.high_level_period, self.update)

        ########################## Camera
        self.camera_period = 1.0 / 10 # seconds
        self.camera_decimation = int(self.camera_period / self.high_level_period)

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
        self.sim_time = Time(seconds=0, nanoseconds=0)
        self.time_delta = Duration(seconds=0, nanoseconds=int(self.high_level_period * 1e9))

        self.actor = Actor()


    def update(self):
        ## Control robot
        if False:
            q_des = np.array([self.last_cmd_msg.motor_cmd[i].q for i in range(12)])
            v_des = np.array([self.last_cmd_msg.motor_cmd[i].dq for i in range(12)])
            tau_des = np.array([self.last_cmd_msg.motor_cmd[i].tau for i in range(12)])
            kp_des = np.array([self.last_cmd_msg.motor_cmd[i].kp for i in range(12)])
            kd_des = np.array([self.last_cmd_msg.motor_cmd[i].kd for i in range(12)])
        else:
            # Camera update
            if self.i == 0:
                q_des = np.zeros(12)
            elif self.i % self.camera_decimation == 0:
                im = self.camera_update() 
                q_des = self.actor.forward(self.low_msg, im=im)
            else:
                q_des = self.actor.forward(self.low_msg, im=None)

            v_des = np.zeros(12)
            tau_des = np.zeros(12)
            kp_des = 40 * np.ones(12)
            kd_des = 1 * np.ones(12)

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

        # Update simulation time
        self.sim_time += self.time_delta
        clock_msg = Clock()
        clock_msg.clock = self.sim_time.to_msg()
        self.clock_publisher.publish(clock_msg)

        timestamp = self.sim_time.to_msg() 

        # Format motor readings
        for joint_idx in range(12):
            low_msg.motor_state[joint_idx].mode = 1
            low_msg.motor_state[joint_idx].q = self.q_current[7 + joint_idx]
            low_msg.motor_state[joint_idx].dq = self.v_current[6 + joint_idx]

        # Contact sensors reading
        ## See https://github.com/inria-paris-robotics-lab/go2_simulation/issues/6
        low_msg.foot_force = (14.2 * np.ones(4) + 0.562 * self.f_current).astype(np.int32).tolist()

        # Format IMU
        # bullet quat
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
        self.low_msg = low_msg

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
        if self.timer.time_until_next_call() < 0 and self.i % self.camera_decimation != 0:
            ratio = 1.0 - self.timer.time_until_next_call() * 1e-9 / self.high_level_period
            self.get_logger().warn(
                "Simulator running slower than real time! Real time ratio : %.2f " % ratio, throttle_duration_sec=0.1
            )
        self.i += 1

    def camera_update(self):
        if self.simulator_name == "pybullet":
            im = self.simulator.get_camera_image()
            img_msg = self.bridge.cv2_to_imgmsg(im, encoding="mono8")
            self.depth_publisher.publish(img_msg)
            return img_msg
        else:
            self.get_logger().warn(f"Camera not implemented for this simulator: {self.simulator_name}")

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
