import cv2
import pybullet
import pybullet_data
import rclpy

from rclpy.node import Node
from unitree_go.msg import LowState, LowCmd
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
from example_robot_data import getModelPath
import os

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

CAMERA_HEIGHT_PX = 60
CAMERA_WIDTH_PX = 106
CAMERA_FOV = 87.0
NEAR_CLIP, FAR_CLIP = 0.01, 2.0

assert NEAR_CLIP > 0.0 and FAR_CLIP > NEAR_CLIP, "Invalid clip plane settings."

class Go2Simulator(Node):
    def __init__(self):
        super().__init__('go2_simulation')
        
        self.nominal_pose = np.array([0.05, 0.6, -1.2, -0.05, 0.6, -1.2, 0.05, 0.6, -1.2, -0.05, 0.6, -1.2])
        self.q0 = self.nominal_pose.tolist()

        ########################### State
        self.last_cmd_msg = LowCmd()
        self.lowstate_publisher = self.create_publisher(LowState, "/lowstate", 10)

        self.last_image_msg = Image() 
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, "/camera", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer to publish periodically
        self.camera_period = 1./10  # seconds
        self.high_level_period = 1./50  # seconds
        self.low_level_sub_step = 12

        ########################## Cmd
        self.create_subscription(LowCmd, "/lowcmd", self.receive_cmd_cb, 10)

        robot_subpath = "go2_description/urdf/go2.urdf"
        self.robot_path = os.path.join(getModelPath(robot_subpath), robot_subpath)
        self.robot_path = "/home/hamlet/Workspace/reinforcement-learning/SoloParkour/go2/go2.urdf"
        self.robot = 0
        self.init_pybullet()

        self.last_msg_received = 0
        self.counter = 0
        self.i = 0

        self.timer = self.create_timer(self.high_level_period, self.update)

        self.last_lin_vel = np.zeros(3, dtype=np.float32)
        self.last_lin_acc = np.zeros(3, dtype=np.float32)

        self.K_TORQUE_LIMIT = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43])
        
        self.low_msg = LowState()


    def init_pybullet(self):
        pybullet.connect(pybullet.GUI)

        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # Load robot
        self.get_logger().info(f"go2_simulator::loading urdf : {self.robot_path}")
        self.robot = pybullet.loadURDF(self.robot_path, [0, 0, 0.4])
        self.get_logger().info(f"go2_simulator::loading urdf : {self.robot}")

        # Print joint names
        num_joints = pybullet.getNumJoints(self.robot)
        feet_names = [name + '_foot' for name in ("FR", "FL", "RR", "RL")]
        feet_names = [name + '_foot' for name in ("FR", "FL", "HR", "HL")]
        self.feet_idx = [-1] * len(feet_names)

        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            self.get_logger().info(f"go2_simulator::joint_info : {joint_name}")

            if link_name in feet_names:
                foot_id = feet_names.index(link_name)
                self.feet_idx[foot_id] = (i, link_name)

        self.get_logger().info(f"go2_simulator::feet_idx : {self.feet_idx}")

        # Load ground plane
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = pybullet.loadURDF("plane.urdf")
        self.collision_ids = [self.plane_id]
        self.load_obstacles()

        pybullet.resetBasePositionAndOrientation(self.plane_id, [0, 0, 0], [0, 0, 0, 1])

        pybullet.setTimeStep(self.high_level_period / self.low_level_sub_step)

        UNITREE_ORDER = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]

        self.joint_order = UNITREE_ORDER
        self.joint_order = [k.replace('RL_', 'HL_').replace('RR_', 'HR_').replace('_hip_joint', '_HAA').replace('_thigh_joint', '_HFE').replace('_calf_joint', '_KFE') for k in self.joint_order]

        self.j_idx = []
        for j in self.joint_order:
            self.j_idx.append(self.get_joint_id(j))


        for i, id in enumerate(self.j_idx):
            pybullet.resetJointState(self.robot, id, self.q0[i], 0.0)
    
        # gravity and feet friction
        pybullet.setGravity(0, 0, -9.81)

        # Somehow this disable joint friction
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.j_idx,
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocities=[0. for i in range(12)],
            forces=[0. for i in range(12)],
        )

        self.robot_T = np.zeros((4, 4))
        self.joint_q = np.zeros(12)

        self.camera_eye_b = np.array([0.35, 0.0, 0.1, 1.0])
        self.camera_target_b = np.array([100.0, 0.0, 0.1, 1.0])

        self.q_des = np.zeros(12) 
        self.v_des = np.zeros(12) 
        self.tau_des = np.zeros(12)
        self.kp_des = np.zeros(12)
        self.kd_des = np.zeros(12)

        self.q = np.zeros(12)
        self.v = np.zeros(12)


    def get_camera_image(self, robot_T):
        up_vec = (robot_T[:3, :3] @ np.array([0.0, 0.0, 1.0])).tolist()
        camera_eye_w = robot_T @ self.camera_eye_b
        camera_target_w = robot_T @ self.camera_target_b

        view_matrix = pybullet.computeViewMatrix(
            camera_eye_w.tolist()[:3], camera_target_w.tolist()[:3], up_vec
        )

        projection_matrix = pybullet.computeProjectionMatrixFOV(
            CAMERA_FOV, CAMERA_WIDTH_PX / CAMERA_HEIGHT_PX, NEAR_CLIP, FAR_CLIP)
        depth = pybullet.getCameraImage(
            CAMERA_WIDTH_PX,
            CAMERA_HEIGHT_PX,
            view_matrix,
            projection_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_NO_SEGMENTATION_MASK,
        )[3] #[:-2, 4:-4] # (58, 98)

        # Convert depth buffer to liNEAR_CLIP depth
        depth = FAR_CLIP * NEAR_CLIP / (FAR_CLIP - (FAR_CLIP - NEAR_CLIP) * depth)
        depth = np.clip(depth, NEAR_CLIP, FAR_CLIP)
        depth = (depth - NEAR_CLIP) / (FAR_CLIP - NEAR_CLIP)
        depth -= 0.5  # center around zero

        # Resize the image with bicubic interpolation to (58, 87)
        depth = cv2.resize(
            depth, (87, 58), interpolation=cv2.INTER_CUBIC
        )

        return np.clip(depth, -0.5, 0.5)

    def load_obstacles(self):
        box_half_length = 1.0
        box_half_width = 2.5
        box_half_height = 0.3
        half_extents = [box_half_length, box_half_width, box_half_height]

        col_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half_extents)
        vis_id = pybullet.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1]
        )

        x_offset = 2.0
        z_offset = 0.0

        num_boxes = 8
        for i in range(num_boxes):
            box_id = pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x_offset, 0, z_offset],
            )

            pybullet.changeDynamics(bodyUniqueId=box_id, linkIndex=-1, lateralFriction=1.0)
            self.collision_ids.append(i)

            x_offset += 2 * box_half_length

            if i + 1 < num_boxes // 2:
                z_offset += box_half_height
            elif i + 1 > num_boxes // 2:
                z_offset -= box_half_height

    def update(self):
        print(f"{self.counter=} Stepping simulation")
        self.step_simulation()

        print("Updating state")
        self.update_state()

        if self.counter % int(self.camera_period / self.high_level_period) == 0:
            print("Updating camera")
            self.update_camera()

        self.counter += 1

    def update_camera(self): 
       # Get the depth image
        depth = self.get_camera_image(self.robot_T)
        depth = ((depth + 0.5) * 255).astype(np.uint8)

        timestamp = self.get_clock().now().to_msg()
        ros_image_msg = self.bridge.cv2_to_imgmsg(depth, encoding='mono8')
        ros_image_msg.header.stamp = timestamp
        self.image_publisher.publish(ros_image_msg)

    def update_state(self):
        # Read sensors
        joint_states = pybullet.getJointStates(self.robot, self.j_idx)

        for joint_idx, joint_state in enumerate(joint_states):
            self.low_msg.motor_state[joint_idx].mode = 1
            self.low_msg.motor_state[joint_idx].dq = (joint_state[0] - self.low_msg.motor_state[joint_idx].q) / self.high_level_period
            self.low_msg.motor_state[joint_idx].q = joint_state[0]

        # Read IMU
        position, orientation = pybullet.getBasePositionAndOrientation(self.robot) # world frame
        linear_vel, angular_vel = pybullet.getBaseVelocity(self.robot) # world frame

        # Convert to body frame
        R = np.array(pybullet.getMatrixFromQuaternion(orientation), dtype=np.float32).reshape(3, 3)
        self.robot_T[:3, :3] = R[:]
        self.robot_T[:3, 3] = position[:]
        self.robot_T[3, 3] = 1
 
        linear_vel = np.dot(R.T, linear_vel).astype(np.float32)
        angular_vel = np.dot(R.T, angular_vel).astype(np.float32)

        self.low_msg.imu_state.accelerometer[:] = linear_vel

        # Bullet uses [x,y,z,w] quaternions while /lowstate expects [w,x,y,z]
        self.low_msg.tick = self.i
        self.low_msg.imu_state.quaternion[0] = orientation[-1]
        self.low_msg.imu_state.quaternion[1:] = orientation[0:3]
        self.low_msg.imu_state.gyroscope = angular_vel

        # Update feet contact states
        for i, (foot_link_id, link_name) in enumerate(self.feet_idx):
            # Check ground plane contacts
            contact_points = [] 
            for collision_id in self.collision_ids:
                contact_points += pybullet.getContactPoints(
                    bodyA=self.robot,
                    bodyB=collision_id,
                    linkIndexA=foot_link_id
                )

            if len(contact_points) > 0:
                self.low_msg.foot_force[i] = 100

        # Robot state
        self.lowstate_publisher.publish(self.low_msg)

    def step_simulation(self):
        for _ in range(self.low_level_sub_step):
            # Get sub step state
            joint_states = pybullet.getJointStates(self.robot, self.j_idx)
            self.q[:] = [joint_state[0] for joint_state in joint_states]
            self.v[:] = [joint_state[1] for joint_state in joint_states]

            tau_cmd = self.tau_des - np.multiply(self.q-self.q_des, self.kp_des) - np.multiply(self.v-self.v_des, self.kd_des)

            # Clip torque command
            tau_cmd = np.clip(tau_cmd, -self.K_TORQUE_LIMIT, self.K_TORQUE_LIMIT) # Nm, torque limit
            # Set actuation
            pybullet.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.j_idx,
                controlMode=pybullet.TORQUE_CONTROL,
                forces=tau_cmd
            )

            # Advance simulation by one step
            pybullet.stepSimulation()

        self.i += 1

    def receive_cmd_cb(self, msg):
        #self.last_cmd_msg = msg
        self.q_des[:] = [msg.motor_cmd[i].q   for i in range(12)]
        # self.v_des[:] = [msg.motor_cmd[i].dq  for i in range(12)] * 0 # No velocity control
        # self.tau_des[:] = [msg.motor_cmd[i].tau for i in range(12)] * 0 # No torque control
        self.kp_des[:] = [msg.motor_cmd[i].kp  for i in range(12)]
        self.kd_des[:] = [msg.motor_cmd[i].kd  for i in range(12)]

        self.last_msg_received = self.counter

        # self.update()

    def get_joint_id(self, joint_name):
        num_joints = pybullet.getNumJoints(self.robot)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            if joint_info[1].decode("utf-8") == joint_name:
                return i
        return None  # Joint name not found

def main(args=None):
    rclpy.init(args=args)
    go2_simulation = None

    try:
        go2_simulation = Go2Simulator()
        rclpy.spin(go2_simulation)
    except rclpy.exceptions.ROSInterruptException:
        pass

    if go2_simulation:
        go2_simulation.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

