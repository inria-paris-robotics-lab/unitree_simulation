import cv2
import numpy as np
from go2_description import GO2_DESCRIPTION_URDF_PATH
import pybullet
import pybullet_data
from scipy.spatial.transform import Rotation as R
from go2_simulation.abstract_wrapper import AbstractSimulatorWrapper
from ament_index_python.packages import get_package_share_directory
import os

# TODO: Properly parametrize camera extrinsics through ROS params
CAMERA_HEIGHT_PX = 60
CAMERA_WIDTH_PX = 106
CAMERA_FOV = 87.0
NEAR_CLIP, FAR_CLIP = 0.01, 2.0

# in extreme-parkour, the camera looks [0, 1] degrees downwards.
# so 0.5 degrees = 0.0087 radians, 1m * sin(0.0087) ~= 0.0086m ~= 0.01m
CAMERA_TARGET_B = np.array([1.3, 0.0, 0.07, 1.0])
CAMERA_EYE_B = np.array([0.3, 0.0, 0.08, 1.0])

class BulletWrapper(AbstractSimulatorWrapper):
    def __init__(self, node, timestep):
        self.node = node
        self.init_pybullet(timestep)

        # Load obstacles
        pad: float = 0.1
        self.box_half_length = 0.45 + pad
        self.box_half_width = 0.75 + 5 * pad
        self.box_half_height = 0.20

        self.x_offset = 3.0
        self.z_offset = 0.0

        self.load_obstacles()


    def init_pybullet(self, timestep):
        cid = pybullet.connect(pybullet.SHARED_MEMORY)
        if cid < 0:
            pybullet.connect(pybullet.GUI, options="--opengl3")
        else:
            pybullet.connect(pybullet.GUI)

        # Load robot
        self.robot = pybullet.loadURDF(GO2_DESCRIPTION_URDF_PATH, [0, 0, 0.4])
        print('URDF loaded from:', GO2_DESCRIPTION_URDF_PATH)
        self.localInertiaPos = pybullet.getDynamicsInfo(self.robot, -1)[3]

        # Load ground plane and other obstacles
        self.env_ids = []  # Keep track of all obstacles

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = pybullet.loadURDF("plane.urdf")
        self.env_ids.append(self.plane_id)
        pybullet.resetBasePositionAndOrientation(self.plane_id, [0, 0, 0], [0, 0, 0, 1])

        self.ramp_id = pybullet.loadURDF(
            os.path.join(get_package_share_directory("go2_simulation"), "data/assets/obstacles.urdf")
        )
        self.env_ids.append(self.ramp_id)

        # Set time step
        pybullet.setTimeStep(timestep)

        # Prepare joint ordering
        self.joint_order = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        self.j_idx = []
        for j in self.joint_order:
            self.j_idx.append(self.get_joint_id(j))

        # Feet ids
        num_joints = pybullet.getNumJoints(self.robot)
        feet_names = [name + "_foot" for name in ("FR", "FL", "RR", "RL")]
        self.foot_link_names = feet_names
        self.feet_idx = [-1] * len(feet_names)

        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name in feet_names:
                foot_id = feet_names.index(link_name)
                self.feet_idx[foot_id] = (i, link_name)

        # Set robot initial config on the ground
        initial_q = [0.0, 1.00, -2.1, 0.0, 1.00, -2.1, 0, 1.00, -2.1, 0, 1.00, -2.1]
        for i, id in enumerate(self.j_idx):
            pybullet.resetJointState(self.robot, id, initial_q[i])

        # gravity and feet friction
        pybullet.setGravity(0, 0, -9.81)

        # Somehow this disable joint friction
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.j_idx,
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocities=[0.0 for i in range(12)],
            forces=[0.0 for i in range(12)],
        )


        self.w_T_b = np.eye(4)

        # Finite differences to compute acceleration
        self.dt = timestep
        self.v_last = None

        assert pybullet.isNumpyEnabled(), "Numpy not enabled in PyBullet!"

    def load_obstacles(self):
        half_extents = [self.box_half_length, self.box_half_width, self.box_half_height]

        col_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half_extents)
        vis_id = pybullet.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1]
        )

        num_boxes = 8
        for i in range(num_boxes):
            box_id = pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[self.x_offset, 0, self.z_offset],
            )

            info = pybullet.getDynamicsInfo(bodyUniqueId=box_id, linkIndex=-1)
            pybullet.changeDynamics(bodyUniqueId=box_id, linkIndex=-1, lateralFriction=1.0)

            self.env_ids.append(i)

            self.x_offset += 2 * self.box_half_length

            if i + 1 < num_boxes // 2:
                self.z_offset += self.box_half_height
            elif i + 1 > num_boxes // 2:
                self.z_offset -= self.box_half_height


    def get_joint_id(self, joint_name):
        num_joints = pybullet.getNumJoints(self.robot)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            if joint_info[1].decode("utf-8") == joint_name:
                return i
        return None  # Joint name not found

    def step(self, tau_cmd):
        # Set actuation
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot, jointIndices=self.j_idx, controlMode=pybullet.TORQUE_CONTROL, forces=tau_cmd
        )

        # Advance simulation by one step
        pybullet.stepSimulation()

        # Get new state
        joint_states = pybullet.getJointStates(self.robot, self.j_idx)
        joint_position = np.array([joint_state[0] for joint_state in joint_states])
        joint_velocity = np.array([joint_state[1] for joint_state in joint_states])

        linear_pose, angular_pose = pybullet.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = pybullet.getBaseVelocity(self.robot)  # Local world aligned frame

        # Offset pos because pybullet doesn't use the same origin
        rot_mat = R.from_quat(angular_pose).as_matrix()
        linear_pose += rot_mat @ self.localInertiaPos

        self.w_T_b[:3, :3] = rot_mat
        self.w_T_b[:3, 3] = linear_pose

        # Transform from Local world aligned to local
        linear_vel = rot_mat.T @ linear_vel
        angular_vel = rot_mat.T @ angular_vel

        # Take base offset into account for linear velocity
        linear_vel += rot_mat.T @ np.cross(self.localInertiaPos, angular_vel)

        q_current = np.concatenate((np.array(linear_pose), np.array(angular_pose), joint_position))
        v_current = np.concatenate((np.array(linear_vel), np.array(angular_vel), joint_velocity))
        a_current = ((v_current - self.v_last) / self.dt) if self.v_last is not None else np.zeros(6 + 12)
        f_current = np.zeros(4)

        self.v_last = v_current

        for i, foot_name in enumerate(self.foot_link_names):
            for collision_id in self.env_ids:
                foot_link_id = self.feet_idx[i][0]

                # Get contact points between foot and ground
                contact_points = pybullet.getContactPoints(
                    bodyA=self.robot,
                    bodyB=collision_id,
                    linkIndexA=foot_link_id
                )

                # Check if there are any contacts
                is_in_contact = len(contact_points) > 0

                if is_in_contact:
                    f_current[i] = 39.4  # roughly 1/4 of the robot mass (0th order approx)
                    break  # No need to check other obstacles for this foot

        return q_current, v_current, a_current, f_current

    def get_camera_image(self):
        ''' Get depth image from the robot's camera in the world frame '''
        rot_mat = self.w_T_b[:3, :3]

        up_vec = (rot_mat @ np.array([0.0, 0.0, 1.0])).tolist()
        CAMERA_EYE_W = self.w_T_b @ CAMERA_EYE_B
        CAMERA_TARGET_W = self.w_T_b @ CAMERA_TARGET_B

        view_matrix = pybullet.computeViewMatrix(
            CAMERA_EYE_W.tolist()[:3], CAMERA_TARGET_W.tolist()[:3], up_vec
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
        )[3][:-2, 4:-4] # (58, 98)

        # Convert depth buffer to liNEAR_CLIP depth
        depth = FAR_CLIP * NEAR_CLIP / (FAR_CLIP - (FAR_CLIP - NEAR_CLIP) * depth)
        depth = np.clip(depth, NEAR_CLIP, FAR_CLIP)
        depth = (depth - NEAR_CLIP) / (FAR_CLIP - NEAR_CLIP)

        # Resize the image with bicubic interpolation to (58, 87)
        depth = cv2.resize(
            depth, (87, 58), interpolation=cv2.INTER_CUBIC
        )

        depth = np.clip(depth, 0, 1)

        return (255. * depth).astype(np.uint8)

