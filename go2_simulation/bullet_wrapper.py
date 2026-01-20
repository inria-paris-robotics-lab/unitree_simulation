import numpy as np
from unitree_description import G1_DESCRIPTION_URDF_PATH
import pybullet
import pybullet_data
from scipy.spatial.transform import Rotation as R
from go2_simulation.abstract_wrapper import AbstractSimulatorWrapper
from ament_index_python.packages import get_package_share_directory
import os


class BulletWrapper(AbstractSimulatorWrapper):
    def __init__(self, node, timestep):
        self.node = node

        # init pybullet
        cid = pybullet.connect(pybullet.SHARED_MEMORY)
        if cid < 0:
            pybullet.connect(pybullet.GUI, options="--opengl3")
        else:
            pybullet.connect(pybullet.GUI)

        # Load robot
        self.robot = pybullet.loadURDF(G1_DESCRIPTION_URDF_PATH, [0, 0, 0.4])
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
        self.joint_name_unitree_order = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.joint_bullet_id = [self.get_joint_id(joint_name) for joint_name in self.joint_name_unitree_order]

        # Feet ids
        num_joints = pybullet.getNumJoints(self.robot)
        feet_names = [name + "_foot" for name in ()]
        self.feet_idx = [-1] * len(feet_names)

        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name in feet_names:
                foot_id = feet_names.index(link_name)
                self.feet_idx[foot_id] = (i, link_name)

        self.q_start = [0.0, 0.0, 0.641, 0.0, 0.0, 0.0, 1.0] + [0.0] * 29
        self.q_start[7 + self.joint_name_unitree_order.index("left_hip_pitch_joint")] = -0.5
        self.q_start[7 + self.joint_name_unitree_order.index("right_hip_pitch_joint")] = -0.5
        self.q_start[7 + self.joint_name_unitree_order.index("left_knee_joint")] = 1.0
        self.q_start[7 + self.joint_name_unitree_order.index("right_knee_joint")] = 1.0
        self.q_start[7 + self.joint_name_unitree_order.index("left_ankle_pitch_joint")] = -0.5
        self.q_start[7 + self.joint_name_unitree_order.index("right_ankle_pitch_joint")] = -0.5

        # gravity and feet friction
        pybullet.setGravity(0, 0, -9.81)

        # Locked base constraint ID
        self.fixed_base_constraint = None

        # Finite differences to compute acceleration
        self.dt = timestep
        self.v_last = None

        # Robot to intial state
        self.reset()

    def reset(self):
        # Set robot initial config on the ground
        pybullet.resetBasePositionAndOrientation(self.robot, self.q_start[:3], self.q_start[3:7])
        for i, id in enumerate(self.joint_bullet_id):
            if id:
                pybullet.resetJointState(self.robot, id, self.q_start[7 + i])

        # Somehow this disable joint friction
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=[joint_id for joint_id in self.joint_bullet_id if joint_id is not None],
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocities=[0.0 for joint_id in self.joint_bullet_id if joint_id is not None],
            forces=[0.0 for joint_id in self.joint_bullet_id if joint_id is not None],
        )

        # Lock torso (as if the robot was hanged)
        pos, orn = pybullet.getBasePositionAndOrientation(self.robot)
        if self.fixed_base_constraint is None:
            self.fixed_base_constraint = pybullet.createConstraint(
                parentBodyUniqueId=self.robot,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=pos,
                childFrameOrientation=orn,
            )

    def unlock_base(self):
        if self.fixed_base_constraint is not None:
            pybullet.removeConstraint(self.fixed_base_constraint)
            self.fixed_base_constraint = None

    def get_joint_id(self, joint_name):
        """
        Returns the pybullet id of the first **non-fixed** joint that match `joint_name`.
        Returns None otherwise
        """
        num_joints = pybullet.getNumJoints(self.robot)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            if joint_info[2] == pybullet.JOINT_FIXED:
                continue
            if joint_info[1].decode("utf-8") != joint_name:
                continue
            return i
        return None  # Joint name not found

    def step(self, tau_cmd):
        # Set actuation
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=[joint_id for joint_id in self.joint_bullet_id if joint_id is not None],
            controlMode=pybullet.TORQUE_CONTROL,
            forces=[tau_cmd[i] for i, joint_id in enumerate(self.joint_bullet_id) if joint_id is not None],
        )

        # Advance simulation by one step
        pybullet.stepSimulation()

        # Get new state
        joint_states_bullet = pybullet.getJointStates(
            self.robot, [joint_id for joint_id in self.joint_bullet_id if joint_id is not None]
        )
        joint_position = np.zeros(len(self.joint_bullet_id))
        joint_velocity = np.zeros(len(self.joint_bullet_id))
        j = 0
        for i, bullet_id in enumerate(self.joint_bullet_id):
            if bullet_id is not None:
                joint_position[i] = joint_states_bullet[j][0]
                joint_velocity[i] = joint_states_bullet[j][1]
                j += 1

        linear_pose, angular_pose = pybullet.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = pybullet.getBaseVelocity(self.robot)  # Local world aligned frame

        # Offset pos because pybullet doesn't use the same origin
        rot_mat = R.from_quat(angular_pose).as_matrix()
        linear_pose += rot_mat @ self.localInertiaPos

        # Transform from Local world aligned to local
        linear_vel = rot_mat.T @ linear_vel
        angular_vel = rot_mat.T @ angular_vel

        # Take base offset into account for linear velocity
        linear_vel += rot_mat.T @ np.cross(self.localInertiaPos, angular_vel)

        q_current = np.concatenate((np.array(linear_pose), np.array(angular_pose), joint_position))
        v_current = np.concatenate((np.array(linear_vel), np.array(angular_vel), joint_velocity))
        a_current = ((v_current - self.v_last) / self.dt) if self.v_last is not None else np.zeros(6 + 29)
        f_current = np.zeros(0)

        self.v_last = v_current

        # Get feet contact
        for i, (joint_idx, link_name) in enumerate(self.feet_idx):
            # Check contact point with any obstacle (ground included)
            contact_points = []
            for id in self.env_ids:
                contact_points += pybullet.getClosestPoints(self.robot, id, 0.005, joint_idx)
            if len(contact_points) > 0:  # If contact
                f_current[i] = 39.4  # roughly 1/4 of the robot mass (0th order approx)

        return q_current, v_current, a_current, f_current
