import numpy as np
from unitree_description import GO2_DESCRIPTION_URDF_PATH
import pybullet
import pybullet_data
from scipy.spatial.transform import Rotation as R
from go2_simulation.abstract_wrapper import AbstractSimulatorWrapper
from ament_index_python.packages import get_package_share_directory
import os


class BulletWrapper(AbstractSimulatorWrapper):
    def __init__(self, node, timestep):
        self.node = node
        self.init_pybullet(timestep)

    def init_pybullet(self, timestep):
        cid = pybullet.connect(pybullet.SHARED_MEMORY)
        if cid < 0:
            pybullet.connect(pybullet.GUI, options="--opengl3")
        else:
            pybullet.connect(pybullet.GUI)

        # Load robot
        self.robot = pybullet.loadURDF(GO2_DESCRIPTION_URDF_PATH, [0, 0, 0.4])
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

        # Finite differences to compute acceleration
        self.dt = timestep
        self.v_last = None

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

        # Get feet contact
        for i, (joint_idx, link_name) in enumerate(self.feet_idx):
            # Check contact point with any obstacle (ground included)
            contact_points = []
            for id in self.env_ids:
                contact_points += pybullet.getClosestPoints(self.robot, id, 0.005, joint_idx)
            if len(contact_points) > 0:  # If contact
                f_current[i] = 39.4  # roughly 1/4 of the robot mass (0th order approx)

        return q_current, v_current, a_current, f_current
