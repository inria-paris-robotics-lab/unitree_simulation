from typing import Tuple, List

Configuration = List[float]
Velocities = List[float]
Accelerations = List[float]
Torques = List[float]
FeetForces = List[float]


class AbstractSimulatorWrapper:
    def step(tau: Torques) -> Tuple[Configuration, Velocities, Accelerations, FeetForces]:
        """
        Take as input torque commands for each joint and ouptut the robot new state.
        Note: the base velocity and acceleration (the first 6 values of the Velocities and Accelerations vector) are expressed in the local frame of the robot base.
        """
        raise NotImplementedError

    def unlock_base() -> None:
        """
        Unlock the robot base. Must be called only once.
        Explanation: The robot should be spawned with it's free-flyer fixed/lock in the world frame.
        This simulate that the real robot would start hanged by the shoulder. After the init procedure is over, the user would put the robot on the ground, which is what this method is modelling.
        """
        raise NotImplementedError

    def reset() -> None:
        """
        Reset the simulation as if the simulator was just instantiated and ready to start. (Must re-lock the base for instance)
        """
        raise NotImplementedError
