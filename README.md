# Unitree Go2 ros2 simulation

This ros2 package allows you to simulate the unitree go2 robot and mimic its ROS2 API. (by replicating the topics most used). The physics is simulated using pybullet or [Simple](https://github.com/Simple-Robotics/Simple).

Dependencies
------------

* The **unitree_ros** stack (please find instructions, [here](https://github.com/unitreerobotics/unitree_ros2/tree/master)).
* [unitree_description](https://github.com/inria-paris-robotics-lab/unitree_description) : Custom package containing robot model
* Pybullet

# Usage
Launch the simulation (GO2 or G1):
```bash
ros2 launch unitree_simulation launch_simulation.py robot:=<go2/g1>
```
### In another terminal:

Unlock the base of the robot :
```bash
ros2 topic pub /unlock_base std_msgs/msg/Empty --once 
```

Reset the simulation:
```bash
ros2 topic pub /unlock_base std_msgs/msg/Empty --once 
```
