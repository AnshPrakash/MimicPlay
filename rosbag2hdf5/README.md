


# Config:

Use the config file to map the topics to required key names

eg.
```
obs_topics:
  /camera/rgb: rgb
  /joint_states: proprio
  /franka_state_controller/O_T_EE: ee_pose

action_topics:
  - /franka_gripper/joint_states
  - /franka_state_controller/joint_states_desired

# whether data is training or test data
mask: true 

```

# Run
```bash
python -m rosbag2hdf5.rosbag2robomimic --folder <folder with rosbag files> --config <path to config file>
```