


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



## RLDS

To read RLDS existing datasets, we can use `tfds` package



```
pip install tensorflow-datasets
pip install envlogger[tfds]
```

```
conda create -n rlds python=3.11
conda activate rlds

pip install rlds[tensorflow]
pip install tfds-nightly
pip install envlogger[tfds]
```

```
pip install -r requirements_rlds.txt 
```


Follow this [tutorial](https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=tErv4WRmgTjE) for performing other transformations to RLDS dataset

**How to read the stored data?**

Check the documentation from [google-deepming/envlogger](https://github.com/google-deepmind/envlogger/tree/main)

## Convert to Rosbag to RLDS

```bash
python -m rosbag2hdf5.rosbag_to_rlds --folder <path-to-folder-with-rosbags> --config <path to MimicPlay/rosbag2hdf5/config.yaml>
```

!! Note: In the `rosbag2hdf5/config.yaml`, include `image` as a substring for the image observation.

**Example**:

```
/zedB/zed_node_B/left/image_rect_color: agentview_image_2 # back camera
```

## Visualise RLDS record

```bash
python -m rosbag2hdf5.visualise_rlds --folder /home/ansh/IROBMAN/code/MimicPlay/rlds_dataset/rlds_20250918_153449/ --plot_3d
```

or 

```
python -m rosbag2hdf5.visualise_rlds --folder /home/ansh/IROBMAN/code/MimicPlay/rlds_dataset/rlds_20250918_153449/ 
```

Check `visualise_rlds.py` for more options to visualise

## Example training script

Install pytorch
```
pip install pytorch
```

Train with the following:
```bash
python -m rosbag2hdf5.example.example_bc_train --folder /home/ansh/IROBMAN/code/MimicPlay/rlds_dataset/rlds_20251029_111900 --obs_key robot0_eef_pos
```

## Further future improvements

1. One can convert to [LeRobot format](https://huggingface.co/docs/lerobot/main/porting_datasets_v3)




