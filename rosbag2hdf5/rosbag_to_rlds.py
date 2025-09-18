import uuid
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import dm_env
from envlogger.backends import tfds_backend_writer, rlds_utils
from tensorflow_datasets.rlds import rlds_base
from envlogger import step_data
import tensorflow_datasets as tfds
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores
from rosbag2hdf5.util.msg_numpyfication import msg_to_numpy


class RosbagToRLDS:
    def __init__(self, folder, obs_topics: dict, action_topics: list):
        self.folder = Path(folder)
        self.obs_topics = obs_topics
        self.action_topics = action_topics
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        
        self.output_dir = Path("rlds_dataset")

        # Output file path (timestamped)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_name = f"rlds_{timestamp}"
        self.output_path = self.output_dir / output_name
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        ## To get the dataset config
        self.shape_dict = {}
        self.shape_dict["observation"] = {k: None for k in self.obs_topics.values()}
        self.shape_dict["action"] = None   # action shape will be determined later
        self._determine_shapes()  # fills self.shape_dict with shapes for obs & action

        
    def _get_ds_config(self) -> rlds_base.DatasetConfig:
        """Build DatasetConfig based on observation dict keys."""

        return rlds_base.DatasetConfig(
            name="rosbag_rlds",  # dataset name
            # observation_info={
            #     k: tf.TensorSpec(shape=self.shape_dict["observation"][k], dtype=tf.float32)
            #     for k in self.obs_topics.values()
                
            # },
            # action_info={
            #     "shape": self.shape_dict["action"],
            #     "dtype": tf.float32,
            # },
            # reward_info={
            #     "shape": (),
            #     "dtype": tf.float32,
            # },
            # discount_info={
            #     "shape": (),
            #     "dtype": tf.float32,
            # }
        )

    def build_dataset(self):
        """Return RLDS dataset: tf.data.Dataset of episodes."""
        bag_files = sorted(self.folder.glob("*.bag"))
        if not bag_files:
            raise FileNotFoundError(f"No rosbag files found in {self.folder}")

        def episode_gen():
            for bag in bag_files:
                yield self._rosbag_to_episode(bag)

        output_signature = {
            "steps": tf.data.DatasetSpec(
                element_spec={
                    "is_first": tf.TensorSpec((), tf.bool),
                    "is_last": tf.TensorSpec((), tf.bool),
                    "observation": {
                        k: tf.TensorSpec(shape=self.shape_dict["observation"][k], dtype=tf.float32)
                        for k in self.obs_topics.values()
                    },
                    "action": tf.TensorSpec(shape=self.shape_dict["action"], dtype=tf.float32),
                    "reward": tf.TensorSpec((), tf.float32),
                    "discount": tf.TensorSpec((), tf.float32),
                    "is_terminal": tf.TensorSpec((), tf.bool),
                }
            ),
            "episode_metadata": {
                "episode_id": tf.TensorSpec((), tf.string),
                "agent_id": tf.TensorSpec((), tf.string),
                "environment_config": tf.TensorSpec((), tf.string),
                "experiment_id": tf.TensorSpec((), tf.string),
                "invalid": tf.TensorSpec((), tf.bool),
            },
        }

        return tf.data.Dataset.from_generator(episode_gen, output_signature=output_signature)

    def _determine_shapes(self):
        """Determine the shape of the observations and actions by inspecting the data."""
        bag_files = sorted(self.folder.glob("*.bag"))
        if not bag_files:
            raise FileNotFoundError(f"No rosbag files found in {self.folder}")

        for bag in bag_files:
            with Reader(bag) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    topic = connection.topic
                    if topic in self.obs_topics:
                        key = self.obs_topics[topic]
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        arr = msg_to_numpy(msg)
                        if self.shape_dict["observation"][key] is None:
                            self.shape_dict["observation"][key] = arr.shape

                    if topic in self.action_topics:
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        act = msg_to_numpy(msg)
                        if self.shape_dict["action"] is None:
                            self.shape_dict["action"] = act.shape
            break  # Only need to inspect the first bag
        print("Determined shapes:")
        print("Observation shapes:", self.shape_dict["observation"])
        print("Action shape:", self.shape_dict["action"])
        

    def _rosbag_to_episode(self, rosbagfile):
        """Convert one rosbag into RLDS episode (dict with steps + metadata)."""
        obs_dict = {key: [] for key in self.obs_topics.values()}
        actions_list = []
        last_gripper_state = -1

        with Reader(rosbagfile) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                if topic in self.obs_topics:
                    key = self.obs_topics[topic]
                    msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                    arr = msg_to_numpy(msg)
                    obs_dict[key].append(tf.convert_to_tensor(arr, dtype=tf.float32))

                    if key == "gripper_joint_states":
                        positions = tf.convert_to_tensor(msg.position, dtype=tf.float32)
                        last_gripper_state = tf.cond(
                            positions[0] < 0.02,
                            lambda: tf.constant(1, dtype=tf.int32),
                            lambda: tf.constant(-1, dtype=tf.int32),
                        )

                if topic in self.action_topics:
                    msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                    act = tf.concat(
                        [tf.convert_to_tensor(msg_to_numpy(msg), dtype=tf.float32),
                        tf.cast([last_gripper_state], tf.float32)],
                        axis=0,
                    )
                    actions_list.append(act)

        T = len(actions_list)

        # build per-step dicts
        step_dicts = []
        for t in range(T):
            step = {
                "is_first": t == 0,
                "is_last": t == T - 1,
                "observation": {k: obs_dict[k][t] for k in obs_dict.keys()},
                "action": actions_list[t],
                "reward": 0.0,
                "discount": 1.0,
                "is_terminal": t == T - 1,
            }
        step_dicts.append(step)
        
        metadata = {
            "episode_id": tf.constant(str(rosbagfile).encode("utf-8")),
            "agent_id": tf.constant("robot1"),
            "environment_config": tf.constant("default"),
            "experiment_id": tf.constant("exp001"),
            "invalid": tf.constant(False),
        }

        # wrap into tf.data.Dataset
        steps_ds = tf.data.Dataset.from_generator(
            lambda: iter(step_dicts),
            output_signature={
                "is_first": tf.TensorSpec((), tf.bool),
                "is_last": tf.TensorSpec((), tf.bool),
                "observation": {
                    k: tf.TensorSpec(shape=self.shape_dict["observation"][k], dtype=tf.float32)
                    for k in obs_dict.keys()
                },
                "action": tf.TensorSpec(shape=self.shape_dict["action"], dtype=tf.float32),
                "reward": tf.TensorSpec((), tf.float32),
                "discount": tf.TensorSpec((), tf.float32),
                "is_terminal": tf.TensorSpec((), tf.bool),
            },
        )
        
        # steps = {
        #     "is_first": [t == 0 for t in range(T)],
        #     "is_last":  [t == T - 1 for t in range(T)],
        #     "observation": {
        #         k: [obs_dict[k][t] for t in range(T)] for k in obs_dict.keys()
        #     },
        #     "action": actions_list,
        #     "reward": [0.0 for _ in range(T)],
        #     "discount": [1.0 for _ in range(T)],
        #     "is_terminal": [t == T - 1 for t in range(T)],
        # }
        # steps_ds = tf.data.Dataset.from_tensor_slices(steps)

        # RLDS episode dict
        # from ipdb import set_trace; set_trace()
        return {
            "steps": steps_ds,
            "episode_metadata": metadata,
        }

    def store_dataset(self, dataset):
        """Store the RLDS dataset to disk."""
        # tf.data.experimental.save(dataset, self.output_path)
        # tf.data.Dataset.save(dataset, self.output_path)
        # print(f"[INFO] RLDS dataset saved to {self.output_path}")
        save_path = str(self.output_path)   # <--- convert Path to str
        tf.data.Dataset.save(dataset, save_path)
        print(f"[INFO] RLDS dataset saved to {save_path}")
        
        
    def store_dataset(self, dataset, split_name: str = "train"):
        
        save_dir = str(self.output_path)  

        # Create a backend writer
        dataset_config = self._get_ds_config()
        writer = tfds_backend_writer.TFDSBackendWriter(
            ds_config=dataset_config,
            data_directory=save_dir,
            split_name=split_name,
            max_episodes_per_file=100,   # you can adjust this
        )

        episode_id = 0
        for episode in dataset:
            print(f"Processing episode {episode_id}")
            from ipdb import set_trace; set_trace()
            steps = episode["steps"]

            # Initialize a new episode
            ep_writer = writer.new_episode(episode_id=episode_id)

            prev_step = None
            for step in steps:
                # Convert your dict step into RLDS-compatible StepData
                s = step_data.StepData(
                    action=step["action"],
                    timestep=dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST if step["is_first"] else (
                            dm_env.StepType.LAST if step["is_last"] else dm_env.StepType.MID
                        ),
                        reward=step["reward"],
                        # discount=step["discount"],
                        observation=step["observation"],
                    ),
                )
                # Convert to RLDS step dict
                rlds_step = rlds_utils.to_rlds_step(prev_step, s)
                ep_writer.add_step(rlds_step)
                prev_step = s

            ep_writer.end_episode()
            episode_id += 1

        writer.close()
        print(f"Dataset successfully stored at {save_dir}")


if __name__ == "__main__":
    import argparse
    import yaml
    import os

    parser = argparse.ArgumentParser(description="Convert ROS bags to robomimic HDF5 format")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing ROS bag files")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML/JSON config file describing obs_topics, action_topics, and mask")

    args = parser.parse_args()

    # Load config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

    obs_topics = config["obs_topics"]
    action_topics = config["action_topics"]
    mask = config.get("mask", True)   # default True if not set

    # Create converter
    converter = RosbagToRLDS(
        folder=args.folder,
        obs_topics=obs_topics,
        action_topics=action_topics
    )

    dataset = converter.build_dataset()
    
    
    converter.store_dataset(dataset)
    
    # # Process all rosbags in folder
    # for episode in dataset.take(1):
    #     for step in episode["steps"].take(3):
    #         print(step)
        
        


# import tensorflow as tf

# raw_dataset = tf.data.TFRecordDataset("rlds_dataset/episode_0.tfrecord")

# feature_description = {
#     "observation": tf.io.VarLenFeature(tf.string),
#     "action": tf.io.FixedLenFeature([], tf.string),
#     "is_terminal": tf.io.FixedLenFeature([], tf.int64),
#     "language_instruction": tf.io.FixedLenFeature([], tf.string),
# }

# def _parse_fn(example_proto):
#     parsed = tf.io.parse_single_example(example_proto, feature_description)
#     action = tf.io.parse_tensor(parsed["action"], out_type=tf.float32)
#     obs = {}
#     for i, k in enumerate(["eef_pos", "eef_quat", "gripper_joint_states"]):  # adapt keys
#         obs[k] = tf.io.parse_tensor(parsed["observation"].values[i], out_type=tf.float32)
#     return {"observation": obs, "action": action, "is_terminal": parsed["is_terminal"]}

# dataset = raw_dataset.map(_parse_fn)

# for step in dataset.take(3):
#     print(step)
