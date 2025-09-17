import uuid
import numpy as np
import tensorflow as tf
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores
from rosbag2hdf5.util.msg_numpyfication import msg_to_numpy


class RosbagToRLDS:
    def __init__(self, folder, obs_topics: dict, action_topics: list):
        self.folder = Path(folder)
        self.obs_topics = obs_topics
        self.action_topics = action_topics
        self.typestore = get_typestore(Stores.ROS1_NOETIC)

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
                        k: tf.TensorSpec(shape=None, dtype=tf.float32)
                        for k in self.obs_topics.values()
                    },
                    "action": tf.TensorSpec(shape=None, dtype=tf.float32),
                    "reward": tf.TensorSpec((), tf.float32),
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

        steps = {
            "is_first": [t == 0 for t in range(T)],
            "is_last":  [t == T - 1 for t in range(T)],
            "observation": {
                k: [obs_dict[k][t] for t in range(T)] for k in obs_dict.keys()
            },
            "action": actions_list,
            "reward": [0.0 for _ in range(T)],
            "is_terminal": [t == T - 1 for t in range(T)],
        }
        steps_ds = tf.data.Dataset.from_tensor_slices(steps)

        # RLDS episode dict
        return {
            "steps": steps_ds,
            "episode_metadata": {
                "episode_id": str(uuid.uuid4()),
                "agent_id": "robot0",
                "environment_config": "default_env",
                "experiment_id": "exp001",
                "invalid": False,
            },
        }


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

    # Process all rosbags in folder
    for episode in converter.build_dataset():
        print(episode["episode_metadata"])
        for step in episode["steps"]:
            print(step)
        break
        


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
