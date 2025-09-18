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
            observation_info={
                k: tfds.features.Tensor(shape=self.shape_dict["observation"][k], dtype=tf.float32)
                for k in self.obs_topics.values()
            },
            action_info=tfds.features.Tensor(shape=self.shape_dict["action"], dtype=tf.float32),
            reward_info=tfds.features.Tensor(shape=(), dtype=tf.float32),
            discount_info=tfds.features.Tensor(shape=(), dtype=tf.float32),
            episode_metadata_info={
                "episode_id": tfds.features.Tensor(shape=(), dtype=tf.string),
                "agent_id": tfds.features.Tensor(shape=(), dtype=tf.string),
                "environment_config": tfds.features.Tensor(shape=(), dtype=tf.string),
                "experiment_id": tfds.features.Tensor(shape=(), dtype=tf.string),
                "invalid": tfds.features.Tensor(shape=(), dtype=tf.bool),
            },
        )

    def build_dataset(self):
        """Return RLDS dataset: tf.data.Dataset of episodes."""
        bag_files = sorted(self.folder.glob("*.bag"))
        if not bag_files:
            raise FileNotFoundError(f"No rosbag files found in {self.folder}")

        def episode_gen():
            for bag in bag_files:
                yield self._rosbag_to_episode(bag)

        return iter(episode_gen())

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
                            self.shape_dict["action"] = (act.shape[0] + 1, )  # +1 for gripper state
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

                # If this topic belongs to obs
                if topic in self.obs_topics:
                    key = self.obs_topics[topic]
                    try:
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        obs_dict[key].append(msg_to_numpy(msg))
                        if key == "gripper_joint_states":
                            positions = np.array(msg.position)
                            drift = positions[0]
                            last_gripper_state = 1 if drift < 0.02 else -1 # if dist < 0.02, assume gripper is closed
                    except Exception as e:
                        print(f"[WARN] Skipping obs topic {topic}: {e}")
                        raise e

                # If this topic belongs to actions
                if topic in self.action_topics:
                    try:
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        actions_list.append( 
                                            np.concatenate([
                                                msg_to_numpy(msg),
                                                np.array([last_gripper_state])]) 
                                            )
                    except Exception as e:
                        print(f"[WARN] Skipping action topic {topic}: {e}")
                        raise

        T = len(actions_list)

        # build per-step dicts
        steps = []
        for t in range(T):
            step = {
                "is_first": np.bool_(t == 0),
                "is_last": np.bool_(t == T - 1),
                "observation": {k: np.asarray(obs_dict[k][t], dtype=np.float32) for k in obs_dict.keys()},
                "action": np.asarray(actions_list[t], dtype=np.float32),
                "reward": np.float32(0.0),
                "discount": np.float32(1.0),
                "is_terminal": np.bool_(t == T - 1),
            }

            steps.append(step)
        
        metadata = {
            "episode_id": str(rosbagfile).encode("utf-8"),
            "agent_id": "robot1",
            "environment_config": "default",
            "experiment_id": "exp001",
            "invalid": False,
        }

        return {
            "steps": steps,
            "episode_metadata": metadata,
        }

        
        
    def store_dataset(self, dataset, split_name: str = "train"):
        """Store the RLDS dataset to disk using TFDS backend writer."""
        save_dir = str(self.output_path)

        # Dataset config for RLDS
        dataset_config = self._get_ds_config()
        writer = tfds_backend_writer.TFDSBackendWriter(
            ds_config=dataset_config,
            data_directory=save_dir,
            split_name=split_name,
            max_episodes_per_file=1,  # adjust if needed
        )

        episode_id = 0
        for episode in dataset:
            print(f"Processing episode {episode_id}")
            steps = episode["steps"]
            metadata = episode["episode_metadata"]
            # Build StepData sequence
            for step in steps:
                step_obj = step_data.StepData(
                    action=step["action"],
                    timestep=dm_env.TimeStep(
                        step_type=(
                            dm_env.StepType.FIRST
                            if step["is_first"]
                            else dm_env.StepType.LAST
                            if step["is_last"]
                            else dm_env.StepType.MID
                        ),
                        reward=step["reward"],
                        discount=step["discount"],
                        observation=step["observation"],
                    ),
                )
                writer._record_step(step_obj, is_new_episode=step["is_first"])
            writer.set_episode_metadata(metadata)
            writer._write_and_reset_episode()
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
    
