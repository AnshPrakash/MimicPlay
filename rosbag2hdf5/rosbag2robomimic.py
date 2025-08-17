import os
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

from rosbag2hdf5.util.msg_numpyfication import msg_to_numpy


class ToRobomimic:
    """
    Purpose of this class: convert the rosbags in a folder into robomimic HDF5 format
    """

    def __init__(self, folder, obs_topics: dict, action_topics: list, mask: bool = True):
        """
        Input:
            folder: a folder containing rosbag files with same topics
            obs_topics: dict { ros_topic_name: robomimic_obs_key }
            action_topics: list of topics in the order of concatenation
            mask: True => training data, False => validation data
        """
        self.folder = Path(folder)
        self.obs_topics = obs_topics
        self.action_topics = action_topics
        self.mask = mask
        self.demo_counter = 0

        # Output file path (timestamped)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = Path("robomimic_hdf5")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        output_name = f"robomimic_dataset_{timestamp}.hdf5"
        self.output_path = self.output_dir / output_name

        # Create single HDF5 file for writing
        self.h5file = h5py.File(self.output_path, "w")
        self.data_group = self.h5file.create_group("data")
        self.mask_group = self.h5file.create_group("mask")

        # Create mask datasets
        self.mask_group.create_dataset("train" if mask else "valid", shape=(0,), maxshape=(None,), dtype="i8")

        # Rosbag deserializer
        self.typestore = get_typestore(Stores.ROS1_NOETIC)


    def add_demo(self, rosbagfile: str):
        """
        Add the rosbag contents as a demo into the HDF5 file
        """
        rosbagfile = Path(rosbagfile)
        demo_name = f"demo_{self.demo_counter}"
        demo_group = self.data_group.create_group(demo_name)

        print(f"[INFO] Processing {rosbagfile} -> {demo_name}")

        # Store obs and actions
        obs_dict = {key: [] for key in self.obs_topics.values()}
        actions_list = []

        with Reader(rosbagfile) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic

                # If this topic belongs to obs
                if topic in self.obs_topics:
                    key = self.obs_topics[topic]
                    try:
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        obs_dict[key].append(self._msg_to_numpy(msg))
                    except Exception as e:
                        print(f"[WARN] Skipping obs topic {topic}: {e}")

                # If this topic belongs to actions
                if topic in self.action_topics:
                    try:
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                        actions_list.append(self._msg_to_numpy(msg))
                    except Exception as e:
                        print(f"[WARN] Skipping action topic {topic}: {e}")

        # Convert obs to datasets
        obs_group = demo_group.create_group("obs")
        for key, values in obs_dict.items():
            obs_group.create_dataset(key, data=np.stack(values))

        # Actions
        demo_group.create_dataset("actions", data=np.stack(actions_list))

        # States (optional – could be some obs concatenation)
        # demo_group.create_dataset("states", data=...)

        # Update mask
        mask_type = "train" if self.mask else "valid"
        dset = self.mask_group[mask_type]
        dset.resize((dset.shape[0] + 1,))
        dset[-1] = self.demo_counter

        self.demo_counter += 1


    def process_rosbags(self):
        """
        Iterate through all rosbags in the folder and add them to HDF5
        """
        bag_files = sorted(self.folder.glob("*.bag"))
        if not bag_files:
            print(f"[ERROR] No rosbag files found in {self.folder}")
            return

        for bag in bag_files:
            self.add_demo(bag)

        # Close file after processing
        self.h5file.close()
        print(f"[INFO] Saved robomimic dataset: {self.output_path}")


    def _msg_to_numpy(self, msg):
        """
        Helper: convert ROS msg -> numpy array
        (you'll need to customize based on message type)
        """
        # Example: assume msg has "position" and "velocity" attributes
        return msg_to_numpy(msg)
        

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
    converter = ToRobomimic(
        folder=args.folder,
        obs_topics=obs_topics,
        action_topics=action_topics,
        mask=mask
    )

    # Process all rosbags in folder
    converter.process_rosbags()

