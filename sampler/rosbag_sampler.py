import argparse
from pathlib import Path
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import get_typestore, Stores
from sampler.freq_estimator import FreqEstimator
from config import TOPICS

class RosbagSampler:
    """
    Class to resample ROSBAG files to a fixed lower frequency.
    """

    
    def __init__(self, input_folder: str, output_folder: str = "processed_folder", target_freq: float = 10.0):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.target_freq = target_freq
        self.dt = 1.0 / target_freq
        self.output_folder.mkdir(exist_ok=True)
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        self.TOPICS = TOPICS

    def find_bag_files(self):
        """Find all .bag files in the input folder."""
        bag_files = sorted(self.input_folder.glob("*.bag"))
        if not bag_files:
            raise FileNotFoundError(f"No .bag files found in {self.input_folder}")
        return bag_files

    def load_bag_messages(self, bag_path: Path):
        """Load all relevant messages from a ROSBAG into memory."""
        topic_data = {topic: [] for topic in self.TOPICS}
        print(f"[INFO] Loading bag: {bag_path}")

        with Reader(str(bag_path)) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in self.TOPICS:
                    try:
                        msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)
                    except Exception as e:
                        print(f"[WARN] Could not deserialize {connection.topic} @ {timestamp}: {e}")
                        continue
                    topic_data[connection.topic].append((timestamp / 1e9, msg, connection.msgtype))

        return topic_data

    def resample_messages(self, topic_data):
        """Resample messages at fixed frequency using sample-and-hold method."""
        if not any(topic_data.values()):
            print("[WARN] No messages found for the specified topics.")
            return []

        # ---  Check that target_freq is lower than current median frequency ---
        freq_checker = FreqEstimator()
        for topic, msgs in topic_data.items():
            if len(msgs) < 2:
                continue
            timestamps = [t for t, _, _ in msgs]
            med_freq = freq_checker.compute_freq(topic, timestamps, visualise=False)
            assert med_freq > self.target_freq, (
                f"[ERROR] Target frequency ({self.target_freq} Hz) is higher than the "
                f"current median frequency ({med_freq:.2f} Hz) for topic '{topic}'."
            )


        start_time = min(vals[0][0] for vals in topic_data.values() if vals)
        end_time = max(vals[-1][0] for vals in topic_data.values() if vals)
        print(f"[INFO] Time range: {start_time:.3f} → {end_time:.3f} sec")

        processed_msgs = []
        current_indices = {topic: 0 for topic in self.TOPICS}
        last_values = {topic: None for topic in self.TOPICS}
        msgtypes = {topic: None for topic in self.TOPICS}

        t = start_time
        while t <= end_time:
            for topic in self.TOPICS:
                data_list = topic_data[topic]
                idx = current_indices[topic]

                while idx < len(data_list) and data_list[idx][0] <= t:
                    last_values[topic] = data_list[idx][1]
                    msgtypes[topic] = data_list[idx][2]
                    idx += 1
                current_indices[topic] = idx

                if last_values[topic] is not None:
                    processed_msgs.append((t, topic, last_values[topic], msgtypes[topic]))

            t += self.dt

        print(f"[INFO] Total messages in processed output: {len(processed_msgs)}")
        return processed_msgs, msgtypes

    def save_processed_bag(self, output_bag: Path, processed_msgs, msgtypes):
        """Write processed messages to a new ROSBAG."""
        print(f"[INFO] Saving processed bag to {output_bag}...")
        with Writer(str(output_bag)) as writer:
            connections = {}
            for topic in self.TOPICS:
                if msgtypes[topic] is not None:
                    connections[topic] = writer.add_connection(topic, msgtypes[topic])

            for ts_sec, topic, msg, msgtype in processed_msgs:
                conn = connections[topic]
                rawdata = self.typestore.serialize_ros1(msg, msgtype)
                writer.write(conn, int(ts_sec * 1e9), rawdata)
        print("[INFO] Done.")

    def process_all_bags(self):
        """Main method to process all ROSBAGs in the input folder."""
        bag_files = self.find_bag_files()
        for bag_file in bag_files:
            topic_data = self.load_bag_messages(bag_file)
            processed_msgs, msgtypes = self.resample_messages(topic_data)
            output_name = f"{bag_file.stem}_{int(self.target_freq)}Hz.bag"
            output_bag = self.output_folder / output_name
            if processed_msgs:
                self.save_processed_bag(output_bag, processed_msgs, msgtypes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample ROSBAG files to a fixed lower frequency.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing raw ROSBAG files.")
    parser.add_argument("--freq", type=float, required=True, help="New lower frequency (Hz) for resampling.")
    args = parser.parse_args()

    sampler = RosbagSampler(input_folder=args.folder, target_freq=args.freq)
    sampler.process_all_bags()
