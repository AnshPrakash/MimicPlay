import argparse
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag1 import Reader
from pathlib import Path
from config import TOPICS


class FreqEstimator:
    def __init__(self, output_dir: Path= None):
        self.output_dir = output_dir
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_freq(self, topic: str, timestamps: list, visualise: bool = False):
        """
        Estimates the freq of the topic based on the stored timestamps
        """
        timestamps = np.array(timestamps)
        deltas = np.diff(timestamps)

        avg_freq = 1.0 / np.mean(deltas)
        mean_dt = np.mean(deltas)
        min_dt = np.min(deltas)
        max_dt = np.max(deltas)
        median_dt = np.median(deltas)
        med_freq = 1.0 / median_dt

        print(f"  Topic: {topic}")
        print(f"  Messages: {len(timestamps)}")
        print(f"  Avg freq: {avg_freq:.3f} Hz")
        print(f"  Median freq: {med_freq:.3f} Hz")
        print(f"  Min Δt: {min_dt:.4f} s")
        print(f"  Max Δt: {max_dt:.4f} s")
        print(f"  Median Δt: {median_dt:.4f} s\n")

        if visualise:
            # Plot and save histogram
            plt.figure(figsize=(8, 5))
            
            if np.allclose(deltas, deltas[0], rtol=1e-8, atol=1e-10):
                bins = [deltas[0]-1e-6, deltas[0]+1e-6]
            else:
                bins = 50
            plt.hist(deltas, bins=bins, color='steelblue', edgecolor='black')
            plt.axvline(median_dt, color='red', linestyle='--', label=f"Median Δt = {median_dt:.4f}s")
            plt.axvline(mean_dt, color='blue', linestyle='--', label=f"Mean Δt = {mean_dt:.4f}s")
            
            # Add median frequency text on the plot
            plt.text(0.95, 0.95, f"Median freq: {med_freq:.3f} Hz",
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
            
            plt.xlabel("Δt between messages (seconds)")
            plt.ylabel("Count")
            plt.title(f"Histogram of Δt for {topic}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            output_file = self.output_dir / f"{topic.strip('/').replace('/', '_')}_hist.png"
            plt.savefig(output_file, dpi=150)
            plt.close()

            print(f"  Saved histogram to: {output_file}")

        return med_freq

def process_rosbag(bag_path: Path, visualise: bool):
    print(f"\n[INFO] Processing bag: {bag_path}")
    topic_timestamps = {topic: [] for topic in TOPICS}

    with Reader(str(bag_path)) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in topic_timestamps:
                topic_timestamps[connection.topic].append(timestamp / 1e9)  # ns -> s

    output_dir = Path("histograms") / bag_path.stem if visualise else Path("histograms")
    estimator = FreqEstimator(output_dir)

    for topic in TOPICS:
        timestamps = topic_timestamps[topic]
        if len(timestamps) < 2:
            print(f"[WARN] Not enough messages for topic: {topic}")
            continue
        estimator.compute_freq(topic, timestamps, visualise=visualise)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate frequencies from ROSBAG files.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing ROSBAG files.")
    parser.add_argument("--visualise", type=lambda x: x.lower() == "true", default=False,
                        help="Whether to save histograms (true/false).")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Provided folder does not exist: {folder_path}")

    # Filter only .bag files
    bag_files = sorted([f for f in folder_path.glob("*.bag")])
    if not bag_files:
        print(f"[ERROR] No .bag files found in {folder_path}")
        exit(1)

    for bag_file in bag_files:
        process_rosbag(bag_file, args.visualise)
