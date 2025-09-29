import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

class RLDSVisualizer:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load_episodes(self):
        """Load episodes from a TFDS RLDS dataset (TFRecord backend)."""
        builder = tfds.builder_from_directory(self.data_directory)
        ds = builder.as_dataset(split='train', shuffle_files=False)

        self.episodes = []
        for episode in ds:
            steps = []
            for step in episode['steps']:
                # Convert observations and actions to numpy
                obs = {k: np.array(v) for k, v in step['observation'].items()}
                act = np.array(step['action']) if step['action'] is not None else None
                steps.append({
                    "observation": obs,
                    "action": act,
                    "reward": float(step['reward']),
                    "discount": float(step['discount']),
                    "is_first": bool(step['is_first']),
                    "is_last": bool(step['is_last']),
                    "is_terminal": bool(step['is_terminal']),
                })
                
            self.episodes.append(steps)
        print(f"Loaded {len(self.episodes)} episodes from TFRecords")

    def plot_observation(self, obs_key, episode_index=0, max_frames=5):
        """Plot a given observation over time for a specific episode.

        - Numeric obs → line plot over time
        - Image obs   → show sampled frames
        """
        if not hasattr(self, "episodes"):
            raise RuntimeError("Episodes not loaded. Call load_episodes() first.")

        steps = self.episodes[episode_index]
        obs_array = np.stack([step["observation"][obs_key] for step in steps])

        # Detect image vs numeric: if last dim is 3 or 1 → likely an image
        if obs_array.ndim >= 3 and obs_array.shape[-1] in (1, 3):
            # Plot a few frames (equally spaced)
            num_steps = obs_array.shape[0]
            frame_indices = np.linspace(0, num_steps - 1, max_frames, dtype=int)

            plt.figure(figsize=(15, 3))
            for i, idx in enumerate(frame_indices):
                plt.subplot(1, len(frame_indices), i + 1)
                img = obs_array[idx]
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)  # grayscale
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)
                plt.axis("off")
                plt.title(f"t={idx}")
            plt.suptitle(f"Image Observation: {obs_key} (Episode {episode_index})")
            plt.show()

        else:
            # Default numeric plot
            time = np.arange(obs_array.shape[0])
            plt.figure(figsize=(12, 6))
            if obs_array.ndim == 1:
                plt.plot(time, obs_array, label=obs_key)
            else:
                for i in range(obs_array.shape[1]):
                    plt.plot(time, obs_array[:, i], label=f"{obs_key}[{i}]")
            plt.xlabel("Timestep")
            plt.ylabel(obs_key)
            plt.title(f"Observation: {obs_key} (Episode {episode_index})")
            plt.legend()
            plt.show()


    def plot_3d_trajectory(self, obs_key, dims=(0, 1, 2), episode_index=0):
        """Plot 3D trajectory of a given observation key (e.g., end-effector)"""
        from mpl_toolkits.mplot3d import Axes3D

        steps = self.episodes[episode_index]
        obs_array = np.stack([step["observation"][obs_key] for step in steps])
        xyz = obs_array[:, dims[0]], obs_array[:, dims[1]], obs_array[:, dims[2]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*xyz, marker='o')
        ax.set_xlabel(f"{obs_key}[{dims[0]}]")
        ax.set_ylabel(f"{obs_key}[{dims[1]}]")
        ax.set_zlabel(f"{obs_key}[{dims[2]}]")
        ax.set_title(f"3D Trajectory: {obs_key} (Episode {episode_index})")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RLDS TFRecord dataset observations")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing TFRecord RLDS files")
    parser.add_argument("--obs_key", type=str, default="robot0_eef_pos",
                        help="Observation key to visualize (e.g., robot0_eef_pos)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to visualize")
    parser.add_argument("--plot_3d", action="store_true",
                        help="Plot as 3D trajectory if applicable")
    args = parser.parse_args()

    visualizer = RLDSVisualizer(data_directory=args.folder)
    visualizer.load_episodes()

    if args.plot_3d:
        visualizer.plot_3d_trajectory(args.obs_key, episode_index=args.episode)
    else:
        visualizer.plot_observation(args.obs_key, episode_index=args.episode)
