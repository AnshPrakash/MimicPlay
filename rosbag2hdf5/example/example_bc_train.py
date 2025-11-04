# eg. python -m rosbag2hdf5.example.example_bc_train --folder /home/ansh/IROBMAN/code/MimicPlay/rlds_dataset/rlds_20251029_111900 --obs_key robot0_eef_pos

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds


# ============================================================
# 1. Load RLDS TFRecord Dataset
# ============================================================
def load_rlds_dataset(folder):
    builder = tfds.builder_from_directory(folder)
    ds = builder.as_dataset(split='train', shuffle_files=False)

    episodes = []

    for episode in ds:
        steps = []
        for step in episode['steps']:
            obs = {k: np.array(v) for k, v in step['observation'].items()}
            act = np.array(step['action']) if step['action'] is not None else None
            
            steps.append({
                "observation": obs,
                "action": act,
            })
        episodes.append(steps)

    print(f"✅ Loaded {len(episodes)} episodes")
    return episodes


# ============================================================
# 2. PyTorch Dataset for Behavioural Cloning
# ============================================================
class BCDataset(Dataset):
    def __init__(self, episodes, obs_key):
        self.obs = []
        self.actions = []

        for ep in episodes:
            for step in ep:
                a = step["action"]
                o = step["observation"][obs_key]

                if a is None:
                    continue

                self.obs.append(o)
                self.actions.append(a)

        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)

        print(f"✅ BC dataset: {len(self.obs)} samples")

        # normalize if numeric
        if self.obs.ndim == 2:
            self.mean = self.obs.mean(axis=0)
            self.std = self.obs.std(axis=0) + 1e-6
            self.obs = (self.obs - self.mean) / self.std

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        o = torch.tensor(self.obs[idx], dtype=torch.float32)
        a = torch.tensor(self.actions[idx], dtype=torch.float32)
        return o, a


# ============================================================
# 3. Simple MLP Policy
# ============================================================
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Behavioural Cloning Training Loop
# ============================================================
def train_bc(model, loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for obs, act in loader:
            pred = model(obs)
            loss = loss_fn(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")


# ============================================================
# 5. Main entry point
# ============================================================
def run_bc(folder, obs_key="robot0_eef_pos", batch_size=64):
    episodes = load_rlds_dataset(folder)

    dataset = BCDataset(episodes, obs_key)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample_obs = dataset.obs[0]
    obs_dim = sample_obs.shape[-1]
    act_dim = dataset.actions.shape[-1]

    model = MLPPolicy(obs_dim, act_dim)

    print("✅ Starting BC training…")
    train_bc(model, loader)
    print("✅ Training complete")

    torch.save(model.state_dict(), "bc_policy.pt")
    print("✅ Saved model → bc_policy.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str,
                        help="Folder containing RLDS TFRecord dataset")
    parser.add_argument("--obs_key", type=str, default="robot0_eef_pos")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_bc(args.folder, args.obs_key, args.batch_size)

# Example Output:
"""
 python -m rosbag2hdf5.example.example_bc_train --folder /home/ansh/IROBMAN/code/MimicPlay/rlds_dataset/rlds_20251029_111900 --obs_key robot0_eef_pos
2025-11-04 11:46:51.421775: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1762253213.310854  475186 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 8056 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3080, pci bus id: 0000:0b:00.0, compute capability: 8.6
2025-11-04 11:46:53.579043: I tensorflow/core/kernels/data/tf_record_dataset_op.cc:396] The default buffer size is 262144, which is overridden by the user specified `buffer_size` of 8388608
2025-11-04 11:46:54.171340: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2025-11-04 11:46:54.639452: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2025-11-04 11:46:55.361324: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
✅ Loaded 3 episodes
✅ BC dataset: 1501 samples
✅ Starting BC training…
Epoch 1/20 | Loss: 0.1287
Epoch 2/20 | Loss: 0.0581
Epoch 3/20 | Loss: 0.0391
Epoch 4/20 | Loss: 0.0302
Epoch 5/20 | Loss: 0.0274
Epoch 6/20 | Loss: 0.0249
Epoch 7/20 | Loss: 0.0230
Epoch 8/20 | Loss: 0.0233
Epoch 9/20 | Loss: 0.0218
Epoch 10/20 | Loss: 0.0212
Epoch 11/20 | Loss: 0.0201
Epoch 12/20 | Loss: 0.0198
Epoch 13/20 | Loss: 0.0189
Epoch 14/20 | Loss: 0.0183
Epoch 15/20 | Loss: 0.0185
Epoch 16/20 | Loss: 0.0174
Epoch 17/20 | Loss: 0.0179
Epoch 18/20 | Loss: 0.0171
Epoch 19/20 | Loss: 0.0170
Epoch 20/20 | Loss: 0.0166
✅ Training complete
✅ Saved model → bc_policy.pt
"""