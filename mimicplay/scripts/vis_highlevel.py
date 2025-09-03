"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    bddl_file (str): if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)

    video_prompt (str): if provided, a task video prompt is loaded and used in the evaluation rollouts

    debug (bool): set this flag to run a quick training run for debugging purposes
"""

import argparse
import json
import numpy as np
import time
import os
import psutil
import sys
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import mimicplay.utils.file_utils as FileUtils


from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import get_exp_dir, rollout_with_stats, load_data_for_training

from mimicplay.utils.triangulation import CameraModel
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np


ZEDA_LEFT_CAM = CameraModel(
        fx=1060.0899658203125,
        fy=1059.0899658203125,
        cx=958.9099731445312,
        cy=561.5670166015625,
        R_wc=R.from_quat([0.81395177, -0.40028226, -0.07631803, -0.41404371]).as_matrix(),
        t_wc=np.array([0.11261126, -0.52195948, 0.55795671])
    )

def get_traj(policy, data):
    return policy.policy.get_action(data['obs'], data['goal_obs'])

def vis_sample(ckpt_path, sample_idx, dataset=None):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    model.start_episode()

    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    ObsUtils.initialize_obs_utils_with_config(config)

    if dataset:
        config.unlock()
        config.train.data = dataset
        config.lock()

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    # env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=1,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    for i, batch in enumerate(valid_loader):
        if i == sample_idx:
            obs = batch   
            break

    # obs = valid_loader.dataset[sample_idx]

    obs = model.policy.process_batch_for_training(obs)
    obs = model.policy.postprocess_batch_for_training(obs, obs_normalization_stats=None)
    # import pdb
    # pdb.set_trace()
    traj = get_traj(model, obs)

    start_img = obs['obs']['agentview_image'][0, :, :, :].permute(1, 2, 0).cpu().numpy()
    goal_img = obs['goal_obs']['agentview_image'][0, :, :, :].permute(1, 2, 0).cpu().numpy()
    traj = traj.reshape(-1, 3)
    points2d = []
    
    for i in range(traj.shape[0]):
        point2d = ZEDA_LEFT_CAM.project_point(traj[i, :])
        points2d.append(point2d)
    points2d = np.array(points2d)  # (N, 2)

    import pdb
    pdb.set_trace()

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: start image with trajectory
    axes[0].imshow(start_img)
    axes[0].scatter(points2d[:, 0], points2d[:, 1], c='red', s=20, marker='o')
    axes[0].plot(points2d[:, 0], points2d[:, 1], c='blue', linewidth=2)  # connect points
    axes[0].set_title("Start Image with Projected Trajectory")
    axes[0].axis("off")

    # --- Right: goal image
    axes[1].imshow(goal_img)
    axes[1].set_title("Goal Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def main(args):
    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        vis_sample(ckpt_path=args.ckpt_path, sample_idx=args.sample_idx, dataset=args.dataset)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default=None,
    #     help="(optional) path to a config json that will be used to override the default settings. \
    #         If omitted, default settings are used. This is the preferred way to run experiments.",
    # )

    # # Algorithm Name
    # parser.add_argument(
    #     "--algo",
    #     type=str,
    #     help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    # )

    # # Experiment Name (for tensorboard, saving models, etc.)
    # parser.add_argument(
    #     "--name",
    #     type=str,
    #     default=None,
    #     help="(optional) if provided, override the experiment name defined in the config",
    # )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # parser.add_argument(
    #     "--bddl_file",
    #     type=str,
    #     default=None,
    #     help="(optional) if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)",
    # )

    # parser.add_argument(
    #     "--video_prompt",
    #     type=str,
    #     default=None,
    #     help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    # )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)

