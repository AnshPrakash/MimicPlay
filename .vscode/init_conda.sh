#!/bin/bash

source ~/.bashrc

conda activate mimicplay

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dreamteam/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

