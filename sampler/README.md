# Sampler @ freq

Samples rosbag topics at a fixed rate.


## Requirement:
1. All the topics should have higher for freq than new freq, as this module is designed to only downsample
2. rosbag generated from `ROS NOETIC`

# Installation

```
cd sampler/
pip install -e .
```


# Freq estimate

This part is to check whether the data has freq more than required freq. We will not upsample or interpolate.


# Process raw data

`Input`: directory containing all the rosbags
`special topics`: For now, this would be grasp topic because it occur rarely and we don't want to loss this data.


## Start processing

```
    python -m sampler.rosbag_sampler --folder <folder-containing-raw-rosbags> --freq <new-lower-freq-for-robot-policy>
```


# Output

All the resampled topics are stored under `processed_folder`
Now, this can further be used for changing into any format


# Visualisation

Execute => 

```
 python -m sampler.freq_estimator --folder <folder-containing-rosbags> --visualise true
```

This will create histograms and show the estimated freq of a rosbag

It is useful for checking the raw rosbag and processed rosbag for any issue.




