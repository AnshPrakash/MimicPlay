

## Resampler's pseudocode:

```

start_time = Get the eariliest start-time from all the topics
end_time = Get the last time-stamp from all the topics


Divide the interval [start_time, end_time] by the target_freq

dt <= 1.0/ target_freq
t <= start_time
while t <= end_time:
    for topic in topics:
        `store` or `select` the msg from the topic which has the greatest timestamp, but timestamp is <= t
    
    for all topic:
        temporarily store the selected messasge in a list for each topic
    t <= t + dt

```