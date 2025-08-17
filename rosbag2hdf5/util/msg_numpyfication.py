import numpy as np

def pose_to_numpy(msg):
    """
    Convert geometry_msgs/Pose to a NumPy array.
    
    Message structure (Pose.msg):contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}:
      - position: geometry_msgs/Point (float64 x, y, z):contentReference[oaicite:5]{index=5}  
      - orientation: geometry_msgs/Quaternion (float64 x, y, z, w):contentReference[oaicite:6]{index=6}  

    Returns:
      np.ndarray of shape (7,) as [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w].
    """
    return np.array([
        msg.position.x, msg.position.y, msg.position.z,
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
    ], dtype=np.float64)


def pose_stamped_to_numpy(msg):
    """
    Convert geometry_msgs/PoseStamped to a NumPy array.
    
    Message structure (PoseStamped.msg):contentReference[oaicite:8]{index=8}:
      - header: std_msgs/Header (ignored in conversion)  
      - pose: geometry_msgs/Pose (position and orientation as above)  

    Returns:
      np.ndarray of shape (7,) as [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w].
    """
    # Reuse pose_to_numpy on the embedded pose field
    return pose_to_numpy(msg.pose)

def joint_state_to_numpy(msg):
    """
    Convert sensor_msgs/JointState to a NumPy array.
    
    Message structure (JointState.msg):contentReference[oaicite:10]{index=10}:
      - header: std_msgs/Header (ignored here)  
      - name: string[] (joint names, ignored in numeric output)  
      - position: float64[] (joint positions)  
      - velocity: float64[] (joint velocities, same length or empty)  
      - effort: float64[] (joint efforts, same length or empty)  

    Returns:
      np.ndarray of shape (3*N,) where N = number of joints. For joint i,
      the output contains [positions.., velocities.., efforts..] in sequence.
    """
    positions = np.array(msg.position)
    velocities = np.array(msg.velocity) if msg.velocity is not None else []
    efforts = np.array(msg.effort) if msg.effort is not None else []
    
    arr = np.concatenate([positions, velocities, efforts])
    
    return arr

# def joy_to_numpy(msg):
#     """
#     Convert sensor_msgs/Joy to a NumPy array.
    
#     Message structure (Joy.msg):contentReference[oaicite:12]{index=12}:
#       - header: std_msgs/Header (ignored)  
#       - axes: float32[] (axis values)  
#       - buttons: int32[] (button values)  

#     Returns:
#       np.ndarray containing [axes..., buttons...] as floats.
#     """
#     # Convert axes and buttons to numpy arrays
#     axes = np.array(msg.axes, dtype=np.float32) if msg.axes else np.array([], dtype=np.float32)
#     buttons = np.array(msg.buttons, dtype=np.float32) if msg.buttons else np.array([], dtype=np.float32)
#     return np.concatenate([axes, buttons])

# def image_to_numpy(msg):
#     """
#     Convert sensor_msgs/Image to a NumPy array.
    
#     Message structure (Image.msg):contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}:
#       - header: std_msgs/Header (ignored)  
#       - height: uint32 (rows)  
#       - width: uint32 (columns)  
#       - encoding: string (e.g., 'rgb8', 'bgr8', 'mono8')  
#       - is_bigendian: uint8 (byte order, ignored)  
#       - step: uint32 (row length in bytes, can verify data size)  
#       - data: uint8[] (pixel buffer, size = step*height)  

#     Returns:
#       np.ndarray with shape (height, width, C) or (height, width), dtype determined by encoding.
#     """
#     height = msg.height
#     width = msg.width
#     enc = msg.encoding.lower()
#     # Create a 1D array of the data buffer
#     np_data = np.frombuffer(msg.data, dtype=np.uint8)
#     # Handle common encodings
#     if enc in ['rgb8', 'bgr8']:
#         # 3 channels, 1 byte each
#         img = np_data.reshape((height, width, 3))
#     elif enc == 'mono8':
#         # 1 channel, 1 byte
#         img = np_data.reshape((height, width))
#     else:
#         # Unsupported encoding
#         raise NotImplementedError(f"Encoding '{msg.encoding}' not supported")
#     return img

# def grasp_action_goal_to_numpy(msg):
#     """
#     Convert franka_gripper/GraspActionGoal to a NumPy array.
    
#     Action goal definition (Grasp.action):contentReference[oaicite:19]{index=19}:
#       - width: float64 (target gripper width in meters)  
#       - epsilon: GraspEpsilon (tolerance window)  
#            - inner: float64:contentReference[oaicite:20]{index=20}  
#            - outer: float64:contentReference[oaicite:21]{index=21}  
#       - speed: float64 (m/s)  
#       - force: float64 (N)  

#     Returns:
#       np.ndarray [width, epsilon.inner, epsilon.outer, speed, force].
#     """
#     goal = msg.goal  # Goal fields inside the ActionGoal
#     inner = goal.epsilon.inner
#     outer = goal.epsilon.outer
#     return np.array([goal.width, inner, outer, goal.speed, goal.force], dtype=np.float64)

def msg_to_numpy(msg):
    """
    Convert a ROS message to a NumPy array based on its type.

    Determines msg._type and calls the corresponding conversion function.
    """
    # Mapping of message type to conversion function
    converters = {
        'geometry_msgs/msg/Pose': pose_to_numpy,
        # 'geometry_msgs/msg/PoseStamped': pose_stamped_to_numpy,
        'sensor_msgs/msg/JointState': joint_state_to_numpy,
        # 'sensor_msgs/msg/Image': image_to_numpy,
        # 'sensor_msgs/msg/Joy': joy_to_numpy,
        # 'franka_gripper/msg/GraspActionGoal': grasp_action_goal_to_numpy, # correct this later
    }
    # from ipdb import set_trace as bp; bp()
    msg_type = getattr(msg, '__msgtype__', None)
    if msg_type in converters:
        return converters[msg_type](msg)
    else:
        from ipdb import set_trace as bp; bp()
        raise ValueError(f"No converter implemented for message type '{msg_type}'")
