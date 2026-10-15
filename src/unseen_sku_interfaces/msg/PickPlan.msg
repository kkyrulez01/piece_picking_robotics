# Planning-only pick sequence; trajectories are NOT execution commands.
# Pose frames are the MoveIt planning frame (normally base_link).
geometry_msgs/PoseStamped pre_pick_pose
geometry_msgs/PoseStamped pick_pose
geometry_msgs/PoseStamped retract_pose
trajectory_msgs/JointTrajectory pre_pick_trajectory
trajectory_msgs/JointTrajectory pick_trajectory
trajectory_msgs/JointTrajectory retract_trajectory
