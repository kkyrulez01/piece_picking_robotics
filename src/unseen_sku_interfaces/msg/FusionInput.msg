# Coordinator-approved observation pair.
# Both sensor messages belong to one selected perception transaction.

# Unique transaction identifier assigned by the coordinator.
string scene_id

# Selected Helios2 point cloud.
sensor_msgs/PointCloud2 cloud

# Selected SAM2 segmentation masks.
ObjectMaskArray masks