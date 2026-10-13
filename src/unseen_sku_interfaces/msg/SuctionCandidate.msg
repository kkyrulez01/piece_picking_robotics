# All positions and normals are expressed in SuctionCandidateArray.header.frame_id.
# This message is a candidate, NOT an independently collision-checked robot pose.
uint32 object_id
uint32 rank
geometry_msgs/Point position
geometry_msgs/Vector3 normal
int32 pixel_u
int32 pixel_v
int64 source_index
float32 sim_score
float32 sim_score_normalized
float32 boundary_distance_px
float32 boundary_distance_normalized
float32 boundary_quality
float32 surface_variation
float32 surface_quality
float32 footprint_support_ratio
int32 footprint_point_count
float32 footprint_height_p90_m
float32 footprint_quality
bool clearance_valid
int32 clearance_collision_count
float32 clearance_minimum_margin_m
float32 ranking_score
