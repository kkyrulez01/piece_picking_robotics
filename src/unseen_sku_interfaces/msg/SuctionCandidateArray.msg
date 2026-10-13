std_msgs/Header header
uint32 scene_index
# OK, NO_CANDIDATES, ERROR
string status
string error_message
# True: the clearance filter had ONLY points assigned to SAM2 object masks.
# True must never be interpreted as safe for automatic robot execution.
bool geometry_is_mask_limited
float32 processing_time_s
SuctionCandidate[] candidates
