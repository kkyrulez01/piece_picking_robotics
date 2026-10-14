# Result of one coordinator-controlled Mask Fusion request.

# Coordinator transaction identifier.
string scene_id

# SAM2 scene index associated with the input.
uint32 scene_index

# True if fusion processing completed successfully.
bool success

# Human-readable result or failure information.
string detail

# Fused observations; populated on successful processing.
FusedObjectArray fused_objects