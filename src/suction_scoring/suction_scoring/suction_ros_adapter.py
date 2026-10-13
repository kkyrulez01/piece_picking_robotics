"""ROS messages <-> NumPy/Python data for Sim-Suction.

Keep these conversions out of sim_suction_node.py and out of the numerical
sim_suction_pipeline.py.
"""
import numpy as np
from sensor_msgs.msg import PointField
from unseen_sku_interfaces.msg import (
    SuctionCandidate as SuctionCandidateMsg,
    SuctionCandidateArray,
)

from .fusion_inputs import FusionObjectData, assemble_fused_scene

_EXPECTED = {
    'x': PointField.FLOAT32,
    'y': PointField.FLOAT32,
    'z': PointField.FLOAT32,
    'u': PointField.INT32,
    'v': PointField.INT32,
    'source_index': PointField.UINT32,
}

def read_fused_cloud(message):
    """
    Decode XYZ, UV and source index from the working fusion PointCloud2.
    """
    fields = {field.name: field for field in message.fields}
    # Compare with expected fields
    for name, datatype in _EXPECTED.items():
            if (name not in fields or fields[name].datatype != datatype
                    or fields[name].count != 1):
                raise ValueError(f'Invalid PointCloud2 field {name!r}; expected {datatype}')

    endian = '>' if message.is_bigendian else '<'
    names = ('x', 'y', 'z', 'u', 'v', 'source_index')
    formats = tuple(
        endian + ('f4' if name in ('x', 'y', 'z')
                  else 'u4' if name == 'source_index' else 'i4')
        for name in names
    )
    dtype = np.dtype({
        'names': names,
        'formats': formats,
        'offsets': tuple(fields[name].offset for name in names),
        'itemsize': int(message.point_step),
    })
    data = np.ndarray(
        shape=(int(message.height), int(message.width)),
        dtype=dtype,
        buffer=message.data,
        strides=(int(message.row_step), int(message.point_step)),
    ).reshape(-1)

    xyz = np.column_stack((data['x'], data['y'], data['z'])).astype(np.float32)
    uv = np.column_stack((data['u'], data['v'])).astype(np.int32)
    source = data['source_index'].astype(np.int64)
    return xyz, uv, source

def fused_array_to_scene(message, bridge):
    """
    Convert incoming fused-object ROS message to pure-Python SceneInput. 
    """
    objects = []
    for incoming in message.objects:
        xyz, uv, source = read_fused_cloud(incoming.cloud)
        if len(xyz) == 0:
            continue
        mask = bridge.imgmsg_to_cv2(incoming.mask, desired_encoding='mono8')
        objects.append(FusionObjectData(
            object_id=int(incoming.object_id),
            mask=np.asarray(mask) > 127,
            xyz_m=xyz,
            uv=uv,
            source_indices=source
        ))

    return assemble_fused_scene(objects) if objects else None

def _float_or_nan(candidate, field):
    value = getattr(candidate, field, None)
    return float(value) if value is not None else float('nan')

def _candidate_to_message(candidate, original_id, rank, scene):
    output = SuctionCandidateMsg()
    output.object_id = int(original_id)
    output.rank = int(rank)
    xyz = np.asarray(candidate.xyz_m, dtype=np.float32).reshape(3)
    normal = np.asarray(candidate.normal, dtype=np.float32).reshape(3)
    output.position.x, output.position.y, output.position.z = map(float, xyz)
    output.normal.x, output.normal.y, output.normal.z = map(float, normal)
    uv = np.asarray(candidate.uv, dtype=np.int64).reshape(2)
    output.pixel_u, output.pixel_v = int(uv[0]), int(uv[1])

    # The candidate index refers to the assembled input array
    # Resolve original Helios source index before publishing
    src =  getattr(candidate, 'source_index', None)
    output.source_index = -1
    if src is not None and 0 <= int(src) < len(scene.original_source_indices):
        output.source_index = int(scene.original_source_indices[int(src)])

    for field in (
        'sim_score', 'sim_score_normalized', 'boundary_distance_px',
        'boundary_distance_normalized', 'boundary_quality',
        'surface_variation', 'surface_quality', 'footprint_support_ratio',
        'footprint_height_p90_m', 'footprint_quality',
        'clearance_minimum_margin_m', 'ranking_score',
    ):
        setattr(output, field, _float_or_nan(candidate, field))

    output.footprint_point_count = int(getattr(candidate, 'footprint_point_count', 0))
    output.clearance_valid = bool(getattr(candidate, 'clearance_valid', False))
    output.clearance_collision_count = int(getattr(candidate, 'clearance_collision_count', 0))

    return output

def status_message(incoming, status, error_message=''):
    output = SuctionCandidateArray()
    output.header = incoming.header
    output.scene_index = int(incoming.scene_index)
    output.geometry_is_mask_limited = True
    output.status = str(status)
    output.error_message = str(error_message)
    return output

def result_to_message(incoming, scene, result):
    output = status_message(incoming, 'NO_CANDIDATES')
    output.geometry_is_mask_limited = bool(result.geometry_is_mask_limited)
    for original_id, candidates in result.candidates_by_object.items():
        for rank, candidate in enumerate(candidates, start=1):
            output.candidates.append(
                _candidate_to_message(
                    candidate,
                    original_id,
                    rank,
                    scene
                )
            )

    if output.candidates:
        output.status = 'OK'

    return output