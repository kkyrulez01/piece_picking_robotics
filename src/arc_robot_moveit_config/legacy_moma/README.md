# Archived MOMA-specific MoveIt configuration

These files were relocated from the active `launch/` and `config/` folders
without rewriting their contents. They are retained for historical reference
only; they are **not** a supported Unseen-SKU launch or hardware configuration.

| Archived file | Previous active location | Why archived |
| --- | --- | --- |
| `moma_moveit.launch.py` | `launch/` | Launched the old `moma_motion_planning` stack and MOMA joint-state remapping. |
| `pick_motion_planner.yaml` | `config/` | Parameters and topics for the obsolete MOMA pick planner. |
| `moma_moveit.rviz` | `config/` | Legacy RViz display configuration. |
| `sensors_real.yaml` | `config/` | Obsolete MechEye/MOMA obstacle-cloud topics. |
| `sensors_3d.yaml` | `config/` | Obsolete Kinect updater; its former name was auto-discovered by the normal MoveItConfigsBuilder defaults. |

The package's `CMakeLists.txt` installs only `launch/`, `config/`, and
`.setup_assistant`; this `legacy_moma/` directory is therefore excluded from
installation by design. The standard generated MoveIt launch files remain
under `launch/`, and all active robot, SRDF, controller, kinematics, Pilz, and
joint-limit configuration remains under `config/` without modification.

**Sensors / collision safety:** This cleanup intentionally leaves no default
3D scene sensor updater. It does not provide an alternative for collision
geometry from the bin, camera, or objects. A validated sensor configuration
and collision scene must be designed before physical execution.

**Physical joint feedback:** This cleanup does not connect the existing joint
convention converter to MoveIt; that is a separate later phase. Do not launch
`moma_moveit.launch.py` for the physical ARC S6.

To restore any historical file for comparison, use Git history or copy it
from this directory into a temporary location; avoid placing
`sensors_3d.yaml` back in the active `config/` directory by default.
