# Phase 2 Integration Plan

## Phase 1 imported from Repo 1
- `teknofest_common/diagnostics_node.py` adapted from `teknofest_robotaksi-main/src/robotaksi_control/robotaksi_control/diagnostics.py`
- `teknofest_perception/lane_detector_node.py` adapted from `teknofest_robotaksi-main/src/robotaksi_perception/robotaksi_perception/lane_detector.py`
- `teknofest_localization/ekf_localizer_node.py` adapted from `teknofest_robotaksi-main/src/robotaksi_control/robotaksi_control/ekf_localizer.py`

## Phase 2 planned imports from Repo 2
- `velocity_planner.py` -> `teknofest_planning/velocity_profile.py`
- `collision_checker.py` -> `teknofest_planning/collision_checker.py`
- From `behavioural_planner.py`, port only helper geometry functions:
  - `get_closest_index`
  - `project_on_linestring`
  - `check_is_after`
  - `check_is_before`
  - `get_stop_index`
  - route corridor check logic

## Exclusions and constraints
- No direct control or vehicle command nodes are added in Phase 1.
- `vehicle.apply_control()` remains only for a future, isolated `carla_control_adapter_node`.
- Default `teknofest_carla_full.launch.py` remains unchanged.
- No old CARLA client API code is imported.
- No legacy Docker or launch architecture from external repos is copied wholesale.
- `matplotlib`, `tkinter`, `live_plotter`, and old CARLA API utilities are excluded.

## Future dependency notes
- If geometry utilities require spatial operations, `shapely` should be added to `requirements.txt` and package dependency metadata.
- `psutil` is required for diagnostics and is currently added to `autonomous_driving/requirements.txt`.
