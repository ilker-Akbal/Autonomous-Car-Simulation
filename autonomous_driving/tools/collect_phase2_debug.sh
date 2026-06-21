#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEBUG_ROOT="${REPO_ROOT}/debug_runs"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${DEBUG_ROOT}/${STAMP}"
ARCHIVE_PATH="${DEBUG_ROOT}/${STAMP}.tar.gz"

mkdir -p "${RUN_DIR}"

source_if_exists() {
  local candidate="$1"
  if [[ -f "${candidate}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${candidate}" >/dev/null 2>&1 || true
    set -u
  fi
}

capture_command() {
  local output_path="$1"
  shift
  if "$@" >"${output_path}" 2>&1; then
    return 0
  fi
  return 0
}

capture_topic_once() {
  local topic_name="$1"
  local output_path="$2"
  if timeout 12s ros2 topic echo "${topic_name}" --once --full-length >"${output_path}" 2>&1; then
    return 0
  fi
  {
    echo "failed_to_capture_topic_once: ${topic_name}"
    echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${output_path}"
  return 0
}

source_if_exists "/opt/ros/humble/setup.bash"
source_if_exists "/opt/ros/iron/setup.bash"
source_if_exists "${REPO_ROOT}/install/setup.bash"

capture_command "${RUN_DIR}/ros2_node_list.txt" ros2 node list
capture_command "${RUN_DIR}/ros2_topic_list.txt" ros2 topic list

capture_topic_once "/adas/carla/status" "${RUN_DIR}/adas_carla_status.txt"
capture_topic_once "/adas/mission/status" "${RUN_DIR}/adas_mission_status.txt"
capture_topic_once "/adas/planning/global_route_debug" "${RUN_DIR}/adas_planning_global_route_debug.txt"
capture_topic_once "/adas/planning/route_debug" "${RUN_DIR}/adas_planning_route_debug.txt"
capture_topic_once "/adas/planning/route_events" "${RUN_DIR}/adas_planning_route_events.txt"
capture_topic_once "/adas/planning/route_events_debug" "${RUN_DIR}/adas_planning_route_events_debug.txt"
capture_topic_once "/adas/planning/lane_plan" "${RUN_DIR}/adas_planning_lane_plan.txt"
capture_topic_once "/adas/planning/lane_debug" "${RUN_DIR}/adas_planning_lane_debug.txt"
capture_topic_once "/adas/control/vehicle_command" "${RUN_DIR}/adas_control_vehicle_command.txt"
capture_topic_once "/adas/control/debug" "${RUN_DIR}/adas_control_debug.txt"
capture_topic_once "/adas/control/adapter_debug" "${RUN_DIR}/adas_control_adapter_debug.txt"

capture_command "${RUN_DIR}/carla_world_snapshot.txt" python3 "${SCRIPT_DIR}/debug_carla_world_snapshot.py"

cat <<EOF >"${RUN_DIR}/README.txt"
phase2_debug_run=${STAMP}
repo_root=${REPO_ROOT}
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

mkdir -p "${DEBUG_ROOT}"
tar -czf "${ARCHIVE_PATH}" -C "${DEBUG_ROOT}" "${STAMP}"

echo "Debug run collected at: ${RUN_DIR}"
echo "Archive created at: ${ARCHIVE_PATH}"
