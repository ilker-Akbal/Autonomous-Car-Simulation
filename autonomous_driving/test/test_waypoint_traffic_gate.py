import json
import math
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

try:
    import rclpy  # noqa: F401
except ModuleNotFoundError:
    rclpy_module = types.ModuleType("rclpy")
    rclpy_node_module = types.ModuleType("rclpy.node")
    rclpy_node_module.Node = type("Node", (), {})
    rclpy_module.node = rclpy_node_module
    sys.modules["rclpy"] = rclpy_module
    sys.modules["rclpy.node"] = rclpy_node_module

try:
    from std_msgs.msg import String  # noqa: F401
except ModuleNotFoundError:
    std_msgs_module = types.ModuleType("std_msgs")
    std_msgs_msg_module = types.ModuleType("std_msgs.msg")

    class String:
        def __init__(self):
            self.data = ""

    std_msgs_msg_module.String = String
    std_msgs_module.msg = std_msgs_msg_module
    sys.modules["std_msgs"] = std_msgs_module
    sys.modules["std_msgs.msg"] = std_msgs_msg_module

from teknofest_control.carla_control_adapter_node import CarlaControlAdapterNode  # noqa: E402
from teknofest_control.control_node import ControlNode  # noqa: E402
from teknofest_planning.traffic_light_manager_node import (  # noqa: E402
    TL_MODE_GREEN_PASS,
    TL_MODE_RED_STOP,
    TRAFFIC_LIGHT_STOP_WAYPOINTS,
    WaypointTrafficGate,
)


class _Publisher:
    def __init__(self):
        self.payload = None

    def publish(self, message):
        self.payload = json.loads(message.data)


class _TrafficLightState:
    def __init__(self, name):
        self.name = name


class _EgoVehicle:
    is_alive = True

    def __init__(self, state):
        self.state = state

    def get_traffic_light_state(self):
        return self.state

    def get_traffic_light(self):
        return None


class _Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Transform:
    def __init__(self, location):
        self.location = location


class _StopWaypoint:
    def __init__(self, x, y):
        self.transform = _Transform(_Location(x, y))


class _TrafficLightActor:
    def __init__(self, actor_id, state, stop_x, stop_y):
        self.id = actor_id
        self.state = _TrafficLightState(state)
        self.stop_waypoint = _StopWaypoint(stop_x, stop_y)

    def get_stop_waypoints(self):
        return [self.stop_waypoint]

    def get_state(self):
        return self.state


class _ActorCollection(list):
    def filter(self, _pattern):
        return self


class _World:
    def __init__(self, actors):
        self.actors = _ActorCollection(actors)

    def get_actors(self):
        return self.actors

    def get_actor(self, actor_id):
        return next((actor for actor in self.actors if actor.id == actor_id), None)


class _DebugDrawer:
    def __init__(self):
        self.points = []
        self.strings = []

    def draw_point(self, location, **kwargs):
        self.points.append({"x": location.x, "y": location.y, "z": location.z, **kwargs})

    def draw_string(self, location, text, **kwargs):
        self.strings.append({"x": location.x, "y": location.y, "z": location.z, "text": text, **kwargs})


class _WorldWithDebug(_World):
    def __init__(self, actors):
        super().__init__(actors)
        self.debug = _DebugDrawer()


class _CarlaLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return _CarlaLocation(self.x + other.x, self.y + other.y, self.z + other.z)


class _VehicleControl:
    def __init__(self, throttle, steer, brake, hand_brake, reverse):
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse


def make_gate(detector_color="unknown", carla_color="unknown", waypoint_color="unknown"):
    gate = WaypointTrafficGate.__new__(WaypointTrafficGate)
    parameters = {
        "front_bumper_offset_m": 0.0,
        "waypoint_candidate_lateral_m": 3.0,
        "waypoint_far_pass_distance_m": 1.5,
        "red_approach_distance_m": 30.0,
        "red_commit_min_distance_m": 8.0,
        "red_hold_distance_m": 1.0,
        "red_comfort_decel_mps2": 2.0,
        "red_stop_buffer_m": 1.5,
        "waypoint_light_match_max_distance_m": 40.0,
        "traffic_light_max_age_s": 0.50,
        "traffic_light_min_confidence": 0.50,
        "ego_role_name": "ego_vehicle",
        "debug_draw_waypoints": True,
        "debug_draw_life_time_s": 0.20,
        "ros_log_period_s": 1.0,
    }
    gate.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    gate.plan_pub = _Publisher()
    gate.log_runtime = lambda _payload: None
    gate.get_logger = lambda: SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
    gate.lane_plan_payload = {"target_speed_mps": 6.0, "stop_request": False, "reason": "cruise"}
    gate.status_payload = {
        "location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"yaw": -90.0},
        "speed_mps": 4.0,
    }
    gate.traffic_light_payload = {
        "selected_detection": {
            "tl_color_filtered": detector_color,
            "tl_confidence": 1.0,
            "tl_detected": detector_color in {"red", "green"},
        },
    }
    gate.last_traffic_light_s = time.time()
    gate.active_waypoint_id = None
    gate.hold_waypoint_id = None
    gate.active_red_stop_target_id = None
    gate.active_red_stop_target_x = None
    gate.active_red_stop_target_y = None
    gate.active_red_stop_target_yaw = None
    gate.waypoint_light_actor_ids = {}
    gate.carla_world = None
    gate.carla_connect_attempted = True
    gate.carla = SimpleNamespace(
        Location=_CarlaLocation,
        Color=lambda r, g, b: SimpleNamespace(r=r, g=g, b=b),
    )
    gate.ego_vehicle = _EgoVehicle(_TrafficLightState(carla_color.title())) if carla_color != "unknown" else None
    gate.last_ego_lookup_s = 0.0
    gate.last_ros_log_s = 0.0
    gate._test_waypoint_color = waypoint_color
    gate.waypoint_traffic_light_color = lambda _id, _wp: (
        gate._test_waypoint_color,
        101 if gate._test_waypoint_color in {"red", "green"} else None,
        0.5 if gate._test_waypoint_color in {"red", "green"} else None,
    )
    return gate


def make_adapter():
    adapter = CarlaControlAdapterNode.__new__(CarlaControlAdapterNode)
    adapter.command_timeout_s = 3.0
    adapter.command_hold_s = 0.7
    adapter.emergency_brake = 0.75
    adapter.clamp = lambda value, low, high: max(low, min(high, value))
    adapter.carla = SimpleNamespace(VehicleControl=_VehicleControl)
    return adapter


def set_waypoint_color(gate, color):
    gate._test_waypoint_color = color


def set_waypoint_signed_distance(gate, waypoint_id, distance_m, lateral_m=0.0):
    waypoint = next(item for item in TRAFFIC_LIGHT_STOP_WAYPOINTS if item["waypoint_id"] == waypoint_id)
    yaw = math.radians(waypoint["yaw_deg"])
    direction_x = math.cos(yaw)
    direction_y = math.sin(yaw)
    perpendicular_x = -direction_y
    perpendicular_y = direction_x
    gate.status_payload["location"]["x"] = waypoint["x"] - direction_x * distance_m + perpendicular_x * lateral_m
    gate.status_payload["location"]["y"] = waypoint["y"] - direction_y * distance_m + perpendicular_y * lateral_m


def tick_payload(gate):
    gate.tick()
    return gate.plan_pub.payload


def test_selects_tl2_after_tl1_is_passed():
    gate = make_gate()
    set_waypoint_signed_distance(gate, "tl_1_stop", -3.5)
    payload = tick_payload(gate)
    assert payload["nearest_stop_waypoint_id"] == "tl_2_stop"
    assert payload["nearest_stop_distance_m"] > 0.0


def test_carla_green_has_priority_and_clears_red_lock():
    gate = make_gate(detector_color="red", waypoint_color="red", carla_color="red")
    set_waypoint_signed_distance(gate, "tl_2_stop", 10.0)
    first = tick_payload(gate)
    assert first["tl_mode"] == TL_MODE_RED_STOP

    gate.ego_vehicle = _EgoVehicle(_TrafficLightState("Green"))
    payload = tick_payload(gate)
    assert payload["tl_mode"] == TL_MODE_GREEN_PASS
    assert payload["final_color"] == "green"
    assert payload["active_red_stop_target_id"] is None
    assert payload["tl_stop_target_id"] is None
    assert payload["final_reason"] == "green_manual_tl_pass"


def test_red_remaining_10m_keeps_target_without_stop():
    gate = make_gate(detector_color="red")
    set_waypoint_signed_distance(gate, "tl_2_stop", 10.0)
    payload = tick_payload(gate)
    assert payload["tl_mode"] == TL_MODE_RED_STOP
    assert payload["tl_stop_target_id"] == "tl_2_stop"
    assert payload["final_stop"] is False
    assert payload["final_reason"] == "red_stop_at_manual_tl_point"


def test_green_clears_previous_active_red_target_immediately():
    gate = make_gate(detector_color="red")
    set_waypoint_signed_distance(gate, "tl_2_stop", 10.0)
    tick_payload(gate)
    gate.ego_vehicle = _EgoVehicle(_TrafficLightState("Green"))
    payload = tick_payload(gate)
    assert payload["tl_mode"] == TL_MODE_GREEN_PASS
    assert payload["active_red_stop_target_id"] is None
    assert payload["tl_stop_target_id"] is None
    assert payload["final_reason"] == "green_manual_tl_pass"


def test_green_at_stop_point_releases_without_confirm_wait():
    gate = make_gate(detector_color="red")
    set_waypoint_signed_distance(gate, "tl_2_stop", 0.3)
    tick_payload(gate)
    gate.ego_vehicle = _EgoVehicle(_TrafficLightState("Green"))
    payload = tick_payload(gate)
    assert payload["tl_mode"] == TL_MODE_GREEN_PASS
    assert payload["tl_stop_target_id"] is None
    assert payload["active_red_stop_target_id"] is None
    assert "green_ignore_reason" not in payload
    assert "red_stop_confirmed_ticks" not in payload


def test_next_stop_marker_only_reports_selected_waypoint():
    gate = make_gate()
    set_waypoint_signed_distance(gate, "tl_2_stop", 30.0)
    marker = gate.next_stop_marker()
    assert marker["waypoint_id"] == "tl_2_stop"
    assert marker["label"] == "NEXT_TL_STOP=tl_2_stop"


def test_draw_waypoint_markers_draws_only_next_stop_label_once():
    gate = make_gate()
    set_waypoint_signed_distance(gate, "tl_2_stop", 30.0)
    world = _WorldWithDebug([])
    gate.carla_world = world
    gate.ensure_carla_world = lambda: world
    gate.draw_waypoint_markers()
    assert len(world.debug.points) == 1
    assert len(world.debug.strings) == 1
    assert world.debug.strings[0]["text"] == "NEXT_TL_STOP=tl_2_stop"


def test_manual_tl_candidate_rejects_too_far_lock():
    valid, reason = WaypointTrafficGate.candidate_validity(
        waypoint_id="tl_2_stop",
        signed_distance=110.0,
        lateral_distance=0.8,
        heading_error_deg=5.0,
        junction_offroute_safety_stop=False,
        safety_reason="",
    )
    assert valid is False
    assert reason == "too_far_for_manual_tl_lock"


def test_control_red_remaining_10m_is_approach():
    profile = ControlNode.manual_tl_stop_profile(
        remaining_to_stop_point=10.0,
        raw_target_speed=6.0,
        current_speed=4.0,
        dt=0.1,
    )
    assert profile["reason"] == "red_stop_at_manual_tl_point"
    assert profile["stop_request"] is False
    assert profile["target_speed"] <= 4.0


def test_control_red_remaining_058m_requests_full_stop():
    profile = ControlNode.manual_tl_stop_profile(
        remaining_to_stop_point=0.58,
        raw_target_speed=6.0,
        current_speed=0.4,
        dt=0.1,
    )
    assert profile["reason"] == "red_manual_tl_reached_stop_point"
    assert profile["stop_request"] is True
    assert profile["brake"] == 1.0
    assert profile["target_speed"] == 0.0


def test_control_green_launch_boost_keeps_minimum_throttle():
    active, boost, reason = ControlNode.green_release_launch_boost(
        now=1.0,
        boost_until_s=0.0,
        reason="green_manual_tl_pass",
        green_release_brake_cleared=True,
        current_speed=0.05,
        target_speed=5.2,
    )
    assert active is True
    assert boost >= 0.35
    assert reason == "force_launch_after_green"


def test_control_green_launch_boost_turns_off_at_higher_speed():
    active, boost, reason = ControlNode.green_release_launch_boost(
        now=1.0,
        boost_until_s=0.0,
        reason="green_manual_tl_pass",
        green_release_brake_cleared=True,
        current_speed=2.1,
        target_speed=5.2,
    )
    assert active is False
    assert boost == 0.0
    assert reason == ""


def test_adapter_red_guard_forces_full_brake_at_065():
    adapter = make_adapter()
    payload = {
        "throttle": 0.2,
        "brake": 0.0,
        "steer": 0.1,
        "reason": "red_stop_at_manual_tl_point",
        "tl_mode": TL_MODE_RED_STOP,
        "remaining_to_stop_point_m": 0.58,
        "stop_request": False,
        "current_speed_mps": 0.4,
    }
    control, reason = adapter.build_vehicle_control(payload, command_age_s=0.1)
    assert reason == "red_manual_tl_reached_stop_point_adapter_guard"
    assert control.throttle == 0.0
    assert control.brake == 1.0
    assert control.reverse is False


def test_adapter_green_pass_clears_brake_and_boosts_launch():
    adapter = make_adapter()
    payload = {
        "throttle": 0.05,
        "brake": 0.4,
        "steer": 0.0,
        "reason": "green_manual_tl_pass",
        "tl_mode": TL_MODE_GREEN_PASS,
        "current_speed_mps": 0.4,
        "target_speed_mps": 5.2,
    }
    control, reason = adapter.build_vehicle_control(payload, command_age_s=0.1)
    assert reason == "green_manual_tl_pass"
    assert control.brake == 0.0
    assert control.throttle >= 0.35
    assert control.reverse is False


def test_waypoint_light_actor_is_matched_by_stop_waypoint_and_cached():
    gate = make_gate()
    del gate.waypoint_traffic_light_color
    tl2 = next(item for item in TRAFFIC_LIGHT_STOP_WAYPOINTS if item["waypoint_id"] == "tl_2_stop")
    near = _TrafficLightActor(42, "Red", tl2["x"] + 0.4, tl2["y"] - 0.2)
    far = _TrafficLightActor(99, "Green", tl2["x"] + 25.0, tl2["y"] + 25.0)
    gate.carla_world = _World([far, near])
    gate.waypoint_light_actor_ids = {}
    color, actor_id, distance = gate.waypoint_traffic_light_color("tl_2_stop", tl2)
    assert color == "red"
    assert actor_id == 42
    assert distance < 1.0


def test_runtime_source_has_no_removed_confirm_strings():
    runtime_files = [
        Path("autonomous_driving/src/teknofest_planning/traffic_light_manager_node.py"),
        Path("autonomous_driving/src/teknofest_control/control_node.py"),
        Path("autonomous_driving/src/teknofest_control/carla_control_adapter_node.py"),
    ]
    banned = [
        "stop_not_confirmed",
        "red_stop_confirmed_ticks",
        "red_hold_after_stop_confirmed",
        "green_after_stop_confirmed",
    ]
    for file_path in runtime_files:
        text = file_path.read_text()
        for item in banned:
            assert item not in text
