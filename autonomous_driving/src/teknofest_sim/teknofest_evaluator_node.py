import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TeknofestEvaluatorNode(Node):
    def __init__(self):
        super().__init__("teknofest_evaluator_node")

        self.declare_parameter("log_dir", "outputs/teknofest_sim_logs")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("mission_event_topic", "/adas/teknofest/events")
        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("control_debug_topic", "/adas/teknofest/route_agent_debug")
        self.declare_parameter("collision_topic", "/adas/events/collision")
        self.declare_parameter("scenario_status_topic", "/adas/teknofest/scenario_status")

        self.log_dir = self.get_parameter("log_dir").value
        self.mission_topic = self.get_parameter("mission_topic").value
        self.mission_event_topic = self.get_parameter("mission_event_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value
        self.control_debug_topic = self.get_parameter("control_debug_topic").value
        self.collision_topic = self.get_parameter("collision_topic").value
        self.scenario_status_topic = self.get_parameter("scenario_status_topic").value

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"teknofest_run_{stamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.files = {
            "mission": os.path.join(self.run_dir, "mission.jsonl"),
            "events": os.path.join(self.run_dir, "mission_events.jsonl"),
            "decisions": os.path.join(self.run_dir, "decisions.jsonl"),
            "control": os.path.join(self.run_dir, "control.jsonl"),
            "collisions": os.path.join(self.run_dir, "collisions.jsonl"),
            "scenario": os.path.join(self.run_dir, "scenario.jsonl"),
            "summary": os.path.join(self.run_dir, "summary.json"),
        }

        self.summary = {
            "run_dir": self.run_dir,
            "started_at": time.time(),
            "mission_completed": False,
            "mission_failed": False,
            "collision_count": 0,
            "passenger_stop_started": 0,
            "passenger_stop_completed": 0,
            "park_entry_reached": False,
            "last_stage": None,
            "last_decision": None,
            "last_control": None,
            "events": [],
        }

        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)
        self.create_subscription(String, self.mission_event_topic, self.event_cb, 10)
        self.create_subscription(String, self.decision_topic, self.decision_cb, 10)
        self.create_subscription(String, self.control_debug_topic, self.control_cb, 10)
        self.create_subscription(String, self.collision_topic, self.collision_cb, 10)
        self.create_subscription(String, self.scenario_status_topic, self.scenario_cb, 10)

        self.timer = self.create_timer(2.0, self.write_summary)

        self.get_logger().info(f"TEKNOFEST evaluator/log hazır: {self.run_dir}")

    def parse(self, raw):
        try:
            return json.loads(raw)
        except Exception:
            return {"raw": raw}

    def append(self, key, raw):
        data = self.parse(raw)
        data["_logger_stamp"] = time.time()

        with open(self.files[key], "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return data

    def mission_cb(self, msg):
        data = self.append("mission", msg.data)
        self.summary["last_stage"] = data.get("stage")

        if data.get("completed"):
            self.summary["mission_completed"] = True

        if data.get("stage") == "FAILED":
            self.summary["mission_failed"] = True

    def event_cb(self, msg):
        data = self.append("events", msg.data)
        event_type = data.get("event_type")
        self.summary["events"].append(data)

        if event_type == "passenger_stop_started":
            self.summary["passenger_stop_started"] += 1

        elif event_type == "passenger_stop_completed":
            self.summary["passenger_stop_completed"] += 1

        elif event_type == "park_entry_reached":
            self.summary["park_entry_reached"] = True

        elif event_type == "mission_completed":
            self.summary["mission_completed"] = True

        elif event_type == "park_timeout":
            self.summary["mission_failed"] = True

        self.write_summary()

    def decision_cb(self, msg):
        data = self.append("decisions", msg.data)
        self.summary["last_decision"] = data

    def control_cb(self, msg):
        data = self.append("control", msg.data)
        self.summary["last_control"] = data

    def collision_cb(self, msg):
        data = self.append("collisions", msg.data)
        self.summary["collision_count"] += 1
        self.summary["events"].append({
            "stamp": time.time(),
            "event_type": "collision",
            "data": data,
        })
        self.write_summary()

    def scenario_cb(self, msg):
        self.append("scenario", msg.data)

    def write_summary(self):
        self.summary["updated_at"] = time.time()
        self.summary["elapsed_s"] = round(
            self.summary["updated_at"] - self.summary["started_at"],
            3,
        )

        self.summary["qualification_flags"] = {
            "mission_completed": bool(self.summary["mission_completed"]),
            "no_collision": self.summary["collision_count"] == 0,
            "park_entry_reached": bool(self.summary["park_entry_reached"]),
            "passenger_stops_done": self.summary["passenger_stop_completed"],
        }

        with open(self.files["summary"], "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=2)

    def destroy_node(self):
        self.write_summary()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestEvaluatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()