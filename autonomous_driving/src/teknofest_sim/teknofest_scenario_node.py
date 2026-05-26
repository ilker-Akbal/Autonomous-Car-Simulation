import json
import math
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class TeknofestScenarioNode(Node):
    def __init__(self):
        super().__init__("teknofest_scenario_node")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("scenario_round", "round_3")
        self.declare_parameter("traffic_manager_port", 8000)
        self.declare_parameter("destroy_on_shutdown", True)

        self.declare_parameter("npc_vehicle_count", 6)
        self.declare_parameter("walker_count", 4)
        self.declare_parameter("static_obstacle_count", 4)
        self.declare_parameter("dynamic_crossing_enabled", True)

        # Trafik levhası test markerları.
        # camera_view modu: levhaları doğrudan ego kameranın görebileceği yakın alana koyar.
        self.declare_parameter("traffic_sign_markers_enabled", True)
        self.declare_parameter("traffic_sign_marker_count", 8)
        self.declare_parameter("traffic_sign_marker_mode", "camera_view")
        self.declare_parameter("traffic_sign_marker_first_distance", 7.0)
        self.declare_parameter("traffic_sign_marker_distance_step", 3.0)
        self.declare_parameter("traffic_sign_marker_side_offset", 1.3)
        self.declare_parameter("traffic_sign_marker_z", 1.45)
        self.declare_parameter("traffic_sign_marker_debug_draw", True)
        self.declare_parameter("traffic_sign_marker_start_index", 7)
        self.declare_parameter("traffic_sign_marker_stride", 6)
        self.declare_parameter("traffic_sign_marker_yaw_offset", -90.0)
        self.declare_parameter(
            "traffic_sign_marker_blueprints",
            ",".join([
                "static.prop.trafficwarning",
                "static.prop.streetsign04",
                "static.prop.streetsign01",
                "static.prop.streetsign",
                "static.prop.busstoplb",
                "static.prop.busstop",
                "static.prop.trafficcone01",
                "static.prop.trafficcone02",
            ]),
        )

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.scenario_round = self.get_parameter("scenario_round").value
        self.traffic_manager_port = int(self.get_parameter("traffic_manager_port").value)
        self.destroy_on_shutdown = bool(self.get_parameter("destroy_on_shutdown").value)

        self.npc_vehicle_count = int(self.get_parameter("npc_vehicle_count").value)
        self.walker_count = int(self.get_parameter("walker_count").value)
        self.static_obstacle_count = int(self.get_parameter("static_obstacle_count").value)
        self.dynamic_crossing_enabled = bool(self.get_parameter("dynamic_crossing_enabled").value)

        self.traffic_sign_markers_enabled = bool(
            self.get_parameter("traffic_sign_markers_enabled").value
        )
        self.traffic_sign_marker_count = int(
            self.get_parameter("traffic_sign_marker_count").value
        )
        self.traffic_sign_marker_mode = str(
            self.get_parameter("traffic_sign_marker_mode").value
        )
        self.traffic_sign_marker_first_distance = float(
            self.get_parameter("traffic_sign_marker_first_distance").value
        )
        self.traffic_sign_marker_distance_step = float(
            self.get_parameter("traffic_sign_marker_distance_step").value
        )
        self.traffic_sign_marker_side_offset = float(
            self.get_parameter("traffic_sign_marker_side_offset").value
        )
        self.traffic_sign_marker_z = float(
            self.get_parameter("traffic_sign_marker_z").value
        )
        self.traffic_sign_marker_debug_draw = bool(
            self.get_parameter("traffic_sign_marker_debug_draw").value
        )
        self.traffic_sign_marker_blueprints = str(
            self.get_parameter("traffic_sign_marker_blueprints").value
        )
        self.traffic_sign_marker_start_index = int(
            self.get_parameter("traffic_sign_marker_start_index").value
        )
        self.traffic_sign_marker_stride = int(
            self.get_parameter("traffic_sign_marker_stride").value
        )
        self.traffic_sign_marker_yaw_offset = float(
            self.get_parameter("traffic_sign_marker_yaw_offset").value
        )

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.created_actors = []
        self.traffic_sign_marker_actors = []
        self.traffic_sign_marker_infos = []

        self.ego = self.wait_for_ego()
        self.tm = self.client.get_trafficmanager(self.traffic_manager_port)
        self.tm.set_global_distance_to_leading_vehicle(3.0)

        self.status_pub = self.create_publisher(String, "/adas/teknofest/scenario_status", 10)

        self.spawn_all()
        self.timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info("TEKNOFEST scenario node hazır.")

    def wait_for_ego(self):
        for _ in range(100):
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)
        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_world_manager_node çalışmalı.")

    def add_actor(self, actor):
        if actor is not None:
            self.created_actors.append(actor)
        return actor

    def get_route_waypoints_ahead(self, count=120, step_m=4.0):
        ego_wp = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if ego_wp is None:
            return []

        waypoints = [ego_wp]
        current = ego_wp

        for _ in range(count - 1):
            nxt = current.next(step_m)
            if not nxt:
                break
            current = random.choice(nxt)
            waypoints.append(current)

        return waypoints

    def spawn_static_obstacles(self):
        obstacle_blueprints = []

        for pattern in [
            "static.prop.trafficcone*",
            "static.prop.streetbarrier*",
            "static.prop.warningconstruction*",
            "static.prop.constructioncone*",
        ]:
            obstacle_blueprints.extend(list(self.bp_lib.filter(pattern)))

        if not obstacle_blueprints:
            self.get_logger().warning("Statik engel blueprint bulunamadı.")
            return

        waypoints = self.get_route_waypoints_ahead(count=80, step_m=5.0)
        usable = waypoints[8:60:10]

        spawned = 0
        for wp in usable:
            if spawned >= self.static_obstacle_count:
                break

            bp = random.choice(obstacle_blueprints)

            # Şeridin kenarına yakın ama yol içinde engel oluştur.
            right_vec = wp.transform.get_right_vector()
            loc = wp.transform.location + self.carla.Location(
                x=right_vec.x * random.choice([-0.7, 0.7]),
                y=right_vec.y * random.choice([-0.7, 0.7]),
                z=0.15,
            )

            transform = self.carla.Transform(
                loc,
                self.carla.Rotation(
                    pitch=0.0,
                    yaw=wp.transform.rotation.yaw + random.uniform(-15.0, 15.0),
                    roll=0.0,
                ),
            )

            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                self.add_actor(actor)
                spawned += 1

        self.get_logger().info(f"Statik engel spawn edildi: {spawned}")

    def spawn_npc_vehicles(self):
        vehicle_bps = [
            bp for bp in self.bp_lib.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]

        spawn_points = list(self.map.get_spawn_points())
        random.shuffle(spawn_points)

        spawned = 0
        ego_loc = self.ego.get_location()

        for sp in spawn_points:
            if spawned >= self.npc_vehicle_count:
                break

            if sp.location.distance(ego_loc) < 25.0:
                continue

            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "teknofest_npc_vehicle")

            actor = self.world.try_spawn_actor(bp, sp)
            if actor is None:
                continue

            actor.set_autopilot(True, self.traffic_manager_port)
            self.add_actor(actor)
            spawned += 1

        self.get_logger().info(f"NPC araç spawn edildi: {spawned}")

    def spawn_walkers(self):
        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        controller_bp = self.bp_lib.find("controller.ai.walker")

        spawned = 0
        ego_loc = self.ego.get_location()

        for _ in range(self.walker_count * 5):
            if spawned >= self.walker_count:
                break

            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue

            if loc.distance(ego_loc) < 15.0:
                continue

            walker_bp = random.choice(walker_bps)
            walker = self.world.try_spawn_actor(walker_bp, self.carla.Transform(loc))

            if walker is None:
                continue

            controller = self.world.try_spawn_actor(
                controller_bp,
                self.carla.Transform(),
                walker,
            )

            if controller is not None:
                controller.start()
                target = self.world.get_random_location_from_navigation()
                if target is not None:
                    controller.go_to_location(target)
                controller.set_max_speed(random.uniform(0.7, 1.4))
                self.add_actor(controller)

            self.add_actor(walker)
            spawned += 1

        self.get_logger().info(f"Yaya spawn edildi: {spawned}")

    def spawn_dynamic_crossing_obstacle(self):
        if not self.dynamic_crossing_enabled:
            return

        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        controller_bp = self.bp_lib.find("controller.ai.walker")
        waypoints = self.get_route_waypoints_ahead(count=50, step_m=4.0)

        if len(waypoints) < 15:
            return

        target_wp = waypoints[14]
        right_vec = target_wp.transform.get_right_vector()

        start_loc = target_wp.transform.location + self.carla.Location(
            x=right_vec.x * 5.0,
            y=right_vec.y * 5.0,
            z=0.2,
        )

        end_loc = target_wp.transform.location + self.carla.Location(
            x=-right_vec.x * 5.0,
            y=-right_vec.y * 5.0,
            z=0.2,
        )

        walker = self.world.try_spawn_actor(
            random.choice(walker_bps),
            self.carla.Transform(start_loc),
        )

        if walker is None:
            return

        controller = self.world.try_spawn_actor(
            controller_bp,
            self.carla.Transform(),
            walker,
        )

        if controller is not None:
            controller.start()
            controller.go_to_location(end_loc)
            controller.set_max_speed(1.1)
            self.add_actor(controller)

        self.add_actor(walker)
        self.get_logger().info("Dinamik geçiş engeli/yaya spawn edildi.")

    def resolve_blueprints(self, patterns_text):
        patterns = [
            p.strip()
            for p in str(patterns_text).split(",")
            if p.strip()
        ]

        blueprints = []
        seen = set()

        for pattern in patterns:
            matches = list(self.bp_lib.filter(pattern))
            if not matches:
                continue

            for bp in matches:
                if bp.id in seen:
                    continue
                seen.add(bp.id)
                blueprints.append(bp)

        return blueprints

    def spawn_single_sign_marker(self, bp, wp, side_sign, index):
        right_vec = wp.transform.get_right_vector()

        side_offset = self.traffic_sign_marker_side_offset * side_sign
        loc = wp.transform.location + self.carla.Location(
            x=right_vec.x * side_offset,
            y=right_vec.y * side_offset,
            z=self.traffic_sign_marker_z,
        )

        yaw = wp.transform.rotation.yaw + self.traffic_sign_marker_yaw_offset
        if side_sign < 0:
            yaw += 180.0

        transform = self.carla.Transform(
            loc,
            self.carla.Rotation(
                pitch=0.0,
                yaw=yaw,
                roll=0.0,
            ),
        )

        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            return None

        self.add_actor(actor)
        self.traffic_sign_marker_actors.append(actor)

        self.get_logger().info(
            "TRAFFIC_SIGN_MARKER_SPAWNED "
            f"idx={index} type_id={actor.type_id} "
            f"loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f}) yaw={yaw:.1f}"
        )
        return actor

    def resolve_blueprints(self, patterns_text):
        patterns = [
            p.strip()
            for p in str(patterns_text).split(",")
            if p.strip()
        ]

        blueprints = []
        seen = set()

        for pattern in patterns:
            matches = list(self.bp_lib.filter(pattern))
            for bp in matches:
                if bp.id in seen:
                    continue
                seen.add(bp.id)
                blueprints.append(bp)

        return blueprints

    def draw_marker_debug(self, actor, index, loc):
        if not self.traffic_sign_marker_debug_draw:
            return

        try:
            color = self.carla.Color(255, 0, 0)
            self.world.debug.draw_string(
                loc + self.carla.Location(z=2.0),
                f"SIGN#{index}",
                draw_shadow=True,
                color=color,
                life_time=9999.0,
                persistent_lines=True,
            )

            bb = actor.bounding_box
            self.world.debug.draw_box(
                bb,
                actor.get_transform().rotation,
                thickness=0.08,
                color=color,
                life_time=9999.0,
                persistent_lines=True,
            )
        except Exception as exc:
            self.get_logger().warning(f"SIGN_DEBUG_DRAW_ERROR idx={index}: {exc}")

    def spawn_sign_marker_at_transform(self, bp, transform, index, forward_m, side_m):
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            self.get_logger().warning(
                f"TRAFFIC_SIGN_MARKER_FAILED idx={index} bp={bp.id} "
                f"loc=({transform.location.x:.1f},{transform.location.y:.1f},{transform.location.z:.1f})"
            )
            return None

        self.add_actor(actor)
        self.traffic_sign_marker_actors.append(actor)

        info = {
            "idx": index,
            "type_id": actor.type_id,
            "forward_m": round(float(forward_m), 2),
            "side_m": round(float(side_m), 2),
            "x": round(float(transform.location.x), 2),
            "y": round(float(transform.location.y), 2),
            "z": round(float(transform.location.z), 2),
            "yaw": round(float(transform.rotation.yaw), 2),
        }
        self.traffic_sign_marker_infos.append(info)

        self.draw_marker_debug(actor, index, transform.location)

        self.get_logger().info(
            "TRAFFIC_SIGN_MARKER_VISIBLE "
            f"idx={index} type={actor.type_id} "
            f"forward={forward_m:.1f}m side={side_m:.1f}m "
            f"loc=({transform.location.x:.1f},{transform.location.y:.1f},{transform.location.z:.1f}) "
            f"yaw={transform.rotation.yaw:.1f}"
        )

        return actor

    def spawn_sign_markers_camera_view(self, props):
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_rot = ego_tf.rotation

        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        spawned = 0

        for i in range(self.traffic_sign_marker_count):
            bp = props[i % len(props)]

            forward_m = self.traffic_sign_marker_first_distance + (
                i * self.traffic_sign_marker_distance_step
            )

            # Sağ-sol dönüşümlü ama görüş alanında kalacak kadar yakın.
            side_sign = 1.0 if i % 2 == 0 else -1.0
            side_m = self.traffic_sign_marker_side_offset * side_sign

            loc = ego_loc + self.carla.Location(
                x=fwd.x * forward_m + right.x * side_m,
                y=fwd.y * forward_m + right.y * side_m,
                z=self.traffic_sign_marker_z,
            )

            # Levhayı ego aracına doğru çevirmeye çalışıyoruz.
            # Bazı CARLA proplarında ön yüz ekseni farklı olabiliyor; bu yüzden
            # 0/180 dönüşümlü yaw veriyoruz.
            yaw = ego_rot.yaw + 180.0
            if i % 2 == 1:
                yaw = ego_rot.yaw

            transform = self.carla.Transform(
                loc,
                self.carla.Rotation(
                    pitch=0.0,
                    yaw=yaw,
                    roll=0.0,
                ),
            )

            actor = self.spawn_sign_marker_at_transform(
                bp=bp,
                transform=transform,
                index=i,
                forward_m=forward_m,
                side_m=side_m,
            )

            if actor is not None:
                spawned += 1

        return spawned

    def spawn_sign_markers_route_view(self, props):
        waypoints = self.get_route_waypoints_ahead(count=90, step_m=4.0)
        if len(waypoints) < 10:
            self.get_logger().warning(
                f"ROUTE_SIGN_MARKER_FAILED yeterli waypoint yok len={len(waypoints)}"
            )
            return 0

        spawned = 0
        start_idx = 5
        stride = 5

        for i, wp in enumerate(waypoints[start_idx::stride]):
            if spawned >= self.traffic_sign_marker_count:
                break

            bp = props[spawned % len(props)]
            right_vec = wp.transform.get_right_vector()

            side_sign = 1.0 if spawned % 2 == 0 else -1.0
            side_m = self.traffic_sign_marker_side_offset * side_sign

            loc = wp.transform.location + self.carla.Location(
                x=right_vec.x * side_m,
                y=right_vec.y * side_m,
                z=self.traffic_sign_marker_z,
            )

            yaw = wp.transform.rotation.yaw + 180.0
            if spawned % 2 == 1:
                yaw = wp.transform.rotation.yaw

            transform = self.carla.Transform(
                loc,
                self.carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
            )

            actor = self.spawn_sign_marker_at_transform(
                bp=bp,
                transform=transform,
                index=spawned,
                forward_m=(start_idx + i * stride) * 4.0,
                side_m=side_m,
            )

            if actor is not None:
                spawned += 1

        return spawned

    def spawn_sign_markers(self):
        """
        Görünür trafik levhası/işaret markerları spawn eder.

        camera_view modu:
        - Levhaları ego aracın ön görüş alanına koyar.
        - CARLA viewport'ta SIGN# etiketi ve kırmızı debug box gösterir.
        - Perception node aynı objeleri /adas/camera/front/image_raw üzerinden görür.

        route_view modu:
        - Levhaları rota waypointleri boyunca dizer.
        """
        if not self.traffic_sign_markers_enabled:
            self.get_logger().info("Trafik levhası marker spawn kapalı.")
            return

        props = self.resolve_blueprints(self.traffic_sign_marker_blueprints)

        if not props:
            self.get_logger().warning(
                "Trafik levhası/işaret blueprint bulunamadı. "
                f"patterns={self.traffic_sign_marker_blueprints}"
            )
            return

        self.get_logger().info(
            "TRAFFIC_SIGN_MARKER_BLUEPRINTS "
            + ", ".join([bp.id for bp in props])
        )

        if self.traffic_sign_marker_mode == "route_view":
            spawned = self.spawn_sign_markers_route_view(props)
        else:
            spawned = self.spawn_sign_markers_camera_view(props)

        self.get_logger().info(
            "TRAFFIC_SIGN_MARKER_SUMMARY "
            f"mode={self.traffic_sign_marker_mode} "
            f"spawned={spawned}/{self.traffic_sign_marker_count} "
            f"created_actor_count={len(self.created_actors)}"
        )

    def spawn_all(self):
        # PURE MISSION TEST MODE:
        # İlk hedef testinde koni/tabela/yaya/NPC spawn etmiyoruz.
        # Çünkü route testinde trafficcone çarpışması collision_halt'a düşürüyor.
        if self.static_obstacle_count > 0:
            self.spawn_static_obstacles()

        if self.npc_vehicle_count > 0:
            self.spawn_npc_vehicles()

        if self.walker_count > 0:
            self.spawn_walkers()

        if self.dynamic_crossing_enabled:
            self.spawn_dynamic_crossing_obstacle()

        if self.traffic_sign_markers_enabled:
            self.spawn_sign_markers()

    def publish_status(self):
        alive_markers = []
        for info, actor in zip(
            getattr(self, "traffic_sign_marker_infos", []),
            getattr(self, "traffic_sign_marker_actors", []),
        ):
            try:
                alive = bool(actor.is_alive)
            except Exception:
                alive = False

            copied = dict(info)
            copied["alive"] = alive
            alive_markers.append(copied)

        payload = {
            "stamp": round(time.time(), 3),
            "scenario_round": self.scenario_round,
            "created_actor_count": len(self.created_actors),
            "traffic_sign_markers_enabled": self.traffic_sign_markers_enabled,
            "traffic_sign_marker_mode": self.traffic_sign_marker_mode,
            "traffic_sign_marker_count": len(self.traffic_sign_marker_actors),
            "traffic_sign_marker_alive_count": len([m for m in alive_markers if m.get("alive")]),
            "traffic_sign_markers": alive_markers,
            "static_or_prop_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("static.") or a.type_id.startswith("traffic.")
            ]),
            "vehicle_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("vehicle.")
            ]),
            "walker_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("walker.")
            ]),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def destroy_node(self):
        if self.destroy_on_shutdown:
            for actor in reversed(getattr(self, "created_actors", [])):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestScenarioNode()

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