#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

# CARLA egg otomatik bulma
home = str(Path.home())
candidate_eggs = []
candidate_eggs += glob.glob(home + "/simulators/CARLA_0.9.15*/PythonAPI/carla/dist/carla-*3.10*.egg")
candidate_eggs += glob.glob(home + "/carla*/PythonAPI/carla/dist/carla-*3.10*.egg")
candidate_eggs += glob.glob("/opt/carla-simulator/PythonAPI/carla/dist/carla-*3.10*.egg")
candidate_eggs += glob.glob("/opt/carla*/PythonAPI/carla/dist/carla-*3.10*.egg")

for egg in candidate_eggs:
    if egg not in sys.path:
        sys.path.append(egg)

try:
    import carla
except Exception as e:
    print("CARLA Python API import edilemedi.")
    print("Hata:", e)
    print("CARLA egg bulunamadıysa scriptteki candidate_eggs yollarını kontrol et.")
    sys.exit(1)

try:
    import pygame
except Exception as e:
    print("pygame import edilemedi.")
    print("Kurmak için:")
    print("  /usr/bin/python3 -m pip install --user pygame")
    print("veya:")
    print("  pip3 install --user pygame")
    print("Hata:", e)
    sys.exit(1)


TASKS = [
    ("gorev_1", "pickup", "GÖREV 1 / yolcu alma"),
    ("gorev_2", "dropoff", "GÖREV 2 / yolcu indirme"),
    ("park_giris", "park_entry", "GÖREV 3 / park giriş"),
]


def yaw_to_forward_right(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    forward = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
    right = carla.Vector3D(math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0), 0.0)
    return forward, right


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def transform_to_dict(name, kind, label, transform, mode, road_id=None, lane_id=None, is_junction=None):
    loc = transform.location
    rot = transform.rotation
    return {
        "name": name,
        "kind": kind,
        "label": label,
        "mode": mode,
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "yaw": float(rot.yaw),
        "pitch": float(rot.pitch),
        "roll": float(rot.roll),
        "road_id": road_id,
        "lane_id": lane_id,
        "is_junction": is_junction,
    }


def save_points(output_path: Path, points):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Captured TEKNOFEST mission points from CARLA spectator",
        "points": points,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_point_block(points):
    print("\n================ KAYDEDİLEN NOKTALAR ================\n")
    for p in points:
        print(f"{p['label']}:")
        print(f"x = {p['x']}")
        print(f"y = {p['y']}")
        print(f"z = {p['z']}")
        print(f"yaw = {p['yaw']}")
        print(f"kind = {p['kind']}")
        print(f"road_id = {p.get('road_id')}")
        print(f"lane_id = {p.get('lane_id')}")
        print(f"is_junction = {p.get('is_junction')}")
        print()
    print("=====================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--output",
        default="autonomous_driving/config/captured_mission_points.json",
        help="Kaydedilecek JSON dosyası",
    )
    parser.add_argument(
        "--snap-to-road",
        action="store_true",
        default=True,
        help="E basınca en yakın driving lane waypoint kaydedilir. Varsayılan açık.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    spectator = world.get_spectator()

    print("\nCARLA bağlantısı tamam.")
    print("Map:", carla_map.name)
    print("\nKONTROLLER:")
    print("  W/S       ileri / geri")
    print("  A/D       sol / sağ")
    print("  R/F       yukarı / aşağı")
    print("  Oklar     kamera yaw/pitch")
    print("  SHIFT     hızlı hareket")
    print("  CTRL      yavaş hareket")
    print("  E         sıradaki görev noktasını kaydet")
    print("  T         RAW kamera koordinatını kaydet")
    print("  BACKSPACE son kaydı sil")
    print("  ESC       çık")
    print("\nÖneri: Görev noktası için E kullan. E, en yakın yol/şerit waypoint'ine snap eder.")
    print("Park girişi yol üstündeyse yine E kullan.")
    print("-----------------------------------------------------\n")

    pygame.init()
    screen = pygame.display.set_mode((720, 220))
    pygame.display.set_caption("TEKNOFEST Mission Point Capture - focus here, press E to save")
    font = pygame.font.SysFont("monospace", 18)

    clock = pygame.time.Clock()
    points = []
    running = True

    last_save_time = 0.0

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_BACKSPACE:
                    if points:
                        removed = points.pop()
                        save_points(output_path, points)
                        print("Son kayıt silindi:", removed["name"])
                        print_point_block(points)

                elif event.key in (pygame.K_e, pygame.K_t):
                    now = time.time()
                    if now - last_save_time < 0.4:
                        continue
                    last_save_time = now

                    if len(points) >= len(TASKS):
                        print("3 nokta zaten kaydedildi. BACKSPACE ile silip tekrar kaydedebilirsin.")
                        continue

                    name, kind, label = TASKS[len(points)]
                    spec_tf = spectator.get_transform()

                    if event.key == pygame.K_e:
                        # En yakın yol waypoint'ine snap et.
                        wp = carla_map.get_waypoint(
                            spec_tf.location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if wp is None:
                            print("Waypoint bulunamadı. Kamerayı yola yaklaştırıp tekrar E bas.")
                            continue
                        tf = wp.transform
                        point = transform_to_dict(
                            name=name,
                            kind=kind,
                            label=label,
                            transform=tf,
                            mode="nearest_driving_waypoint",
                            road_id=int(wp.road_id),
                            lane_id=int(wp.lane_id),
                            is_junction=bool(wp.is_junction),
                        )
                    else:
                        # RAW spectator koordinatı.
                        point = transform_to_dict(
                            name=name,
                            kind=kind,
                            label=label,
                            transform=spec_tf,
                            mode="raw_spectator",
                            road_id=None,
                            lane_id=None,
                            is_junction=None,
                        )

                    points.append(point)
                    save_points(output_path, points)

                    print(f"\nKAYDEDİLDİ: {label}")
                    print(f"mode = {point['mode']}")
                    print(f"x = {point['x']}")
                    print(f"y = {point['y']}")
                    print(f"z = {point['z']}")
                    print(f"yaw = {point['yaw']}")
                    print(f"road_id = {point.get('road_id')}, lane_id = {point.get('lane_id')}, junction = {point.get('is_junction')}")
                    print(f"Dosya: {output_path}")
                    print_point_block(points)

        keys = pygame.key.get_pressed()
        tf = spectator.get_transform()
        loc = tf.location
        rot = tf.rotation

        speed = 20.0
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed = 60.0
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            speed = 5.0

        forward, right = yaw_to_forward_right(rot.yaw)
        move = carla.Vector3D(0.0, 0.0, 0.0)

        if keys[pygame.K_w]:
            move += forward * speed * dt
        if keys[pygame.K_s]:
            move -= forward * speed * dt
        if keys[pygame.K_d]:
            move += right * speed * dt
        if keys[pygame.K_a]:
            move -= right * speed * dt
        if keys[pygame.K_r]:
            move.z += speed * dt
        if keys[pygame.K_f]:
            move.z -= speed * dt

        loc.x += move.x
        loc.y += move.y
        loc.z += move.z

        yaw_speed = 75.0
        pitch_speed = 55.0
        if keys[pygame.K_LEFT]:
            rot.yaw -= yaw_speed * dt
        if keys[pygame.K_RIGHT]:
            rot.yaw += yaw_speed * dt
        if keys[pygame.K_UP]:
            rot.pitch = clamp(rot.pitch + pitch_speed * dt, -89.0, 89.0)
        if keys[pygame.K_DOWN]:
            rot.pitch = clamp(rot.pitch - pitch_speed * dt, -89.0, 89.0)

        spectator.set_transform(carla.Transform(loc, rot))

        screen.fill((20, 20, 20))
        lines = [
            f"Map: {carla_map.name}",
            f"Kayit: {len(points)}/3    Siradaki: {TASKS[len(points)][2] if len(points) < len(TASKS) else 'TAMAM'}",
            f"Spectator x={loc.x:.3f} y={loc.y:.3f} z={loc.z:.3f} yaw={rot.yaw:.2f} pitch={rot.pitch:.2f}",
            "WASD hareket | R/F yukari/asagi | Oklar bakis | E waypoint kaydet | T raw kaydet | ESC cik",
            f"Output: {output_path}",
        ]
        y = 12
        for line in lines:
            surf = font.render(line, True, (230, 230, 230))
            screen.blit(surf, (12, y))
            y += 34

        pygame.display.flip()

    pygame.quit()
    save_points(output_path, points)
    print_point_block(points)
    print("Çıkıldı. Dosya:", output_path)


if __name__ == "__main__":
    main()
