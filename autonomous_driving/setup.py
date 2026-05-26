from setuptools import find_packages, setup

package_name = "autonomous_driving"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/autonomous_driving"]),
        ("share/autonomous_driving", ["package.xml"]),
        (
            "share/autonomous_driving/launch",
            [
                "launch/teknofest_carla_full.launch.py",
                "launch/teknofest_sign_static_test.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ilker",
    maintainer_email="ilker.akbal4822@gop.edu.tr",
    description="TEKNOFEST Robotaksi CARLA simulation, perception, mission, decision and control package",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = ros2_nodes.camera_node:main",
            "lidar_node = ros2_nodes.lidar_node:main",
            "perception_node = ros2_nodes.perception_node:main",
            "decision_node = ros2_nodes.decision_node:main",
            "control_node = ros2_nodes.control_node:main",

            "carla_world_manager_node = ros2_nodes.carla_world_manager_node:main",
            "carla_spectator_follow_node = teknofest_sim.carla_spectator_follow_node:main",
            "carla_sensor_bridge_node = ros2_nodes.carla_sensor_bridge_node:main",
            "carla_control_adapter_node = ros2_nodes.carla_control_adapter_node:main",
            "carla_scenario_manager_node = ros2_nodes.carla_scenario_manager_node:main",
            "carla_logger_node = ros2_nodes.carla_logger_node:main",

            "teknofest_mission_node = teknofest_sim.teknofest_mission_node:main",
            "teknofest_route_agent_node = teknofest_sim.teknofest_route_agent_node:main",
            "global_route_planner_node = teknofest_sim.global_route_planner_node:main",
            "lane_assist_node = teknofest_sim.lane_assist_node:main",
            "teknofest_scenario_node = teknofest_sim.teknofest_scenario_node:main",
            "teknofest_evaluator_node = teknofest_sim.teknofest_evaluator_node:main",
            "teknofest_sign_static_test_node = teknofest_sim.teknofest_sign_static_test_node:main",
            "teknofest_sign_overlay_node = teknofest_sim.teknofest_sign_overlay_node:main",
            "teknofest_spectator_follow_node = teknofest_sim.teknofest_spectator_follow_node:main",
            "teknofest_route_signs_node = teknofest_sim.teknofest_route_signs_node:main",
        ],
    },
)
