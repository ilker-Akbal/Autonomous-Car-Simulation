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
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ilker",
    maintainer_email="ilker.akbal4822@gop.edu.tr",
    description="TEKNOFEST Robotaksi CARLA simulation and mission package",
    license="MIT",
    entry_points={
        "console_scripts": [
            "carla_world_manager_node = ros2_nodes.carla_world_manager_node:main",
            "carla_sensor_bridge_node = ros2_nodes.carla_sensor_bridge_node:main",
            "teknofest_mission_node = teknofest_sim.teknofest_mission_node:main",
            "clean_lane_vision_node = ros2_nodes.clean_lane_vision_node:main",
            "clean_phase1_driver_node = ros2_nodes.clean_phase1_driver_node:main",
            "clean_carla_control_node = ros2_nodes.clean_carla_control_node:main",
            "carla_spectator_follow_node = teknofest_sim.carla_spectator_follow_node:main",
            "teknofest_spectator_follow_node = teknofest_sim.teknofest_spectator_follow_node:main",
        ],
    },
)
