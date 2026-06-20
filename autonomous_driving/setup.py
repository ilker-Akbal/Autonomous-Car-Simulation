from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'autonomous_driving'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'missions'), glob('missions/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ilker',
    maintainer_email='ilker@example.com',
    description='Minimal CARLA ROS2 sensor skeleton for TEKNOFEST simulation.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carla_world_manager_node = ros2_nodes.carla_world_manager_node:main',
            'carla_sensor_bridge_node = ros2_nodes.carla_sensor_bridge_node:main',
            'carla_spectator_follow_node = teknofest_sim.carla_spectator_follow_node:main',
            'teknofest_spectator_follow_node = teknofest_sim.teknofest_spectator_follow_node:main',
            'viewport_camera_follow_node = teknofest_sim.viewport_camera_follow_node:main',
            'teknofest_diagnostics_node = teknofest_common.diagnostics_node:main',
            'lane_detector_node = teknofest_perception.lane_detector_node:main',
            'ekf_localizer_node = teknofest_localization.ekf_localizer_node:main',
            'simple_route_planner_node = teknofest_planning.simple_route_planner_node:main',
            'lane_follower_node = teknofest_planning.lane_follower_node:main',
            'control_node = teknofest_control.control_node:main',
            'carla_control_adapter_node = ros2_nodes.carla_control_adapter_node:main',
        ],
    },
)
