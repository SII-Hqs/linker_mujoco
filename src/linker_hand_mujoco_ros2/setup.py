import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'linker_hand_mujoco_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml', 'finger_l20_vec.pkl', 'L20A_CONTACT_NODE.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='linkerhand',
    maintainer_email='linkerhand@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'linker_hand_mujoco_ros2_node=linker_hand_mujoco_ros2.linker_hand_mujoco_ros2:main',
            'linker_hand_l20a_contact_node=linker_hand_mujoco_ros2.contact.linker_hand_l20a_contact_node:main',
            'replay_l20a_contact_cmd=linker_hand_mujoco_ros2.replay_l20a_contact_cmd:main',
            'calculate_l20a_dynamics=linker_hand_mujoco_ros2.calculate_l20a_dynamics:main',
        ],
    },
)
