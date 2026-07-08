import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'linker_hand_mujoco_ros2'


def package_files(directory):
    files = []
    for path in glob(os.path.join(directory, '**', '*'), recursive=True):
        if os.path.isfile(path):
            files.append(path)
    return files


def install_tree(directory, install_root=None):
    install_root = install_root or directory
    entries = []
    for path in package_files(directory):
        rel_dir = os.path.relpath(os.path.dirname(path), directory)
        target_dir = os.path.join('share', package_name, install_root)
        if rel_dir != '.':
            target_dir = os.path.join(target_dir, rel_dir)
        entries.append((target_dir, [path]))
    return entries

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml', 'finger_l20_vec.pkl', 'L20A_CONTACT_NODE.md']),
    ]
    + install_tree(os.path.join(package_name, 'urdf'), 'urdf')
    + install_tree(os.path.join(package_name, 'resources'), 'resources')
    + install_tree('retargeting_data'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='linkerhand',
    maintainer_email='linkerhand@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'linker_hand_mujoco_ros2_node=linker_hand_mujoco_ros2.nodes.mujoco_node:main',
            'linker_hand_l20a_contact_node=linker_hand_mujoco_ros2.nodes.l20a_contact_node:main',
            'replay_l20a_contact_cmd=linker_hand_mujoco_ros2.nodes.replay_contact_cmd:main',
            'calculate_l20a_dynamics=linker_hand_mujoco_ros2.tools.calculate_l20a_dynamics:main',
        ],
    },
)
