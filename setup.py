import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'trainit_setup_assistant'

setup(
    name=package_name,
    version='4.3.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'examples'), glob('examples/*.yaml')),
    ],
    # Jinja2 templates + default configs travel WITH the python package so the
    # generator can find them at runtime relative to __file__ (and so a non-symlink
    # install copies them into site-packages). Patterns are explicit per dir because
    # setuptools package_data does not reliably expand recursive '**'.
    package_data={
        package_name: [
            'generator/templates/description/*.j2',
            'generator/templates/moveit_config/*.j2',
            'generator/templates/app/*.j2',
            'config/*.yaml',
            'resources/isaac/*.py',
            'resources/isaac_cell_kit/*',
            'resources/isaac_cell_kit/scripts/*',
        ],
    },
    install_requires=['setuptools', 'pydantic>=2', 'jinja2', 'pyyaml'],
    zip_safe=True,
    maintainer='fra',
    maintainer_email='ros.master.ai@gmail.com',
    description='TrainIt Setup Assistant — GUI + headless generator for TrainIt '
                'Motion Runtime application bundles.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # headless: project.yaml -> generated bundle (the GUI-independent contract)
            'trainit_generate = trainit_setup_assistant.cli.generate:main',
            # equivalence/build verification of a generated bundle vs a golden one
            'trainit_verify = trainit_setup_assistant.cli.verify:main',
            # compute robot.disable_collisions via collisions_updater (needs ROS)
            'trainit_compute_collisions = trainit_setup_assistant.cli.compute_collisions:main',
            # the RViz-native GUI wizard (added in M6)
            'trainit_setup_assistant = trainit_setup_assistant.cli.gui:main',
            # install the reusable Isaac-cell tooling into a <cell>_isaac package
            'trainit_isaac_kit = trainit_setup_assistant.cli.isaac_kit:main',
        ],
    },
)
