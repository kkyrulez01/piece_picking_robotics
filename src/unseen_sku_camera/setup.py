from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'unseen_sku_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install YAML configuration files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        # Install calibration_data
        (os.path.join("share", package_name, "calibration_data"), glob("calibration_data/*.npz")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='support',
    maintainer_email='kokyoongkang@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "helios2_node = unseen_sku_camera.helios2_node:main",
        ],
    },
)
