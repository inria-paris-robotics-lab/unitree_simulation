from setuptools import setup
from glob import glob

package_name = "unitree_simulation"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/data/assets", glob("data/assets/*")),
    ],
    install_requires=["setuptools", "pybullet"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="earlaud@inria.fr",
    description="Basic simulator wrapper to mimic real Go2 ROS2 control API.",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            "simulation_node = unitree_simulation.simulation_node:main",
        ],
    },
)
