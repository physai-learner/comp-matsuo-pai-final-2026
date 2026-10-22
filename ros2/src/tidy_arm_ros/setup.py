from glob import glob

from setuptools import setup

package_name = "tidy_arm_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "sim_node = tidy_arm_ros.sim_node:main",
            "planner_node = tidy_arm_ros.planner_node:main",
            "tidy = tidy_arm_ros.commander:main",
            "evaluate = tidy_arm_ros.evaluator:main",
        ],
    },
)
