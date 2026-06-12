cmake_minimum_required(VERSION 3.0.2)
project(warehouse_qmix)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  move_base_msgs
  actionlib
  tf
  wpb_mani_behaviors
  gazebo_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gen_map.py
  scripts/test_single.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
