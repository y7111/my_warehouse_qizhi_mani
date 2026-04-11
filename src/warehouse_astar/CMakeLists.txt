cmake_minimum_required(VERSION 3.0.2)
project(warehouse_astar)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  move_base_msgs
  nav_msgs
  rospy
  std_msgs
  tf
  message_generation
)

add_service_files(
  FILES
  RequestTask.srv
  AStarPlan.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib
    geometry_msgs
    move_base_msgs
    nav_msgs
    rospy
    std_msgs
    tf
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/astar_planner.py
  scripts/warehouse_manager.py
  scripts/robot_agent.py
  scripts/init_robot_poses.py
  scripts/test_nav.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
