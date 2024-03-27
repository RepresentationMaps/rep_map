cmake_minimum_required(VERSION 3.0.2)

project(rep_map)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
set(DISABLE_PAL_FLAGS 1) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  pluginlib
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros
  roscpp
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  geometry_msgs
  rep_plugins
  message_generation
  geometry_msgs
  visualization_msgs
)

find_package(Boost REQUIRED) # for boost::shared_ptr, required by pluginlib plugin loader
find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(yaml-cpp REQUIRED)
pkg_search_module(IlmBase REQUIRED OpenEXR)
include_directories(
    include
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${OpenVDB_INCLUDE_DIR}
    ${IlmBase_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

add_service_files(
  FILES
  AddPlugin.srv
  RemovePlugin.srv
  Query.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs 
)

catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
        geometry_msgs
        pcl_ros
        pcl_conversions
        pluginlib
        roscpp
        sensor_msgs
        std_msgs
        tf2_ros
        tf2_geometry_msgs
        visualization_msgs
    DEPENDS
        TBB
        OpenVDB
        PCL
        Boost
	      IlmBase
        yaml-cpp
)

add_library(${PROJECT_NAME}
            src/vdb2pc.cpp
            src/rep_map_base.cpp
            src/vdb2pc_publisher.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenVDB_LIBRARIES}
  ${IlmBase_LIBRARIES}
  ${catkin_LIBRARIES}
  ${TBB_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

add_executable(rep_map_node
  nodes/rep_map.cpp
)
target_link_libraries(rep_map_node
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${OpenVDB_LIBRARIES}
  ${TBB_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${rep_plugins_LIBRARIES}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
