cmake_minimum_required(VERSION 3.0.2)
project(rgc_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread ")

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  image_transport
  nav_msgs
  rosbag
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  ground_msg
  tf
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs rosbag roscpp rospy sensor_msgs std_msgs ground_msg
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
)

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
target_include_directories(fast_gicp PUBLIC
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(scanRegistration src/scanRegistration.cpp)
target_link_libraries(scanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(RGC_odometer src/RGC_odometer.cpp)
target_link_libraries(RGC_odometer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} fast_gicp)

add_executable(RGC_mapping src/RGC_mapping.cpp)
target_link_libraries(RGC_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})