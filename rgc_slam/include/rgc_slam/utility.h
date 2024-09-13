#pragma once

#include <ros/ros.h>

#include "ground_msg/groundparam.h"

#include <std_msgs/Header.h>
#include "std_msgs/Float64MultiArray.h"
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <boost/format.hpp>

using namespace std;
using namespace Eigen;
using namespace pcl;

#define R_EARTH 6371393.0 // m

int USE_IMU = 1, USE_GNSS = 0, USE_GROUND = 0, LoopClosureEnable = 1;
float init_x = 0, init_y = 0, init_z = 0, init_yaw = 0;

typedef pcl::PointXYZI PointType;

typedef pcl::PointXYZINormal PointType2;



double NormalizationAngle(double angle)
{
    if (angle > M_PI)
        angle -= 2 * M_PI;
    else if (angle < -M_PI)
        angle += 2 * M_PI;

    return angle;
}

double NormalizationRollPitchAngle(double angle)
{
    if (angle > M_PI / 2)
        angle -= M_PI;
    else if (angle < -M_PI / 2)
        angle += M_PI;

    return angle;
}

class Utility
{
  public:
    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }
    //预积分add
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }
    
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

};

class Mid_Filter
{
  public:
    double data_buf[500], filte_buf[500];
    int filter_count = 0, filter_size = 201;

    Mid_Filter(int size)
    {
        filter_size = size;
    }
    
    double MFilter(double data)
    {
        double tem = 0;

        data_buf[filter_count] = data;
        filter_count++;
        if (filter_count >= filter_size) filter_count = 0;

        for (int i = 0; i < filter_size; i++) filte_buf[i] = data_buf[i];
    
        for (int i = 0; i < filter_size - 1; i++)
        {
            for (int j = 0; j < filter_size - i - 1; j++)
            {
                if (filte_buf[j] > filte_buf[j + 1])
                {
                    tem = filte_buf[j]; 
                    filte_buf[j] = filte_buf[j + 1]; 
                    filte_buf[j + 1] = tem;
                }
            }
        }
        tem = filte_buf[(filter_size - 1) / 2];
        return tem;
    }

};

typedef struct
{
    int count = 0;
    
    double t = 0;
    double k = 0.002;
    double ax = 0, ay = 0, az = 0;
    double gx = 0, gy = 0, gz = 0;
    Eigen::Vector3d ba = Eigen::Vector3d(0.23054, -0.22046, -0.14313);
    Eigen::Vector3d bg = Eigen::Vector3d(0.00127, -0.00061, -0.00267);
    // Eigen::Vector3d ba = Eigen::Vector3d(-0.004485, -0.021716, 0.025998);
    // Eigen::Vector3d bg = Eigen::Vector3d(0.003796, 0.001246, 0.002479);

    Eigen::Quaterniond Q_init = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d R_init = Eigen::Matrix3d::Identity();
    Eigen::Quaterniond q_eigen = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d Rwi = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Pwi = Eigen::Vector3d::Zero();
    Eigen::Vector3d Vwi = Eigen::Vector3d::Zero();
    Eigen::Quaterniond Qwi = Eigen::Quaterniond::Identity();

    double roll = 0, pitch = 0, yaw = 0;
    double roll_acc = 0, pitch_acc = 0, yaw_acc = 0;
    double roll_init = -0.013, pitch_init = 0.0096, yaw_init = 0;
} imu_s;

typedef struct
{
    // std_msgs::Header header;
    double t=0;
    int count = 0;
    double latitude=0, longitude=0, altitude=0;

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();

    float covariance[9] = {0};

} GNSS_s;

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

class IntegrationBase
{
    public:
    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

    int O_P = 0;
    int O_R = 3;
    int O_V = 6;
    int O_BA = 9;
    int O_BG = 12;

    double ACC_N = 0.2;
    double GYR_N = 0.02;
    double ACC_W = 0.002;
    double GYR_W = 2.0e-4;
    Eigen::Vector3d G{0.0, 0.0, 9.81};

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
    IntegrationBase(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        //从大jacobian中分别取出 p,v,q,ba,bg 相对于 ba,bg 的 jacobian ;
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        //然后通过一阶近似对预积分值（测量值）进行估计，也即校正；
        //消除掉acc bias和gyro bias的影响, 对应IMU model中的\hat{\alpha},\hat{\beta},\hat{\gamma};
        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        //然后将（运动模型）预测值和预积分值（测量值）做差，获得imu残差（imu_factor）；
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }
};

typedef struct
{
    Eigen::Vector3d vector_norm = Eigen::Vector3d::Zero();
    Eigen::Vector3d vector_1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d vector_2 = Eigen::Vector3d::Zero();
    double distance = 0;
    double source = 0;
} ground_s;
