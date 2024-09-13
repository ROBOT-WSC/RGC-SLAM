#include "rgc_slam/utility.h"
#include "rgc_slam/tic_toc.h"
#include "lidarFactor.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <fast_gicp/gicp/fast_vgicp.hpp>

Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Eigen::Quaterniond q_init(1, 0, 0, 0);
Eigen::Quaterniond q_body2world(1, 0, 0, 0);
Eigen::Vector3d t_init(0, 0, 0);
Eigen::Quaterniond q_w_curr_f(1, 0, 0, 0);
Eigen::Quaterniond q_w_curr_delta(1, 0, 0, 0);
Eigen::Vector3d t_w_curr_f(0, 0, 0);
Eigen::Quaterniond q_last_curr_l(1, 0, 0, 0);
Eigen::Vector3d t_last_curr_l(0, 0, 0);
Eigen::Quaterniond q_w_curr_imu(1, 0, 0, 0);

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
double para_g[3] = {0, 0, 9.81};
double para_w[2] = {0, 0};
double para_vi[3] = {0, 0, 0};
double para_vj[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
Eigen::Map<Eigen::Vector3d> g_init(para_g);
Eigen::Map<Eigen::Vector3d> vi_init(para_vi);
Eigen::Map<Eigen::Vector3d> vj_init(para_vj);

Mid_Filter accx_MF(201), accy_MF(41), accz_MF(41);

struct DeltaGFactor_p
{
	DeltaGFactor_p(Eigen::Vector3d t_ij_, Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, double dt_)
		: t_ij(t_ij_), q_curr(q_curr_), pre_integration(pre_integration_), dt(dt_){}

	template <typename T>
	bool operator()(const T* g_i_j, const T* v_i, T* residuals) const
	{
        Eigen::Matrix<T, 3, 1> delta_p{T(pre_integration->delta_p.x()), T(pre_integration->delta_p.y()), T(pre_integration->delta_p.z())};
		Eigen::Matrix<T, 3, 1> delta_v{T(pre_integration->delta_v.x()), T(pre_integration->delta_v.y()), T(pre_integration->delta_v.z())};
		Eigen::Matrix<T, 3, 1> delta_t{T(t_ij.x()), T(t_ij.y()), T(t_ij.z())};
        Eigen::Quaternion<T> q_w_curr{T(q_curr(3,0)),T(q_curr(0,0)),T(q_curr(1,0)),T(q_curr(2,0))};

        Eigen::Matrix<T, 3, 1> erro;
        Eigen::Matrix<T, 3, 1> g_curr{g_i_j[0], g_i_j[1], g_i_j[2]};
        Eigen::Matrix<T, 3, 1> v_curr{v_i[0], v_i[1], v_i[2]};

        erro = delta_t + q_w_curr*(0.5*g_curr*T(dt)*T(dt) - v_curr*T(dt)) - delta_p;

        residuals[0]=erro(0,0);
        residuals[1]=erro(1,0);
        residuals[2]=erro(2,0);

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d t_ij_, const Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, const double dt_) 
	{
	  return (new ceres::AutoDiffCostFunction<DeltaGFactor_p, 3, 3, 3>(new DeltaGFactor_p(t_ij_, q_curr_, pre_integration_, dt_)));
	}

	Eigen::Vector3d t_ij;
    Eigen::Matrix<double, 4, 1> q_curr;
    IntegrationBase* pre_integration;
	double dt;
};

struct DeltaGFactor_v
{
	DeltaGFactor_v(Eigen::Vector3d t_ij_, Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, double dt_)
		: t_ij(t_ij_), q_curr(q_curr_), pre_integration(pre_integration_), dt(dt_){}

	template <typename T>
	bool operator()(const T* g_i_j, const T* v_i, const T* v_j, T* residuals) const
	{
        Eigen::Matrix<T, 3, 1> delta_p{T(pre_integration->delta_p.x()), T(pre_integration->delta_p.y()), T(pre_integration->delta_p.z())};
		Eigen::Matrix<T, 3, 1> delta_v{T(pre_integration->delta_v.x()), T(pre_integration->delta_v.y()), T(pre_integration->delta_v.z())};
		Eigen::Matrix<T, 3, 1> delta_t{T(t_ij.x()), T(t_ij.y()), T(t_ij.z())};
        Eigen::Quaternion<T> q_w_curr{T(q_curr(3,0)),T(q_curr(0,0)),T(q_curr(1,0)),T(q_curr(2,0))};

        Eigen::Matrix<T, 3, 1> erro;
        Eigen::Matrix<T, 3, 1> g_curr{g_i_j[0], g_i_j[1], g_i_j[2]};
        Eigen::Matrix<T, 3, 1> v_curr{v_i[0], v_i[1], v_i[2]};
        Eigen::Matrix<T, 3, 1> v_last{v_j[0], v_j[1], v_j[2]};

        erro = q_w_curr*(v_last - v_curr + g_curr*T(dt))-delta_v;

        residuals[0]=erro(0,0);
        residuals[1]=erro(1,0);
        residuals[2]=erro(2,0);

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d t_ij_, const Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, const double dt_) 
	{
	  return (new ceres::AutoDiffCostFunction<DeltaGFactor_v, 3, 3, 3, 3>(new DeltaGFactor_v(t_ij_, q_curr_, pre_integration_, dt_)));
	}

	Eigen::Vector3d t_ij;
    Eigen::Matrix<double, 4, 1> q_curr;
    IntegrationBase* pre_integration;
	double dt;
};

struct adjustGFactor_p
{
	adjustGFactor_p(Eigen::Vector3d t_ij_, Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, double dt_, Eigen::Vector3d g0_)
		: t_ij(t_ij_), q_curr(q_curr_), pre_integration(pre_integration_), dt(dt_), g0(g0_){}

	template <typename T>
	bool operator()(const T* w_i_j, const T* v_i, T* residuals) const
	{
        Eigen::Vector3d g_norm=g0.normalized();
        Eigen::Matrix<T, 3, 1> delta_p{T(pre_integration->delta_p.x()), T(pre_integration->delta_p.y()), T(pre_integration->delta_p.z())};
		Eigen::Matrix<T, 3, 1> delta_v{T(pre_integration->delta_v.x()), T(pre_integration->delta_v.y()), T(pre_integration->delta_v.z())};
		Eigen::Matrix<T, 3, 1> delta_t{T(t_ij.x()), T(t_ij.y()), T(t_ij.z())};
        Eigen::Matrix<T, 3, 1> gc0{T(g_norm.x()), T(g_norm.y()), T(g_norm.z())};
        Eigen::Quaternion<T> q_w_curr{T(q_curr(3,0)),T(q_curr(0,0)),T(q_curr(1,0)),T(q_curr(2,0))};
        Eigen::Matrix<T, 3, 1> erro;
        Eigen::Matrix<T, 3, 1> v_curr{v_i[0], v_i[1], v_i[2]};
        Eigen::Vector3d b1_,b2_;
        Eigen::Vector3d ee{1,0,0};
        b1_ = (g_norm).cross(ee);
        b2_ = (g_norm).cross(b1_);
        Eigen::Matrix<T, 3, 1> b1{T(b1_.x()), T(b1_.y()), T(b1_.z())};
        Eigen::Matrix<T, 3, 1> b2{T(b2_.x()), T(b2_.y()), T(b2_.z())};
        Eigen::Matrix<T, 3, 1> g1 = T(9.81)*gc0 + w_i_j[0]*b1 +w_i_j[1]*b2;
        erro = delta_t + q_w_curr*(0.5*g1*T(dt)*T(dt) - v_curr*T(dt)) - delta_p;

        residuals[0]=erro(0,0);
        residuals[1]=erro(1,0);
        residuals[2]=erro(2,0);

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d t_ij_, const Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, const double dt_, const Eigen::Vector3d g0_) 
	{
	  return (new ceres::AutoDiffCostFunction<adjustGFactor_p, 3, 2, 3>(new adjustGFactor_p(t_ij_, q_curr_, pre_integration_, dt_, g0_)));
	}

	Eigen::Vector3d t_ij, g0;
    Eigen::Matrix<double, 4, 1> q_curr;
    IntegrationBase* pre_integration;
	double dt;
};

struct adjustGFactor_v
{
	adjustGFactor_v(Eigen::Vector3d t_ij_, Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, double dt_, Eigen::Vector3d g0_)
		: t_ij(t_ij_), q_curr(q_curr_), pre_integration(pre_integration_), dt(dt_), g0(g0_){}

	template <typename T>
	bool operator()(const T* w_i_j, const T* v_i, const T* v_j, T* residuals) const
	{
        Eigen::Vector3d g_norm=g0.normalized();
        Eigen::Matrix<T, 3, 1> delta_p{T(pre_integration->delta_p.x()), T(pre_integration->delta_p.y()), T(pre_integration->delta_p.z())};
		Eigen::Matrix<T, 3, 1> delta_v{T(pre_integration->delta_v.x()), T(pre_integration->delta_v.y()), T(pre_integration->delta_v.z())};
		Eigen::Matrix<T, 3, 1> delta_t{T(t_ij.x()), T(t_ij.y()), T(t_ij.z())};
        Eigen::Matrix<T, 3, 1> gc0{T(g_norm.x()), T(g_norm.y()), T(g_norm.z())};
        Eigen::Quaternion<T> q_w_curr{T(q_curr(3,0)),T(q_curr(0,0)),T(q_curr(1,0)),T(q_curr(2,0))};

        Eigen::Matrix<T, 3, 1> erro;
        Eigen::Matrix<T, 3, 1> v_curr{v_i[0], v_i[1], v_i[2]};
        Eigen::Matrix<T, 3, 1> v_last{v_j[0], v_j[1], v_j[2]};
        Eigen::Vector3d b1_,b2_;
        Eigen::Vector3d ee{1,0,0};
        b1_ = (g_norm).cross(ee);
        b2_ = (g_norm).cross(b1_);
        Eigen::Matrix<T, 3, 1> b1{T(b1_.x()), T(b1_.y()), T(b1_.z())};
        Eigen::Matrix<T, 3, 1> b2{T(b2_.x()), T(b2_.y()), T(b2_.z())};
        Eigen::Matrix<T, 3, 1> g1 = T(9.81)*gc0 + w_i_j[0]*b1 +w_i_j[1]*b2;

        erro = q_w_curr*(v_last - v_curr + g1*T(dt))-delta_v;

        residuals[0]=erro(0,0);
        residuals[1]=erro(1,0);
        residuals[2]=erro(2,0);

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d t_ij_, const Eigen::Matrix<double, 4, 1> q_curr_, IntegrationBase* pre_integration_, const double dt_, const Eigen::Vector3d g0_) 
	{
	  return (new ceres::AutoDiffCostFunction<adjustGFactor_v, 3, 2, 3, 3>(new adjustGFactor_v(t_ij_, q_curr_, pre_integration_, dt_, g0_)));
	}

	Eigen::Vector3d t_ij, g0;
    Eigen::Matrix<double, 4, 1> q_curr;
    IntegrationBase* pre_integration;
	double dt;
};


Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);

    double cr = cos(roll); double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);

    Eigen::Matrix3d R;
    R<<  1,   0,    -sp,
            0,   cr,   sr*cp,
            0,   -sr,  cr*cp;

    return R;
}

class vg_ICP
{
    public:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloudFullRes;
    ros::Subscriber subCornerPointsSharp ;
    ros::Subscriber subSurfPointsFlat;
    ros::Subscriber subgroundparam;
    
    ros::Subscriber sub_imu;
    ros::Subscriber sub_RTK_data;
    ros::Subscriber sub_GPS_data;

    ros::Publisher pubLaserCloudCorner;
    ros::Publisher pubLaserCloudSurf;
    ros::Publisher pubLaserCloudFullRes;
    ros::Publisher pubLaserCloudsub;
    ros::Publisher pubgroundparam;
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubLaserPath;
    ros::Publisher pubRTKPath, pubGPSPath, pubRTKOdometry, pubGPSOdometry;

    std::mutex mBuf;

    queue<std::pair<double, Eigen::Vector3d>> accBuf;
    queue<std::pair<double, Eigen::Vector3d>> gyrBuf;
    queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
    queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
    queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
    queue<ground_msg::groundparam> groundBuf;
    
    ground_msg::groundparam ground_param;

    Eigen::Vector3d t_rl;
    Eigen::Matrix3d R_rl, R_il;
    Eigen::Matrix4d T_rl;

    imu_s IMU;
    IntegrationBase *IMU_Integration;
    GNSS_s RTK_new, RTK0;
    GNSS_s GPS_new, GPS0;
    ground_s ground_cur, ground_last, ground_mean;
    Eigen::Quaterniond delta_q_imu;
    Eigen::Quaterniond delta_q_imu2;
    Eigen::Vector3d delta_p_imu;
    Eigen::Vector3d delta_v_imu;
    Eigen::Vector3d linearized_ba;
    Eigen::Vector3d linearized_bg;
    Eigen::Vector3d acc0, acc1, gyr0, gyr1;
    vector<std::pair<double, Eigen::Vector3d>> accVector, gyrVector;

    double timeCornerPointsSharp = 0;
    double timeSurfPointsFlat = 0;
    double timeLaserCloudFullRes = 0;
    double timegroundparam = 0;
    double prevTime = 0, curTime = 0, farme_dt=0;

    float keyframeAddingDistance = 0.3; // 0.5m
    float keyframeAddingAngle = 0.2; // 57.3*0.2 degree 

    pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
    pcl::PointCloud<PointType>::Ptr laserCloudFullLast;
    pcl::PointCloud<PointType>::Ptr laserCloudFullnow;
    pcl::PointCloud<PointType2>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType2>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType2>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType2>::Ptr laserCloudSurfLast;
    
    pcl::PointCloud<PointType>::Ptr laserCloudsubmap;
    pcl::PointCloud<PointType>::Ptr surroundingmap;

    deque<pcl::PointCloud<PointType>::Ptr> surroundingCloud;
    deque<Eigen::Quaterniond> surrounding_q;
    deque<Eigen::Vector3d> surrounding_t;
    deque<ground_s> histoary_ground;
    deque<Eigen::Quaterniond> histoary_pose;
    int slipwide = 3;
    int ground_feq = 0;
    
    int firstinit = 2;//2
    int firstflagnum = 10;//10

    float planeResolution1 = 0.2;
    float planeResolution2 = 0.3;
    float planeResolution3 = 0.5;
    float down_simple_vgicp = 1;
    float vgicp_source = 1;
    float vgicp_source_sub = 1;

    int full_correspondence = 0;
    int laserCloudCornerNum = 0;
    int laserCloudSurfNum = 0;
    int laserCloudFullResNum = 0;

    int source_lim = 20;
    int target_lim = 20;



    int skipFrameNum = 1;
    const float SCAN_PERIOD = 0.1;
    const float DISTANCE_SQ_THRESHOLD = 9.0;
    const float NEARBY_SCAN = 2.5;
    double deg2rad = M_PI / 180.0, rad2deg = 180.0 / M_PI;
    int changegroundflag = 25;
    int gflag = 0;
    int imuflag = 1;

    nav_msgs::Path laserPath;

    std::ofstream f_save_pose_evo;
    std::string saveDirectory = "/home/wsc/1111/database/aold";

    Eigen::Affine3f T_Drift;
	Eigen::Vector3f t_drift;
	Eigen::Quaternionf q_drift;

    Eigen::Affine3f T_Drift2;
	Eigen::Vector3f t_drift2;
	Eigen::Quaternionf q_drift2;

    std::thread ICPThread;

   vg_ICP()
    {
        nh.param<int>("mapping_skip_frame", skipFrameNum, 1);
        printf("--LaserOdometry: Mapping %d Hz \n", 10 / skipFrameNum);

        nh.param<int>("USE_IMU", USE_IMU, 1);
        nh.param<int>("USE_GROUND", USE_GROUND, 1);
        printf("--LaserOdometry: USE_IMU %d \n", USE_IMU);
        printf("--LaserOdometry: USE_GROUND %d \n", USE_GROUND);
        nh.param<float>("init_x", init_x, 0);
        nh.param<float>("init_y", init_y, 0);
        nh.param<float>("init_z", init_z, 0);
        nh.param<float>("init_yaw", init_yaw, 0);
        nh.param<string>("saveDirectory", saveDirectory, "/home/robot220/code_ws/catkin_ws_robot_navigation/database");
        printf("--LaserOdometry: init_x %f \n", init_x);
        printf("--LaserOdometry: init_y %f \n", init_y);
        printf("--LaserOdometry: init_z %f \n", init_z);
        printf("--LaserOdometry: init_yaw %f \n", init_yaw);

        subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 10, &vg_ICP::laserCloudFullResHandler, this);
        subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 10, &vg_ICP::laserCloudSharpHandler, this);
        subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 10, &vg_ICP::laserCloudFlatHandler, this);
        
        subgroundparam = nh.subscribe("/ground_param", 200, &vg_ICP::groundHandler, this);
        sub_imu = nh.subscribe("/mynteye/imu/data_raw", 200, &vg_ICP::imu_callback, this); ///mynteye/imu/data_raw
        sub_RTK_data = nh.subscribe("/RTK/data_raw", 200, &vg_ICP::RTK_data_raw_callback, this);
        sub_GPS_data = nh.subscribe("/GPS/data_raw", 200, &vg_ICP::GPS_data_raw_callback, this);

        pubLaserCloudCorner = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner", 10);
        pubLaserCloudSurf = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf", 10);
        pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 10);
        pubLaserCloudsub = nh.advertise<sensor_msgs::PointCloud2>("/sub_cloud", 10);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 10);
        pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 10);
        pubgroundparam = nh.advertise<ground_msg::groundparam>("/ground_param_gable",100);

        pubRTKPath = nh.advertise<nav_msgs::Path>("/RTK_path", 10);
        pubRTKOdometry = nh.advertise<nav_msgs::Odometry>("/RTK_odom", 10);
        pubGPSPath = nh.advertise<nav_msgs::Path>("/GPS_path", 10);
        pubGPSOdometry = nh.advertise<nav_msgs::Odometry>("/GPS_odom", 10);

        R_il = Utility::ypr2R(Eigen::Vector3d(-1.29, -0.15, 0.65));
        // R_il = Utility::ypr2R(Eigen::Vector3d(179.603, -179.624, -179.937)); 
        t_rl = Eigen::Vector3d(0.68, 0, 0.34);
        R_rl = Utility::ypr2R(Eigen::Vector3d(0.0, -0.0, 0.0));
        T_rl = Eigen::Matrix4d::Identity();
		T_rl.block<3, 3>(0, 0) = R_rl;
        T_rl.block<3, 1>(0, 3) = t_rl;

        laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
        laserCloudFullLast.reset(new pcl::PointCloud<PointType>());
        laserCloudFullnow.reset(new pcl::PointCloud<PointType>());
        cornerPointsSharp.reset(new pcl::PointCloud<PointType2>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType2>());
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType2>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType2>());
        laserCloudsubmap.reset(new pcl::PointCloud<PointType>());
        surroundingmap.reset(new pcl::PointCloud<PointType>());
        boost::format fmt_pose("%s/%s");
        f_save_pose_evo.open((fmt_pose % saveDirectory % "Odometry_pose_evo.txt").str(), std::fstream::out);


        ICPThread = std::thread(&vg_ICP::ICP_thread, this);
    }
~vg_ICP()
    {
        f_save_pose_evo.close();
        printf("--LaserOdometry exit !!! \n");
    }

    void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullResMsg)
    {
        mBuf.lock();
        fullPointsBuf.push(laserCloudFullResMsg);
        mBuf.unlock();
    }

    void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharpMsg)
    {
        mBuf.lock();
        cornerSharpBuf.push(cornerPointsSharpMsg);
        mBuf.unlock();
    }

    void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlatMsg)
    {
        mBuf.lock();
        surfFlatBuf.push(surfPointsFlatMsg);
        mBuf.unlock();
    }

    void groundHandler(const ground_msg::groundparam & ground_msg)
    {
        mBuf.lock();
        groundBuf.push(ground_msg);
        mBuf.unlock();
    }

    void imu_callback(const sensor_msgs::Imu& imu_msg)
    {
        static int count = 0;
        if (count < 100)
		{
			count++;
			return; // 丢弃前0.5秒的数据；
		}

		double t = imu_msg.header.stamp.toSec();
		double ax = imu_msg.linear_acceleration.x;
		double ay = imu_msg.linear_acceleration.y;
		double az = imu_msg.linear_acceleration.z;
		double rx = imu_msg.angular_velocity.x;
		double ry = imu_msg.angular_velocity.y;
		double rz = imu_msg.angular_velocity.z;
		Eigen::Vector3d acc(ax, ay, az);
		Eigen::Vector3d gyr(rx, ry, rz);

		// if (get_init_imu_bias(acc, gyr, IMU) != 1)
		// {
		// 	return;
		// }

        Eigen::Vector3d acc_new, gyr_new;
        acc_new = acc - IMU.ba;
		gyr_new = gyr - IMU.bg;

		mBuf.lock();
		accBuf.push(std::make_pair(t, acc_new));
		gyrBuf.push(std::make_pair(t, gyr_new));
		mBuf.unlock();

		IMU.t = t;
		IMU.ax = acc_new.x();
		IMU.ay = acc_new.y();
		IMU.az = acc_new.z();
		IMU.gx = gyr_new.x();
		IMU.gy = gyr_new.y();
		IMU.gz = gyr_new.z();
		IMU.count++;
		ComplementaryFilter(IMU);
    }

    void RTK_data_raw_callback(const sensor_msgs::NavSatFix & rtk_raw)
    {
        RTK_new.count++;
        RTK_new.t = rtk_raw.header.stamp.toSec();
        RTK_new.latitude = rtk_raw.latitude;
        RTK_new.longitude = rtk_raw.longitude;
        RTK_new.altitude = rtk_raw.altitude;
        for (int i = 0; i < 9; i++)
        {
            RTK_new.covariance[i] = rtk_raw.position_covariance[i];
        }
        if (RTK_new.count < 10)
        {
            RTK0 = RTK_new;
            // printf("--RTK0: lon: %.7f, lat: %.7f, alt: %.3f \n", RTK0.longitude, RTK0.latitude, RTK0.altitude);
            return;
        }

        Eigen::Vector3d p_tem;
        GPS_to_XYZ(RTK_new, RTK0, p_tem);
        // printf("--RTK: lon: %.7f, lat: %.7f, alt: %.3f -- x: %.3f, y: %.3f, z: %.3f \n", RTK_new.longitude, RTK_new.latitude, RTK_new.altitude, p_tem.x(), p_tem.y(), p_tem.z());

        pub_RTK_path(RTK_new.t, p_tem);

        pub_RTK_odometry(pubRTKOdometry, RTK_new);
    }

    void GPS_data_raw_callback(const sensor_msgs::NavSatFix & gps_raw)
    {
        GPS_new.count++;
        GPS_new.t = gps_raw.header.stamp.toSec();
        GPS_new.latitude = gps_raw.latitude;
        GPS_new.longitude = gps_raw.longitude;
        GPS_new.altitude = gps_raw.altitude;
        for (int i = 0; i < 9; i++)
        {
            GPS_new.covariance[i] = gps_raw.position_covariance[i];
        }
        
        if (GPS_new.count < 10)
        {
            GPS0 = GPS_new;
            // printf("--RTK0: lon: %.7f, lat: %.7f, alt: %.3f \n", RTK0.longitude, RTK0.latitude, RTK0.altitude);
            return;
        }

        Eigen::Vector3d p_tem;
        GPS_to_XYZ(GPS_new, GPS0, p_tem);
        // printf("--GPS: lon: %.7f, lat: %.7f, alt: %.3f -- x: %.3f, y: %.3f, z: %.3f \n", GPS_new.longitude, GPS_new.latitude, GPS_new.altitude, p_tem.x(), p_tem.y(), p_tem.z());

        pub_GPS_path(GPS_new.t, p_tem);

        pub_GPS_odometry(pubGPSOdometry, GPS_new);
    }



    void ComplementaryFilter(imu_s & t_imu)
	{
		static imu_s imu_last;
		static int first_flag = 1;
		static double d_t = 0.0, ax = 0, ay = 0, ratio_x = 0, ratio_y = 0;
		static Eigen::Vector3d imu_angular, imu_acc;

		d_t = t_imu.t - imu_last.t;
		if (first_flag == 1)
		{
			imu_last = t_imu;
			d_t = 0.005;
			first_flag = 0;
		}


		t_imu.ax = accx_MF.MFilter(t_imu.ax);
		t_imu.ay = accy_MF.MFilter(t_imu.ay);
		t_imu.az = accz_MF.MFilter(t_imu.az);

		if (t_imu.count < 300) 
        {
            t_imu.k = 0.9; 
        }
        else
        {
            t_imu.k = 0.002;
        }

		if (abs(t_imu.gz * rad2deg) < 0.2)
		{
			t_imu.gz = 0;
		}

		if (t_imu.count > 300)
		{
			Eigen::Matrix3d Rimu = Utility::ypr2R(Eigen::Vector3d(0, t_imu.pitch, t_imu.roll) * rad2deg);
			imu_acc = Rimu * Eigen::Vector3d(0, 0, 9.81);
			ax = imu_acc.x();
			ratio_x = abs(ax) / abs(t_imu.ax) ;
			if (abs(t_imu.ax) > 0.3 && ratio_x < 0.8)
			{
				t_imu.ax = ratio_x * t_imu.ax + (1 - ratio_x) * ax;
			}

			ay = imu_acc.y();
			ratio_y = abs(ay) / abs(t_imu.ay);
			if (abs(t_imu.ay) > 0.3 && ratio_y < 0.8)
			{
				t_imu.ay = ratio_y * t_imu.ay + (1 - ratio_y) * ay;
			}
		}

		t_imu.roll_acc = atan2(t_imu.ay, t_imu.az);
		t_imu.pitch_acc = -atan2(t_imu.ax, t_imu.az);

        Eigen::Matrix3d R_eul2w = eulerRates2bodyRates(Eigen::Vector3d(t_imu.roll, t_imu.pitch, t_imu.yaw));
        Eigen::Vector3d delta_euler = R_eul2w.inverse()*Eigen::Vector3d(t_imu.gx, t_imu.gy, t_imu.gz);
        t_imu.gx = delta_euler.x();
        t_imu.gy = delta_euler.y();
        t_imu.gz = delta_euler.z();

		t_imu.roll = t_imu.k * t_imu.roll_acc + (1.0 - t_imu.k) * (t_imu.roll + t_imu.gx * d_t);
		t_imu.pitch = t_imu.k * t_imu.pitch_acc + (1.0 - t_imu.k) * (t_imu.pitch + t_imu.gy * d_t);
		t_imu.yaw += t_imu.gz / 0.9998 * d_t; // 0.9998

		if (abs(t_imu.gz * rad2deg) > 5.0)
		{
			double low = 0.005;
			t_imu.roll = low * t_imu.roll + (1 - low) * imu_last.roll;
			t_imu.pitch = low * t_imu.pitch + (1 - low) * imu_last.pitch;
		}

		t_imu.roll = NormalizationRollPitchAngle(t_imu.roll);
		t_imu.pitch = NormalizationRollPitchAngle(t_imu.pitch);
		t_imu.yaw = NormalizationAngle(t_imu.yaw);
		t_imu.Rwi = Utility::ypr2R(Eigen::Vector3d(t_imu.yaw, t_imu.pitch, t_imu.roll) * rad2deg);

		imu_last = t_imu;
    }

    int get_init_imu_bias(Eigen::Vector3d acc, Eigen::Vector3d gyr, imu_s & t_imu)
    {
        static std::vector<double> ax_buf, ay_buf, az_buf;
        static std::vector<double> gx_buf, gy_buf, gz_buf;
        static int filter_count = 0, filter_size = 700;
        static int init_flag = 0;
        double ax_t = acc.x(), ay_t = acc.y(), az_t = acc.z();
		double gx_t = gyr.x(), gy_t = gyr.y(), gz_t = gyr.z();

        if (init_flag == 0)
        {
            printf("--calculating init imu bias, please wait 5s : %.3f \n", 1.0 * filter_count / 200);

            if (checkImuState(ax_t, ay_t, az_t, gx_t, gy_t, gz_t))
            {
				ax_buf.push_back(ax_t);
				ay_buf.push_back(ay_t);
				az_buf.push_back(az_t);
				gx_buf.push_back(gx_t);
				gy_buf.push_back(gy_t);
				gz_buf.push_back(gz_t);

                filter_count++;
                if (filter_count >= filter_size)
                {
					std::sort(ax_buf.begin(), ax_buf.end());
					std::sort(ay_buf.begin(), ay_buf.end());
					std::sort(az_buf.begin(), az_buf.end());
					std::sort(gx_buf.begin(), gx_buf.end());
					std::sort(gy_buf.begin(), gy_buf.end());
					std::sort(gz_buf.begin(), gz_buf.end());

                    ax_t = 0, ay_t = 0, az_t = 0, gx_t = 0, gy_t = 0, gz_t = 0;
                    for (int i = 50; i < filter_size - 50; i++) 
                    {
                        ax_t += ax_buf[i];
                        ay_t += ay_buf[i];
                        az_t += az_buf[i];
                        gx_t += gx_buf[i];
                        gy_t += gy_buf[i];
                        gz_t += gz_buf[i];
                    }
                    t_imu.ba.x() = ax_t / (filter_size - 100);
                    t_imu.ba.y() = ay_t / (filter_size - 100);
                    t_imu.ba.z() = az_t / (filter_size - 100);
                    t_imu.bg.x() = gx_t / (filter_size - 100);
                    t_imu.bg.y() = gy_t / (filter_size - 100);
                    t_imu.bg.z() = gz_t / (filter_size - 100);

                    t_imu.roll_init = atan2(t_imu.ba.x(), t_imu.ba.z());
                    t_imu.pitch_init = -atan2(t_imu.ba.x(), sqrt(t_imu.ba.y() * t_imu.ba.y() + t_imu.ba.z() * t_imu.ba.z()));
                    
                    t_imu.Q_init = Utility::ypr2R(Eigen::Vector3d(0, t_imu.pitch_init, t_imu.roll_init) * rad2deg);
                    t_imu.Q_init.normalize();
                    t_imu.R_init = t_imu.Q_init.toRotationMatrix();

                    init_flag = 1;

                    printf("\n--init imu gyro bias(rad): ax: %.5f, ay: %.5f, az: %.5f, gx: %.5f, gy: %.5f, gz: %.5f \n\n\n ", 
                    t_imu.ba.x(), t_imu.ba.y(), t_imu.ba.z(), t_imu.bg.x(), t_imu.bg.y(), t_imu.bg.z());

                    printf("--init imu rotation(deg): roll: %.3f, pitch: %.3f, yaw: %.3f \n\n\n",
                    t_imu.roll_init*57.3, t_imu.pitch_init*57.3, t_imu.yaw_init*57.3);
                } 
            }
			else
			{
				printf("--robot imu is not static !!! \n");
			}
        }
        return init_flag;
    }

    bool checkImuState(double ax, double ay, double az, double wx, double wy, double wz)
    {
        double kGravity = 9.81;
        double kAccelerationThreshold = 0.5;
        double kAngularVelocityThreshold = 0.05;

        double acc_magnitude = sqrt(ax * ax + ay * ay + az * az);

        if (fabs(acc_magnitude - kGravity) > kAccelerationThreshold)
            return false;

        if (fabs(wx) > kAngularVelocityThreshold ||
            fabs(wy) > kAngularVelocityThreshold ||
            fabs(wz) > kAngularVelocityThreshold)
            return false;

        return true;
    }

    void GPS_to_XYZ(GNSS_s &G1, GNSS_s G0, Eigen::Vector3d &p)
    {
        double lon = G1.longitude - G0.longitude;
        double lat = G1.latitude - G0.latitude;
        p.x() = lon * R_EARTH * cos(G1.latitude * deg2rad) * deg2rad ;
        p.y() = lat * R_EARTH * deg2rad ;
        p.z() = G1.altitude - G0.altitude;

        G1.pos = p;
    }

    void pub_RTK_path(double t_tem, Eigen::Vector3d p_tem)
	{
        static nav_msgs::Path RTK_Path;
		geometry_msgs::PoseStamped RobotPose;
		RobotPose.header.stamp = ros::Time().fromSec(t_tem);
		RobotPose.header.frame_id = "world";
        RobotPose.pose.orientation.x = 0;
		RobotPose.pose.orientation.y = 0;
		RobotPose.pose.orientation.z = 0;
		RobotPose.pose.orientation.w = 1;
		RobotPose.pose.position.x = p_tem.x();
		RobotPose.pose.position.y = p_tem.y();
		RobotPose.pose.position.z = p_tem.z();
		RTK_Path.header.stamp = RobotPose.header.stamp;
		RTK_Path.header.frame_id = "world";
		RTK_Path.poses.push_back(RobotPose);
		pubRTKPath.publish(RTK_Path);
	}

    void pub_RTK_odometry(ros::Publisher thisPub, GNSS_s Gt)
	{
        nav_msgs::Odometry OdometryROS;
        OdometryROS.header.stamp = ros::Time().fromSec(Gt.t);
        OdometryROS.header.frame_id = "camera_init";
        OdometryROS.child_frame_id = "GNSS";
        OdometryROS.pose.pose.position.x = Gt.pos.x();
        OdometryROS.pose.pose.position.y = Gt.pos.y();
        OdometryROS.pose.pose.position.z = Gt.pos.z();
        OdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0, 0, 0);
        for(int i = 0; i < 9; i++)
        {
            OdometryROS.pose.covariance[i] = Gt.covariance[i];
        }
        thisPub.publish(OdometryROS);
	}

    void pub_GPS_path(double t_tem, Eigen::Vector3d p_tem)
	{
        static nav_msgs::Path RTK_Path;
		geometry_msgs::PoseStamped RobotPose;
		RobotPose.header.stamp = ros::Time().fromSec(t_tem);
		RobotPose.header.frame_id = "world";
		RobotPose.pose.orientation.x = 0;
		RobotPose.pose.orientation.y = 0;
		RobotPose.pose.orientation.z = 0;
		RobotPose.pose.orientation.w = 1;
		RobotPose.pose.position.x = p_tem.x();
		RobotPose.pose.position.y = p_tem.y();
		RobotPose.pose.position.z = p_tem.z();
		RTK_Path.header.stamp = RobotPose.header.stamp;
		RTK_Path.header.frame_id = "world";
		RTK_Path.poses.push_back(RobotPose);
		pubGPSPath.publish(RTK_Path);
	}

    void pub_GPS_odometry(ros::Publisher thisPub, GNSS_s Gt)
	{
        nav_msgs::Odometry OdometryROS;
        OdometryROS.header.stamp = ros::Time().fromSec(Gt.t);
        OdometryROS.header.frame_id = "camera_init";
        OdometryROS.child_frame_id = "GNSS";
        OdometryROS.pose.pose.position.x = Gt.pos.x();
        OdometryROS.pose.pose.position.y = Gt.pos.y();
        OdometryROS.pose.pose.position.z = Gt.pos.z();
        OdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0, 0, 0);
        for(int i = 0; i < 9; i++)
        {
            OdometryROS.pose.covariance[i] = Gt.covariance[i];
        }
        thisPub.publish(OdometryROS);
	}

    // ICP线程
    void ICP_thread()
    {
        int frameCount = 0;
        int submapflag = 0;
        ros::Rate loop_rate(500); //设置周期休眠时间
        printf("--ICP_thread begain !!! \n");
        while(1)
        {   
            loop_rate.sleep();
            while (!cornerSharpBuf.empty() && !surfFlatBuf.empty() && !fullPointsBuf.empty() && !groundBuf.empty())
            {
                mBuf.lock();
                while (!cornerSharpBuf.empty() && cornerSharpBuf.front()->header.stamp.toSec() < fullPointsBuf.front()->header.stamp.toSec())
                    cornerSharpBuf.pop();
                if (cornerSharpBuf.empty())
				{
					mBuf.unlock();
					break;
				}
               
                while (!surfFlatBuf.empty() && surfFlatBuf.front()->header.stamp.toSec() < fullPointsBuf.front()->header.stamp.toSec())
                    surfFlatBuf.pop();
                if (surfFlatBuf.empty())
				{
					mBuf.unlock();
					break;
				}

                while (!groundBuf.empty() && groundBuf.front().header.stamp.toSec() < fullPointsBuf.front()->header.stamp.toSec())
                    groundBuf.pop();
                if (groundBuf.empty())
				{
					mBuf.unlock();
					break;
				}
                
                timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
                timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
                timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
                timegroundparam = groundBuf.front().header.stamp.toSec();

                if (timeCornerPointsSharp != timeLaserCloudFullRes || timeSurfPointsFlat != timeLaserCloudFullRes || timegroundparam != timeLaserCloudFullRes) 
                {
                    printf("--laserodometry_thread: unsync messeage! \n");
                    mBuf.unlock();
                    ROS_BREAK();
                }
                TicToc t_whole;
                curTime = timeLaserCloudFullRes;

                if (fullPointsBuf.size() < firstinit)
                {
                    mBuf.unlock();
                    break;
                }
                static int first_flag = 0;
                if (first_flag < firstflagnum)
                {
                    first_flag++;
                    fullPointsBuf.pop(); 
                    groundBuf.pop();
                    prevTime = curTime;

                    t_w_curr = Eigen::Vector3d(init_x, init_y, init_z);
                    if (USE_IMU)
                    {
                        q_w_curr = IMU.Rwi * R_il;
                        q_w_curr = Utility::ypr2R(Utility::R2ypr(Eigen::Matrix3d(q_w_curr)) + Eigen::Vector3d(init_yaw, 0, 0));
					    q_w_curr.normalize();
                        q_w_curr_f = q_w_curr;

                    }
                    else
                    {
                        q_w_curr = Eigen::Quaterniond::Identity();
                        q_w_curr_f = q_w_curr;
                    }
                    
                    mBuf.unlock();
                    break;
                }
                if (USE_IMU)
                {
                    if (!getIMUInterval(prevTime, curTime, accVector, gyrVector))
                    {
                        fullPointsBuf.pop();
                        groundBuf.pop();
                        mBuf.unlock();
                        break;
                    }
                    delta_q_imu = Eigen::Quaterniond::Identity();
                    delta_q_imu2 = Eigen::Quaterniond::Identity();
                    delta_p_imu = Eigen::Vector3d::Zero();
                    delta_v_imu = Eigen::Vector3d::Zero();
                    linearized_ba = IMU.ba;
                    linearized_bg = IMU.bg;
                    IMU_Integration = new IntegrationBase{IMU.ba, IMU.bg};
                    for (size_t i = 0; i < accVector.size(); i++)
                    {
                        double dt = 0;
                        if (i == 0)
                            {
                                dt = accVector[i].first - prevTime;
                                acc0 = accVector[i].second;
                                acc1 = accVector[i].second;
                                gyr0 = gyrVector[i].second;
                                gyr1 = gyrVector[i].second;
                            }
                        else if (i == (accVector.size() - 1))
                            {
                                dt = curTime - accVector[i - 1].first;
                                acc0 = accVector[i-1].second;
                                acc1 = accVector[i].second;
                                gyr0 = gyrVector[i-1].second;
                                gyr1 = gyrVector[i].second;
                            }
                        else
                            {
                                dt = accVector[i].first - accVector[i - 1].first;
                                acc0 = accVector[i-1].second;
                                acc1 = accVector[i].second;
                                gyr0 = gyrVector[i-1].second;
                                gyr1 = gyrVector[i].second;
                            }              
                        IMU_preintegration(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                        IMU_preintegration2(dt, acc0, acc1, gyr0, gyr1, IMU_Integration);
                    }
                    q_last_curr = delta_q_imu; 
                    q_last_curr.normalize();
                }
                cornerPointsSharp->clear();
                pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
                laserCloudCornerNum = cornerPointsSharp->size();
                cornerSharpBuf.pop(); 

                surfPointsFlat->clear();
                pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
                laserCloudSurfNum = surfPointsFlat->size();
                surfFlatBuf.pop();

                laserCloudFullRes->clear();
                pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
                laserCloudFullResNum = laserCloudFullRes->size();
                fullPointsBuf.pop();

                ground_param = groundBuf.front();
                groundBuf.pop();
                ground_cur.vector_norm = Eigen::Vector3d(ground_param.normx,ground_param.normy,ground_param.normz);
                ground_cur.vector_1 = Eigen::Vector3d(ground_param.vector1x,ground_param.vector1y,ground_param.vector1z);
                ground_cur.vector_2 = Eigen::Vector3d(ground_param.vector2x,ground_param.vector2y,ground_param.vector2z);
                ground_cur.distance = ground_param.distance;
                ground_cur.source = ground_param.source;
                mBuf.unlock();
                farme_dt = curTime - prevTime;
                prevTime = curTime;

                adjustDistortion();
                
                float timee = 0;
                if(laserCloudFullLast->points.size()!=0)
                {
                    if(submapflag == 0)
                    {
                        surroundingCloud.push_back(laserCloudFullLast);
                        surrounding_q.push_back(q_init);
                        surrounding_t.push_back(t_init);
                        histoary_ground.push_back(ground_cur);
                        histoary_pose.push_back(q_w_curr_delta);

                        *laserCloudsubmap += *laserCloudFullLast;
                    }
                    submapflag++;
                    TicToc t_opt;
                    //source downfilter
                    pcl::PointCloud<PointType> FullPointsLessFlatDS;
                    pcl::VoxelGrid<PointType> downSizeFilterFull;
                    pcl::PointCloud<PointType>::Ptr FullPointsLessFlat(new pcl::PointCloud<PointType>);
                    downSizeFilterFull.setLeafSize(planeResolution1, planeResolution1, planeResolution1);
                    downSizeFilterFull.setInputCloud(laserCloudFullRes);
                    downSizeFilterFull.filter(FullPointsLessFlatDS);
                    *FullPointsLessFlat+=FullPointsLessFlatDS;
                    laserCloudFullResNum = FullPointsLessFlat->size();
                    //target downfilter
                    pcl::PointCloud<PointType> FullPointsLessFlatlastDS;
                    pcl::VoxelGrid<PointType> downSizeFilterFulllast;
                    pcl::PointCloud<PointType>::Ptr FullPointsLessFlatlast(new pcl::PointCloud<PointType>);
                    downSizeFilterFulllast.setLeafSize(planeResolution2, planeResolution2, planeResolution2);
                    downSizeFilterFulllast.setInputCloud(laserCloudsubmap);
                    downSizeFilterFulllast.filter(FullPointsLessFlatlastDS);
                    *FullPointsLessFlatlast+=FullPointsLessFlatlastDS;
                    //pre 
                    Eigen::Matrix4f T2;
                    T2.setIdentity();
                    T2.block<3,3>(0,0) = (q_last_curr.toRotationMatrix()).cast<float>();
                    T2.topRightCorner(3, 1) = t_last_curr.cast<float>();
                    //define vgicp
                    fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI> vgicp;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>);
                    vgicp.setResolution(down_simple_vgicp);
                    vgicp.setMaximumIterations(25);
                    vgicp.setMaxCorrespondenceDistance(2);
                    vgicp.setTransformationEpsilon(1e-6);
                    vgicp.setEuclideanFitnessEpsilon(1e-6);
                    vgicp.setRANSACIterations(0);
                    vgicp.setNumThreads(14);
                    vgicp.setInputTarget(FullPointsLessFlatlast);
                    vgicp.setInputSource(FullPointsLessFlat);
                    vgicp.align(*aligned, T2);
                    vgicp_source = vgicp.getFitnessScore();
                    T_Drift = vgicp.getFinalTransformation();
		            t_drift = T_Drift.translation();
		            q_drift = T_Drift.rotation();
                    t_last_curr_l = t_drift.cast<double>();
                    q_last_curr_l = Eigen::Quaterniond(q_drift).cast<double>();

                    para_q[0] = q_last_curr_l.x();
                    para_q[1] = q_last_curr_l.y();
                    para_q[2] = q_last_curr_l.z();
                    para_q[3] = q_last_curr_l.w();
                    para_t[0] = t_last_curr_l.x();
                    para_t[1] = t_last_curr_l.y();
                    para_t[2] = t_last_curr_l.z();    
                    //因子优化的方式组合
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;
                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);
                    ceres::CostFunction* lidar_dq_factor = DeltaRFactor::Create(q_last_curr_l.x(), q_last_curr_l.y(), q_last_curr_l.z(), q_last_curr_l.w(), vgicp_source);
                    problem.AddResidualBlock(lidar_dq_factor, NULL, para_q);
                    //地面约束
                    Eigen::Vector3d ground_norm_cur = q_last_curr_l*ground_cur.vector_norm;
                    double ground_distance_cur = ground_cur.distance + ground_norm_cur.dot(t_last_curr_l);
                    double ground_erro_1 = (ground_last.distance*ground_last.vector_norm - ground_distance_cur*ground_norm_cur).norm();
                    double ground_erro_2 = abs(ground_last.vector_1.dot(ground_norm_cur));
                    double ground_erro_3 = abs(ground_last.vector_2.dot(ground_norm_cur));
                    Eigen::Vector3d d_ypr = Utility::R2ypr(delta_q_imu.toRotationMatrix());
                    //地面约束位姿更新                    
                    std::cout<<"grounderro1:"<<ground_erro_1<<"grounderro2:"<<ground_erro_2<<"grounderro3:"<<ground_erro_3<<std::endl;
                    if(ground_erro_1>=0.02&&ground_erro_2>=0.02&&fabs(d_ypr.y()) > 0.5)
                    {
                        ROS_WARN("-------ground change-----");
                        changegroundflag=0;
                        gflag=1;
                        
                    }
                    if(gflag==1&&changegroundflag<25)
                    {                     
                      changegroundflag++;
                      if(changegroundflag==25)
                      {
                        Eigen::Quaterniond last_q;
                        Eigen::Vector3d temp;
                        double pr_erro = 1000;
                        
                        Eigen::Vector3d now_ypr = Utility::R2ypr(q_w_curr.toRotationMatrix());
                        for(int i=0;i<histoary_pose.size();i++)
                        {
                            temp = Utility::R2ypr(histoary_pose[i].toRotationMatrix());
                            double p_erro = temp.y()-now_ypr.y();
                            double r_erro = temp.z()-now_ypr.z();
                            if (sqrt(p_erro * p_erro + r_erro * r_erro) < pr_erro) // 模长大于0.5度，即每秒5度时，则加大imu置信度；
                            {
                                pr_erro = sqrt(p_erro * p_erro + r_erro * r_erro);
                                last_q = histoary_pose[i];
                            }
                        }
                        std::cout<<"pr_erro----------------------"<<pr_erro<<"----------------------------"<<std::endl;
                        std::cout<<"pr_erro----------------------"<<pr_erro<<"----------------------------"<<std::endl;
                        if(pr_erro<4)
                        {
                            q_w_curr_delta = last_q;
                            gflag=0;
                            ROS_WARN("this ground can find same ground in histoary plane");
                        }
                        else
                        {
                            q_w_curr_delta = q_w_curr;
                            histoary_pose.push_back(q_w_curr_delta);
                            gflag=0;
                        }
                      }
                    }                   
                    q_w_curr_f = q_w_curr_delta.conjugate()*q_w_curr;
                    q_w_curr_f.normalize();
                    if(USE_GROUND && gflag == 0)
                    {
                        ceres::CostFunction* lidar_dp_factor = DeltaPFactor::Create(t_last_curr_l.x(), t_last_curr_l.y(), t_last_curr_l.z(), vgicp_source/10);
                        problem.AddResidualBlock(lidar_dp_factor, NULL, para_t); 
                        double ground_cov = 0.2;
                        // ROS_INFO("----------------------now using ground factor---------------------");
                        Eigen::Matrix<double, 4, 1> q_curr{q_w_curr_f.x(),q_w_curr_f.y(),q_w_curr_f.z(),q_w_curr_f.w()};
                        ceres::CostFunction* ground_factor = Ground_DeltaFactor::Create(ground_last, ground_cur, q_curr,ground_cov);
                        problem.AddResidualBlock(ground_factor, NULL, para_q, para_t);
                    }
                    else
                    {
                        t_last_curr = t_last_curr_l;
                        // ROS_WARN("--------------now ground unuse---------------");
                    }
                    //IMU角度约束
                    if (USE_IMU&&imuflag==1)
                        {
                            // 添加相对约束： 旋转方差设置为0.01；
                            double imu_cov = 0.0004;
                            Eigen::Vector3d d_ypr = Utility::R2ypr(delta_q_imu.toRotationMatrix());
                            if (d_ypr.norm() > 0.6) // 模长大于0.5度，即每秒5度时，则加大imu置信度；
                            {
                                imu_cov = 0.0005;
                            }
                            else
                            {
                                imu_cov = 1-vgicp_source;
                            }
                            ceres::CostFunction* imu_dq_factor = DeltaRFactor::Create(delta_q_imu.x(), delta_q_imu.y(), delta_q_imu.z(), delta_q_imu.w(), imu_cov);
                            problem.AddResidualBlock(imu_dq_factor, NULL, para_q);
                        }
                    //优化重力方向
                    if(submapflag==1)
                    {
                        Eigen::Vector3d Vij;
                        Eigen::Vector3d tij;
                        Vij = t_last_curr/farme_dt;
                        if(Vij.norm()<0.1)
                        {
                            tij =  Eigen::Vector3d::Zero();
                        }
                        else
                        {
                            tij = t_last_curr;
                        }
                        ceres::Problem::Options problem_g_options;
                        ceres::Problem problem_g(problem_g_options);
                        problem_g.AddParameterBlock(para_g, 3);
                        problem_g.AddParameterBlock(para_vi, 3);
                        problem_g.AddParameterBlock(para_vj, 3);
                        Eigen::Matrix<double, 4, 1> q_cur_g{q_w_curr.x(),q_w_curr.y(),q_w_curr.z(),q_w_curr.w()};
                        ceres::CostFunction* init_g_factor_p = DeltaGFactor_p::Create(tij, q_cur_g, IMU_Integration, farme_dt);
                        ceres::CostFunction* init_g_factor_v = DeltaGFactor_v::Create(tij, q_cur_g, IMU_Integration, farme_dt);
                        problem_g.AddResidualBlock(init_g_factor_p, NULL, para_g, para_vi);
                        problem_g.AddResidualBlock(init_g_factor_v, NULL, para_g, para_vi, para_vj);
                        ceres::Solver::Options options_g;
                        options_g.linear_solver_type = ceres::DENSE_QR;
                        options_g.max_num_iterations = 8;
                        options_g.minimizer_progress_to_stdout = false;
                        ceres::Solver::Summary summary_g;
                        ceres::Solve(options_g, &problem_g, &summary_g); 
                        //二自由度调整
                        ceres::Problem::Options problem_adj_options;
                        ceres::Problem problem_adj(problem_adj_options);
                        problem_adj.AddParameterBlock(para_w, 2);
                        problem_adj.AddParameterBlock(para_vi, 3);
                        problem_adj.AddParameterBlock(para_vj, 3);
                        Eigen::Vector3d g0;
                        g0 = g_init;                                             
                        ceres::CostFunction* adjust_g_factor_p = adjustGFactor_p::Create(tij, q_cur_g, IMU_Integration, farme_dt, g0);
                        ceres::CostFunction* adjust_g_factor_v = adjustGFactor_v::Create(tij, q_cur_g, IMU_Integration, farme_dt, g0);
                        problem_adj.AddResidualBlock(adjust_g_factor_p, NULL, para_w, para_vi);
                        problem_adj.AddResidualBlock(adjust_g_factor_v, NULL, para_w, para_vi, para_vj);
                        ceres::Solver::Options options_adj;
                        options_adj.linear_solver_type = ceres::DENSE_QR;
                        options_adj.max_num_iterations = 8;
                        options_adj.minimizer_progress_to_stdout = false;
                        ceres::Solver::Summary summary_adj;
                        ceres::Solve(options_adj, &problem_adj, &summary_adj); 
                        Eigen::Vector3d b1,b2;
                        Eigen::Vector3d ee{1,0,0};
                        b1 = (g_init.normalized()).cross(ee);
                        b2 = (g_init.normalized()).cross(b1);
                        g_init = 9.81*g_init.normalized() + para_w[0]*b1 + para_w[1]*b2;
                        if(vi_init.norm()<0.05)
                        {
                            
                            vi_init = Eigen::Vector3d::Zero();
                            vj_init = Eigen::Vector3d::Zero();
                        }
                        Eigen::Vector3d g_w{0,0,9.81};
                        Eigen::Vector3d u = (g_init.cross(g_w)).normalized();
                        double sett = atan2((g_init.cross(g_w)).norm(),g_init.dot(g_w));
                        Eigen::AngleAxisd rotation_vector(sett,u);
                        Eigen::Quaterniond q_init2(rotation_vector);
                        q_init2.normalize();
                        q_body2world = q_init2;                      
                    }
                    IMU_Integration->G = g_init;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 6;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary); 
                    Eigen::Vector3d t_last_tmp, t_last_tmp1, t_last_tmp2;
                    t_last_tmp1 = q_w_curr * t_last_curr;
                    t_last_tmp2 = q_w_curr * t_last_curr_l;
                    t_last_tmp.x() = t_last_tmp2.x();
                    t_last_tmp.y() = t_last_tmp2.y();
                    t_last_tmp.z() = t_last_tmp1.z();
                    t_last_curr = q_w_curr.conjugate()*t_last_tmp;
                    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                    q_w_curr = q_w_curr * q_last_curr;
                    q_w_curr.normalize(); 
                    
                    //重力约束
                    if (USE_IMU&&imuflag==1)
                    {
                        Eigen::Vector3d ypr_w = Utility::R2ypr(q_w_curr.toRotationMatrix());
                        Eigen::Vector3d ypr_i = Utility::R2ypr(IMU.Rwi * R_il);
                        ypr_w.y() = 0.95 * ypr_w.y() + 0.05 * ypr_i.y(); // 由于IMU的pitch和roll角度是可观的，因此用IMU的重力方向再次约束；
                        ypr_w.z() = 0.95 * ypr_w.z() + 0.05 * ypr_i.z();
                        q_w_curr = Utility::ypr2R(ypr_w);
                        q_w_curr.normalize();
                    }
                   

                    //局部地图匹配
                    if(!surroundingCloud.empty())
                    {
                        Eigen::Vector3d ypr_b = Utility::R2ypr((surrounding_q.back()).toRotationMatrix());
                        Eigen::Vector3d ypr_c = Utility::R2ypr(q_w_curr.toRotationMatrix());
                        float dx = (surrounding_t.back()).x() - t_w_curr.x();
			            float dy = (surrounding_t.back()).y() - t_w_curr.y();
			            float dz = (surrounding_t.back()).z() - t_w_curr.z();
			            float dyaw = ypr_b.x() - ypr_c.x();
			            float dpitch = ypr_b.y() - ypr_c.y();
			            float droll = ypr_b.z() - ypr_c.z();
                        if (dyaw > M_PI) dyaw = dyaw - M_PI * 2; 
			            if (dyaw < -M_PI) dyaw = dyaw + M_PI * 2; 
                        if (abs(droll) > keyframeAddingAngle || 
                            abs(dpitch) > keyframeAddingAngle || 
                            abs(dyaw) > keyframeAddingAngle || 
                            sqrt(dx * dx + dy * dy + dz * dz) > keyframeAddingDistance || submapflag<slipwide-1)
                        {
                            surroundingCloud.push_back(transformPointCloud(FullPointsLessFlat,q_w_curr,t_w_curr));
                            surrounding_q.push_back(q_w_curr);
                            surrounding_t.push_back(t_w_curr);
                        }
                    }
                    laserCloudsubmap->clear();
                    if(surroundingCloud.size() > slipwide)
                        {
                            surroundingCloud.pop_front();
                            surrounding_q.pop_front();
                            surrounding_t.pop_front();
                        }
                    if(surroundingCloud.size() > 1)
                    {
                        
                        for(int i = 0; i < surroundingCloud.size(); i++)
                        {
                            surroundingmap->clear();
                            surroundingmap = (transformPointCloud(surroundingCloud[i],q_w_curr.conjugate(),(-1*(q_w_curr.conjugate()*t_w_curr))))->makeShared();
                            *laserCloudsubmap += *surroundingmap;
                        } 
                    }      
                }

                // *laserCloudsubmap += *laserCloudFullRes;
                // surroundingCloud.push_back(laserCloudFullRes);
                // surrounding_q.push_back(q_init);
                // surrounding_t.push_back(t_init);

                nav_msgs::Odometry laserOdometry;
                laserOdometry.header.frame_id = "camera_init";
                laserOdometry.child_frame_id = "laser_odom";
                laserOdometry.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserOdometry.pose.pose.orientation.x = q_w_curr.x(); // q_w_curr
                laserOdometry.pose.pose.orientation.y = q_w_curr.y();
                laserOdometry.pose.pose.orientation.z = q_w_curr.z();
                laserOdometry.pose.pose.orientation.w = q_w_curr.w();
                laserOdometry.pose.pose.position.x = t_w_curr.x(); // t_w_curr
                laserOdometry.pose.pose.position.y = t_w_curr.y();
                laserOdometry.pose.pose.position.z = t_w_curr.z();
                pubLaserOdometry.publish(laserOdometry);

                static tf::TransformBroadcaster br;
                tf::Transform transform;
                tf::Quaternion q;
                transform.setOrigin(tf::Vector3(0,0,0)); // t_w_curr
                q.setW(q_w_curr.w()); // q_w_curr
                q.setX(q_w_curr.x());
                q.setY(q_w_curr.y());
                q.setZ(q_w_curr.z());
                transform.setRotation(q);
                br.sendTransform(tf::StampedTransform(transform, laserOdometry.header.stamp, "world", "laser_odom"));

                

                Eigen::Matrix4d T_lk = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d T_rk = Eigen::Matrix4d::Identity();
                T_lk.block<3, 1>(0, 3) = t_w_curr;
                T_lk.block<3, 3>(0, 0) = Eigen::Matrix3d(q_w_curr);
                T_rk = T_rl * T_lk * T_rl.inverse();
                Eigen::Vector3d t_robot;
                Eigen::Quaterniond q_robot;
                t_robot = T_rk.block<3, 1>(0, 3);
                q_robot = T_rk.block<3, 3>(0, 0);

                geometry_msgs::PoseStamped RobotPose;
                RobotPose.header.stamp = laserOdometry.header.stamp;
                RobotPose.header.frame_id = "world";
                RobotPose.pose.orientation.x = q_robot.x();
                RobotPose.pose.orientation.y = q_robot.y();
                RobotPose.pose.orientation.z = q_robot.z();
                RobotPose.pose.orientation.w = q_robot.w();
                RobotPose.pose.position.x = t_robot.x();
                RobotPose.pose.position.y = t_robot.y();
                RobotPose.pose.position.z = t_robot.z();
                laserPath.header.stamp = laserOdometry.header.stamp;
                laserPath.header.frame_id = "world";
                laserPath.poses.push_back(RobotPose);
                pubLaserPath.publish(laserPath);

                f_save_pose_evo << std::fixed << std::setprecision(6) << laserOdometry.header.stamp.toSec() << " " <<  std::setprecision(9) << t_w_curr.x() << " " << t_w_curr.y() << " " << t_w_curr.z() << " " 
						    << q_w_curr.x() << " " << q_w_curr.y() << " " << q_w_curr.z() << " " << q_w_curr.w() << std::endl;

                // kd-tree赋值
                laserCloudFullLast->clear();
                laserCloudFullLast = laserCloudFullRes->makeShared();
                Eigen::Vector3d ypr_w = Utility::R2ypr(q_last_curr.toRotationMatrix());
                ground_last=ground_cur;
                // 点云发布
                if (frameCount % skipFrameNum == 0)
                {
                    sensor_msgs::PointCloud2 laserCloudCornerMsg;
                    pcl::toROSMsg(*cornerPointsSharp, laserCloudCornerMsg);
                    laserCloudCornerMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                    laserCloudCornerMsg.header.frame_id = "aft_mapped"; // /camera
                    pubLaserCloudCorner.publish(laserCloudCornerMsg);

                    sensor_msgs::PointCloud2 laserCloudSurfMsg;
                    pcl::toROSMsg(*surfPointsFlat, laserCloudSurfMsg);
                    laserCloudSurfMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                    laserCloudSurfMsg.header.frame_id = "aft_mapped";
                    pubLaserCloudSurf.publish(laserCloudSurfMsg);

                    sensor_msgs::PointCloud2 laserCloudFullRes3;
                    pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                    laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                    laserCloudFullRes3.header.frame_id = "aft_mapped";
                    pubLaserCloudFullRes.publish(laserCloudFullRes3);

                    ground_param.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                    ground_param.header.frame_id = "aft_mapped";
                    pubgroundparam.publish(ground_param);

                    sensor_msgs::PointCloud2 laserCloudFullsub;
                    pcl::toROSMsg(*laserCloudsubmap, laserCloudFullsub);
                    laserCloudFullsub.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                    laserCloudFullsub.header.frame_id = "world";
                    pubLaserCloudsub.publish(laserCloudFullsub);
                    boost::format fmt_pcd("%s/%s/%d.%s");
                    pcl::io::savePCDFileASCII((fmt_pcd % saveDirectory % "fullpoints" % frameCount % "pcd").str(), *laserCloudFullRes);
                }


                printf("--laserOdometry_%d: full:%d, full_opt:%d, time: %.1f \n", frameCount,  laserCloudFullResNum,  full_correspondence, t_whole.toc());

                if (t_whole.toc() > 100)
                    ROS_WARN("--laserOdometry: process over 100ms !!!");

                frameCount++;
                if (USE_IMU)
                {
                    delete IMU_Integration;
                }
            }
            std::chrono::milliseconds dura(2);
			std::this_thread::sleep_for(dura);
        }


    }
    
     bool getIMUInterval(double t0, double t1, vector<std::pair<double, Eigen::Vector3d>> & accVector, vector<std::pair<double, Eigen::Vector3d>> & gyrVector)
    {
        accVector.clear();
        gyrVector.clear();

        if (accBuf.empty())
        {
            printf("--laserOdometry-getIMUInterval: not receive imu! \n");
            return false;
        }
        if (t0 <= accBuf.front().first && t1 <= accBuf.front().first)
        {
            printf("--laserOdometry-getIMUInterval: wait for lidar... \n");
            return false;
        }
        
        if (t1 <= accBuf.back().first)//队列
        {
            while (accBuf.front().first <= t0) // 舍弃之前的IMU，只取两帧图像时间之间的IMU数据；
            {
                accBuf.pop();
                gyrBuf.pop();
            }
            while (accBuf.front().first < t1)
            {
                accVector.push_back(accBuf.front());
                accBuf.pop();
                gyrVector.push_back(gyrBuf.front());
                gyrBuf.pop();
            }
            accVector.push_back(accBuf.front());
            gyrVector.push_back(gyrBuf.front());
        }
        else
        {
            printf("--laserOdometry-getIMUInterval: not enough imu... \n");
            printf("t0: %f, t1: %f, imu: %f \n", t0, t1, accBuf.back().first);
            return false;
        }
        return true;
    }

    void IMU_preintegration(double t, double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        delta_q_imu = delta_q_imu * Eigen::Quaterniond(1, gyr(0) * dt / 2, gyr(1) * dt / 2, gyr(2) * dt / 2);
        delta_q_imu.normalize();
    }

    void IMU_preintegration2(double dt, const Eigen::Vector3d &acc_0, const Eigen::Vector3d &acc_1, const Eigen::Vector3d &gyr_0, const Eigen::Vector3d &gyr_1, IntegrationBase *IMU_Int)
    {
        Eigen::Vector3d un_acc_0 = delta_q_imu2 * (acc_0);
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1);
        IMU_Int->delta_q = delta_q_imu2 * Eigen::Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);
        IMU_Int->delta_q.normalize();
        Eigen::Vector3d un_acc_1 = IMU_Int->delta_q * (acc_1);
        Eigen::Vector3d un_acc = 0.5*(un_acc_0 + un_acc_1);
        IMU_Int->delta_p = delta_p_imu + delta_v_imu * dt + 0.5 * un_acc * dt * dt;
        IMU_Int->delta_v = delta_v_imu + un_acc * dt;
        IMU_Int->sum_dt+=dt;
        delta_q_imu2 = IMU_Int->delta_q;
        delta_p_imu = IMU_Int->delta_p;
        delta_v_imu = IMU_Int->delta_v;
    }


    void adjustDistortion()
    {
        // 点云去畸变，并转换到这一帧的时间戳处(即lidar帧最后一个点的坐标系下)
        Eigen::Quaterniond q_last_inverse = q_last_curr.inverse();
        // 这里存在一个假设：在去除点云畸变时认为机器人在lidar扫描一帧的过程中是匀速运动的
        for (int i = 0; i < laserCloudCornerNum; i++)
        {
            double s = 1 - (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
            Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_inverse);
            Eigen::Vector3d t_point_last = s * t_last_curr;
            Eigen::Vector3d point(cornerPointsSharp->points[i].x, cornerPointsSharp->points[i].y, cornerPointsSharp->points[i].z);
            Eigen::Vector3d point_end = q_point_last * (point - t_point_last);
            cornerPointsSharp->points[i].x = point_end.x();
            cornerPointsSharp->points[i].y = point_end.y();
            cornerPointsSharp->points[i].z = point_end.z();
        }

        for (int i = 0; i < laserCloudSurfNum; i++)
        {
            double s = 1 - (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
            Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_inverse);
            Eigen::Vector3d t_point_last = s * t_last_curr;
            Eigen::Vector3d point(surfPointsFlat->points[i].x, surfPointsFlat->points[i].y, surfPointsFlat->points[i].z);
            Eigen::Vector3d point_end = q_point_last * (point - t_point_last);
            surfPointsFlat->points[i].x = point_end.x();
            surfPointsFlat->points[i].y = point_end.y();
            surfPointsFlat->points[i].z = point_end.z();
        }

        for (int i = 0; i < laserCloudFullResNum; i++)
        {
            double s = 1 - (laserCloudFullRes->points[i].intensity - int(laserCloudFullRes->points[i].intensity)) / SCAN_PERIOD;
            Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_inverse);
            Eigen::Vector3d t_point_last = s * t_last_curr;
            Eigen::Vector3d point(laserCloudFullRes->points[i].x, laserCloudFullRes->points[i].y, laserCloudFullRes->points[i].z);
            Eigen::Vector3d point_end = q_point_last * (point - t_point_last);
            laserCloudFullRes->points[i].x = point_end.x();
            laserCloudFullRes->points[i].y = point_end.y();
            laserCloudFullRes->points[i].z = point_end.z();
        }
    }

    void TransformToStart(PointType const *const pi, PointType *const po)
    {
        // 使用当前优化得到的q、t将当前lidar点云向上一帧点云坐标系下转化
        Eigen::Vector3d point(pi->x, pi->y, pi->z);
        Eigen::Vector3d un_point = q_last_curr * point + t_last_curr;

        po->x = un_point.x();
        po->y = un_point.y();
        po->z = un_point.z();
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Quaterniond q, Eigen::Vector3d t)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

            PointType *pointFrom;
            int cloudSize = cloudIn->size();
            cloudOut->resize(cloudSize);
            
            for (int i = 0; i < cloudSize; ++i)
            {
                pointFrom = &cloudIn->points[i];
                Eigen::Vector3d point_curr(pointFrom->x, pointFrom->y, pointFrom->z);
                Eigen::Vector3d point_w = q * point_curr + t;
                cloudOut->points[i].x = point_w.x();
                cloudOut->points[i].y = point_w.y();
                cloudOut->points[i].z = point_w.z();
                cloudOut->points[i].intensity = pointFrom->intensity;
            }
            return cloudOut;
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    vg_ICP vg_icp;

    ros::spin();
    return 0;
}