#include "lidarFactor.hpp"
#include "rgc_slam/utility.h" 
#include "rgc_slam/tic_toc.h"

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);
Eigen::Quaterniond q_histoary(1, 0, 0, 0);
Eigen::Vector3d t_histoary(0, 0, 0);
Eigen::Quaterniond q_w_curr_f(1, 0, 0, 0);
Eigen::Quaterniond q_w_curr_f2(1, 0, 0, 0);
Eigen::Quaterniond q_w_curr_delta(1, 0, 0, 0);
Eigen::Vector3d t_w_curr_delta(0, 0, 0);
Eigen::Quaterniond q_w_last2(1, 0, 0, 0);
Eigen::Vector3d t_w_last2(0, 0, 0);
ground_s g_w_curr_delta;
ground_s ground_last2;

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_w_curr(para_t);
double para_q_last[4] = {0, 0, 0, 1};
double para_t_last[3] = {0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_last(para_q_last);
Eigen::Map<Eigen::Vector3d> t_w_last(para_t_last);



Mid_Filter accx_MF(201), accy_MF(41), accz_MF(41);

class LaserMapping
{
  public:
	ros::NodeHandle nh;

	ros::Subscriber subLaserCloudFullRes;
	ros::Subscriber subLaserCloudCorner;
	ros::Subscriber subLaserCloudSurf;
	ros::Subscriber subLaserOdometry;
	ros::Subscriber subgroundparam;
	
	ros::Subscriber sub_imu;
	ros::Subscriber sub_scan;
	
	ros::Publisher pubLaserCloudSurround;
	ros::Publisher pubLaserCloudMap;
	ros::Publisher pubScanCloudMap;

	ros::Publisher pubOdomAftMapped;
	ros::Publisher pubOdomAftMappedHighFrec;
	ros::Publisher pubLaserAfterMappedPath;
	ros::Publisher pubLoamPath;
	ros::Publisher pubaLoamRobotPath;
	ros::Publisher pubGlobalPoseGraphPath;

	ros::Publisher pubMarker;
	ros::Publisher pubLoopConstraintEdge;
	ros::Publisher pubGlobalPoseGraphPoint;
	ros::Publisher pubGlobalMapKeyPose;
	ros::Publisher pubGlobalMapKeyPoseDS;
	ros::Publisher pubRobotPose;

	std::mutex mBuf;

	queue<std::pair<double, imu_s>> IMUBuf;
	queue<std::pair<double, Eigen::Vector3d>> accBuf;
	queue<std::pair<double, Eigen::Vector3d>> gyrBuf;
	queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
	queue<sensor_msgs::PointCloud2ConstPtr> laserCornerBuf;
	queue<sensor_msgs::PointCloud2ConstPtr> laserSurfBuf;
	queue<sensor_msgs::PointCloud2ConstPtr> scanBuf;
	queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
	queue<ground_msg::groundparam> groundBuf;

	ground_msg::groundparam ground_param;

	Eigen::Matrix3d R_rl, R_il;
	Eigen::Vector3d t_rl;
	Eigen::Matrix4d T_rl;
	imu_s IMU, IMUTemp, IMULast;
	ground_s ground_cur, ground_last, ground_mean, ground_histoary;
	vector<std::pair<double, Eigen::Vector3d>> accVector, gyrVector;
	Eigen::Quaterniond delta_q_imu;

	double timeLaserCloudFullRes = 0;
	double timeLaserCloudCorner = 0;
	double timeLaserCloudSurf = 0;
	double timeLaserOdometry = 0;
	double timeLaserground = 0;
	double timeScan = 0;
	double prevTime = 0, curTime = 0;

	pcl::PointCloud<PointType>::Ptr lidarScan;
	pcl::PointCloud<PointType2>::Ptr laserCloudCorner;
    pcl::PointCloud<PointType2>::Ptr laserCloudSurf;
	pcl::PointCloud<PointType>::Ptr lidarScanDS, lidarScanLastDS;
    pcl::PointCloud<PointType2>::Ptr laserCloudCornerDS, laserCloudCornerLastDS;
    pcl::PointCloud<PointType2>::Ptr laserCloudSurfDS, laserCloudSurfLastDS;
	pcl::PointCloud<PointType2>::Ptr surroundingMapDS;
    pcl::PointCloud<PointType2>::Ptr laserCloudCornerFromMap;
	pcl::PointCloud<PointType2>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType2>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType2>::Ptr laserCloudSurfFromMapDS;
	int laserCloudCornerDSNum = 0, laserCloudCornerLastDSNum = 0;
    int laserCloudSurfDSNum = 0, laserCloudSurfLastDSNum = 0;
	int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;

	std::mutex mKeyframe;

	pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D, copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D, copy_cloudKeyPoses6D;
	std::map<int, pcl::PointCloud<PointType2>::Ptr> cornerCloudKeyFrames;
    std::map<int, pcl::PointCloud<PointType2>::Ptr> surfCloudKeyFrames;
    std::map<int, pcl::PointCloud<PointType>::Ptr> scanCloudKeyFrames;
	deque<pcl::PointCloud<PointType2>::Ptr> surroundingCornerCloud;
	deque<pcl::PointCloud<PointType2>::Ptr> surroundingSurfCloud;
	vector<int> surroundingExistingKeyPoseID;
	
    deque<Eigen::Quaterniond> histoary_pose;
	deque<Eigen::Vector3d> histoary_trans;
	deque<ground_s> histoary_plane;

	int initfirst = 0;//2
	int initskipframe = 0;//1 20

	int keyFrameNum = 0;
	float keyframeAddingDistance = 0.5; // 0.5m
    float keyframeAddingAngle = 0.3; // 57.3*0.2 degree 
    float surroundingKeyframeDensity = 0.3; // m
    float surroundingKeyframeSearchRadius = 15; // m
	int mapping_count = 0, laserOdom_count = 0;
	double cost_mapping = 0;

	int changegroundflag = 25;
    int gflag = 0;
	
	Eigen::Vector3d t_robot;
	Eigen::Quaterniond q_robot;

	pcl::PointCloud<PointType2>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType2>::Ptr nearHistoryKeyFrameCloud;
	std::map<int, PointTypePose> KeyPose6D;
	std::map<int, float> travel_distance; // m
	std::map<int, float> travel_angle; // rad
	std::map<int, PointTypePose> correctedKeyPose6DByLoop;

	int loopClosureCount = 0;
	bool bnewKeyFrame = false;
	bool bLoopIsClosed = false;
	bool bKeyFramePoseGraphUpdated = false;
	float poseGraphSearchRadius = 15;
	float historyKeyframeSearchRadius = 5; // meters, key frame that is within n meters from current pose will be considerd for loop closure
	float historyKeyframeFitnessScore = 0.1; // icp threshold, the smaller the better alignment
    int historyKeyframeSearchNum = 50; // number of hostory key frames will be fused into a submap for loop closure
	float DistanceByLoop = 0;
	float DRIFT_FACTOR = 0.02; 
	float loopKeyframeDisDiff = 20;
	
	bool bGNSSIsAdded = false;

    Eigen::Affine3f T_Drift;
	Eigen::Vector3f t_drift;
	Eigen::Quaternionf q_drift;

	typedef struct
	{
		int key_curr;
		int key_loop;
		PointTypePose keyPose6DCurr;
		PointTypePose keyPose6DLoop;
		Eigen::Vector3f t_loop_curr;
		Eigen::Quaternionf q_loop_curr;
		float noise;
	} loopInfo;

	std::map<int, loopInfo> loopInfoContainer;
	std::vector<std::pair<int, int> > currLoopKeyContainer;

	pcl::KdTreeFLANN<PointType2>::Ptr kdtreeCornerFromMap;
	pcl::KdTreeFLANN<PointType2>::Ptr kdtreeSurfFromMap;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
	pcl::VoxelGrid<PointType2> downSizeFilterCorner;
    pcl::VoxelGrid<PointType2> downSizeFilterSurf;
    pcl::VoxelGrid<PointType2> downSizeFilterICP;
	pcl::VoxelGrid<PointType> downSizeFilterScan;

	float lineResolution = 0, planeResolution = 0;
	float globalMapDensity = 0.4;
	float globalMapVisualizationPoseDensity = 0.5;
	int map_update = 1;
	int loaded_map_size = 0;
	float init_x = 0, init_y = 0, init_z = 0, init_yaw = 0;
	double deg2rad = M_PI / 180.0, rad2deg = 180.0 / M_PI;

	nav_msgs::Path laserAfterMappedPath;
	nav_msgs::Path aloam_robot_Path;
	nav_msgs::Path globalPoseGraphPath;

	std::ofstream f_save_pose;
	std::ofstream f_save_pose_evo;
	std::ifstream f_read_pose;
	std::string saveDirectory = "/home/robot220/code_ws/catkin_ws_robot_navigation/database";

	std::thread LaserMappingThread;
	std::thread PoseGraphThread;
	
	// SCManager scManager;

	LaserMapping()
    {
		nh.param<float>("mapping_line_resolution", lineResolution, 0.2);
		nh.param<float>("mapping_plane_resolution", planeResolution, 0.4);
		printf("--LaserMapping: line resolution: %.2f, plane resolution: %.2f \n", lineResolution, planeResolution);

		nh.param<int>("USE_IMU", USE_IMU, 1);
		nh.param<int>("USE_GROUND2", USE_GROUND, 1);
		nh.param<int>("LoopClosureEnable", LoopClosureEnable, 1);
		printf("--LaserMapping: USE_IMU: %d \n", USE_IMU);
		printf("--LaserMapping: USE_GROUND: %d \n", USE_GROUND);
		printf("--LaserMapping: LoopClosureEnable: %d \n", LoopClosureEnable);

		nh.param<float>("keyframeAddingDistance", keyframeAddingDistance, 0.5);
		nh.param<float>("keyframeAddingAngle", keyframeAddingAngle, 0.2);
		nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 15);
		nh.param<float>("globalMapDensity", globalMapDensity, 0.4);
		
		printf("--LaserMapping: keyframeAddingDistance: %.1f \n", keyframeAddingDistance);
        printf("--LaserMapping: keyframeAddingAngle: %.1f \n", keyframeAddingAngle);
		printf("--LaserMapping: surroundingKeyframeSearchRadius: %.1f \n", surroundingKeyframeSearchRadius);
		printf("--LaserMapping: globalMapDensity: %.1f \n", globalMapDensity);

		nh.param<string>("saveDirectory", saveDirectory, "/home/robot220/code_ws/catkin_ws_robot_navigation/database");
		printf("--LaserMapping: saveDirectory: %s \n", saveDirectory.c_str());
		nh.param<int>("map_update", map_update, 1); // default: update map

		printf("--LaserMapping: map_update: %d \n", map_update);

		nh.param<float>("init_x", init_x, 0);
        nh.param<float>("init_y", init_y, 0);
        nh.param<float>("init_z", init_z, 0);
        nh.param<float>("init_yaw", init_yaw, 0);
		printf("--LaserMapping: init_x %f \n", init_x);
        printf("--LaserMapping: init_y %f \n", init_y);
        printf("--LaserMapping: init_z %f \n", init_z);
        printf("--LaserMapping: init_yaw %f \n", init_yaw);

		subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 10, &LaserMapping::laserCloudFullResHandler, this);
		subLaserCloudCorner = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner", 10, &LaserMapping::laserCloudCornerHandler, this);
		subLaserCloudSurf = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf", 10, &LaserMapping::laserCloudSurfHandler, this);
		subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 10, &LaserMapping::laserOdometryHandler, this);
		subgroundparam = nh.subscribe("/ground_param_gable", 200, &LaserMapping::groundHandler, this);

		sub_imu = nh.subscribe("/mynteye/imu/data_raw", 200, &LaserMapping::imu_callback, this);///imu_raw /kitti/oxts/imu/extract
		sub_scan = nh.subscribe("/cloud_scan", 20, &LaserMapping::lidar_scan_callback, this);

		pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 10);
		pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 10);
		pubScanCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map_scan", 10);

		pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
		pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/odom", 10);
		pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 10);
		pubLoamPath = nh.advertise<nav_msgs::Path>("/loam_path", 10);
		pubaLoamRobotPath = nh.advertise<nav_msgs::Path>("/aloam_robot_path", 10);
		pubGlobalPoseGraphPath = nh.advertise<nav_msgs::Path>("/global_pose_graph_path", 10);

		pubMarker = nh.advertise<visualization_msgs::Marker>("/robot_marker", 10);
		pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 10);
		pubGlobalPoseGraphPoint = nh.advertise<sensor_msgs::PointCloud2>("/global_pose_graph_point", 10);
		pubGlobalMapKeyPose = nh.advertise<sensor_msgs::PointCloud2>("/global_map_keyPose", 10);
		pubGlobalMapKeyPoseDS = nh.advertise<sensor_msgs::PointCloud2>("/global_map_keyPoseDS", 10);
		pubRobotPose = nh.advertise<geometry_msgs::PoseStamped>("/robot_pose", 10);

		R_il = Utility::ypr2R(Eigen::Vector3d(-1.29, -0.15, 0.65)); // 将IMU和激光雷达与重力对齐；
		// R_il = Utility::ypr2R(Eigen::Vector3d(179.603, -179.624, -179.937)); 
		// Eigen::Quaterniond qq_li(0.999479, -0.000357, 0.0194128, 0.0257984); // 将IMU和激光雷达与重力齐；
		// R_il = qq_li.inverse();
		t_rl = Eigen::Vector3d(0.68, 0, 0.34);
		R_rl = Utility::ypr2R(Eigen::Vector3d(0.0, 0.0, -0.0));
		T_rl = Eigen::Matrix4d::Identity();
		T_rl.block<3, 3>(0, 0) = R_rl;
		T_rl.block<3, 1>(0, 3) = t_rl;

		allocateMemory();

		downSizeFilterCorner.setLeafSize(lineResolution, lineResolution,lineResolution);
		downSizeFilterSurf.setLeafSize(planeResolution, planeResolution, planeResolution);
		downSizeFilterICP.setLeafSize(lineResolution, lineResolution, lineResolution);
		
		boost::format fmt_pose("%s/%s");
		f_read_pose.open((fmt_pose % saveDirectory % "pose.txt").str(), std::fstream::in);

		if(map_update == 1)
		{
			f_save_pose.open((fmt_pose % saveDirectory % "pose.txt").str(), std::fstream::out);
			f_save_pose_evo.open((fmt_pose % saveDirectory % "pose_evo.txt").str(), std::fstream::out);
		}

		LaserMappingThread = std::thread(&LaserMapping::LaserMapping_thread, this);
		PoseGraphThread = std::thread(&LaserMapping::poseGraphOptimizationThread, this);
	}

	 ~LaserMapping()
    {
		saveKeyPoseToFileAsTUM();
		f_save_pose.close();
		f_save_pose_evo.close();
        printf("--LaserMapping exit !!! \n");
    }

	void allocateMemory()
    {
		lidarScan.reset(new pcl::PointCloud<PointType>());
		laserCloudCorner.reset(new pcl::PointCloud<PointType2>());
        laserCloudSurf.reset(new pcl::PointCloud<PointType2>());
		lidarScanDS.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerDS.reset(new pcl::PointCloud<PointType2>());
        laserCloudSurfDS.reset(new pcl::PointCloud<PointType2>());
		lidarScanLastDS.reset(new pcl::PointCloud<PointType>());
		laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType2>());
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType2>());
		
		surroundingMapDS.reset(new pcl::PointCloud<PointType2>());
		laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType2>());
		laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType2>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType2>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType2>());

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
		
		latestKeyFrameCloud.reset(new pcl::PointCloud<PointType2>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType2>());

		kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType2>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType2>());
		kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
	}

	void saveKeyPoseToFileAsTUM()
	{
		// 将机器人位姿按照TUM数据集格式写入文件；
		int keySize = cloudKeyPoses6D->size();
		for (int i = 0; i < keySize; ++i)
		{
			PointTypePose KeyPose6D = cloudKeyPoses6D->points[i];
			if (map_update == 1)
			{
				f_save_pose << std::fixed << KeyPose6D.intensity << " " << KeyPose6D.x << " " << KeyPose6D.y << " " << KeyPose6D.z << " " 
						    << KeyPose6D.yaw << " " << KeyPose6D.pitch << " " << KeyPose6D.roll << " " << KeyPose6D.time << " " 
							<< travel_distance[i] << " " << travel_angle[i] << std::endl;
				// tf::Quaternion q = tf::createQuaternionFromRPY(KeyPose6D.roll, KeyPose6D.pitch, KeyPose6D.yaw);
				// f_save_pose_evo << std::fixed << KeyPose6D.time << " " << KeyPose6D.x << " " << KeyPose6D.y << " " << KeyPose6D.z << " " 
				// 		    << q.x() << " " << q.y()<< " " << q.z() << " " << q.w() << std::endl;
			}
		}
		for (int i = 0; i<globalPoseGraphPath.poses.size();i++)
		{
			if (map_update == 1)
			{
				f_save_pose_evo << std::fixed << std::setprecision(6) << globalPoseGraphPath.poses[i].header.stamp.toSec() << " " <<  std::setprecision(9) << globalPoseGraphPath.poses[i].pose.position.x << " " << globalPoseGraphPath.poses[i].pose.position.y << " " << globalPoseGraphPath.poses[i].pose.position.z << " " 
						    << globalPoseGraphPath.poses[i].pose.orientation.x << " " << globalPoseGraphPath.poses[i].pose.orientation.y << " " << globalPoseGraphPath.poses[i].pose.orientation.z << " " << globalPoseGraphPath.poses[i].pose.orientation.w << std::endl;
			}
		}
	}

	// 回调函数
	void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullResMsg)
	{
		mBuf.lock();
		fullResBuf.push(laserCloudFullResMsg);
		mBuf.unlock();
	}

	void laserCloudCornerHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerMsg)
	{
		mBuf.lock();
		laserCornerBuf.push(laserCloudCornerMsg);
		mBuf.unlock();
	}

	void laserCloudSurfHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfMsg)
	{
		mBuf.lock();
		laserSurfBuf.push(laserCloudSurfMsg);
		mBuf.unlock();
	}

	void groundHandler(const ground_msg::groundparam & ground_msg)
    {
        mBuf.lock();
        groundBuf.push(ground_msg);
        mBuf.unlock();
    }

	void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
	{
		mBuf.lock();
		odometryBuf.push(laserOdometry);
		mBuf.unlock();

		laserOdom_count++;

		// high frequence publish
		Eigen::Quaterniond q_wodom_curr;
		Eigen::Vector3d t_wodom_curr;
		q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
		q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
		q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
		q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
		t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
		t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
		t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

		Eigen::Quaterniond q_w_new = q_wmap_wodom * q_wodom_curr;
		Eigen::Vector3d t_w_new = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

		nav_msgs::Odometry odomAftMapped;
		odomAftMapped.header.frame_id = "camera_init";
		odomAftMapped.child_frame_id = "aft_mapped";
		odomAftMapped.header.stamp = laserOdometry->header.stamp;
		odomAftMapped.pose.pose.orientation.x = q_w_new.x();
		odomAftMapped.pose.pose.orientation.y = q_w_new.y();
		odomAftMapped.pose.pose.orientation.z = q_w_new.z();
		odomAftMapped.pose.pose.orientation.w = q_w_new.w();
		odomAftMapped.pose.pose.position.x = t_w_new.x();
		odomAftMapped.pose.pose.position.y = t_w_new.y();
		odomAftMapped.pose.pose.position.z = t_w_new.z();
		pubOdomAftMappedHighFrec.publish(odomAftMapped);

		// if (USE_IMU)
		// {
		// 	printf("--IMU_ypr_%d: %.3f  %.3f  %.3f\n", IMU.count, IMU.yaw*57.3, IMU.pitch*57.3, IMU.roll*57.3);
		// }

		// printf("--Mapping: corner:%d, surf:%d; map-corner:%d, surf:%d; KF:%d; cost:%.1f \n",
		// laserCloudCornerDSNum, laserCloudSurfDSNum, laserCloudCornerFromMapDSNum, laserCloudSurfFromMapDSNum, 
		// keyFrameNum, cost_mapping);

		// lidar to robot frame
		Eigen::Vector3d ypr;
		Eigen::Matrix3d r_robot;
		Eigen::Matrix4d T_lk = Eigen::Matrix4d::Identity();
		Eigen::Matrix4d T_rk = Eigen::Matrix4d::Identity();
		T_lk.block<3, 1>(0, 3) = t_w_new;
		T_lk.block<3, 3>(0, 0) = Eigen::Matrix3d(q_w_new);
		T_rk = T_rl * T_lk * T_rl.inverse();
		t_robot = T_rk.block<3, 1>(0, 3);
		r_robot = T_rk.block<3, 3>(0, 0);
		q_robot = Eigen::Quaterniond(r_robot);

		// T_lk.block<3, 1>(0, 3) = t_wodom_curr;
		// T_lk.block<3, 3>(0, 0) = Eigen::Matrix3d(q_wodom_curr);
		// T_rk = T_rl * T_lk * T_rl.inverse();
		// Eigen::Vector3d t_odom_robot = T_rk.block<3, 1>(0, 3);
		// Eigen::Quaterniond q_odom_robot = Eigen::Quaterniond(T_rk.block<3, 3>(0, 0));

		// // 2021.06.19
		// ypr = Utility::R2ypr(Eigen::Matrix3d(q_w_new));
		// printf("--lidar_pose_%d : x:%.3f y:%.3f z:%.3f , y:%.2f p:%.2f r:%.2f, time:%.1f \n",
		// 	mapping_count, t_w_new(0),t_w_new(1),t_w_new(2),  ypr(0), ypr(1), ypr(2), cost_mapping);

		// ypr = Utility::R2ypr(r_robot);
		// printf("--robot_pose_%d : x:%.3f y:%.3f z:%.3f , y:%.2f p:%.2f r:%.2f \n\n",
		// 	mapping_count, t_robot(0),t_robot(1),t_robot(2),  ypr(0), ypr(1), ypr(2));
		// // 2021.06.19

		// ypr = Utility::R2ypr(Eigen::Matrix3d(q_wodom_curr));
		// printf("--odom_lidar_%d : x:%.3f y:%.3f z:%.3f , y:%.2f p:%.2f r:%.2f \n",
		// 	laserOdom_count, t_wodom_curr(0),t_wodom_curr(1),t_wodom_curr(2),  ypr(0), ypr(1), ypr(2));

		// ypr = Utility::R2ypr(Eigen::Matrix3d(q_odom_robot));
		// printf("--odom_robot_%d : x:%.3f y:%.3f z:%.3f , y:%.2f p:%.2f r:%.2f \n",
		// 	laserOdom_count, t_odom_robot(0),t_odom_robot(1),t_odom_robot(2),  ypr(0), ypr(1), ypr(2));

		static tf::TransformBroadcaster br_mapped;
		tf::Transform transform;
		tf::Quaternion q;
		transform.setOrigin(tf::Vector3(t_robot(0),t_robot(1),t_robot(2)));
		q.setW(q_robot.w());
		q.setX(q_robot.x());
		q.setY(q_robot.y());
		q.setZ(q_robot.z());
		transform.setRotation(q);
		br_mapped.sendTransform(tf::StampedTransform(transform, laserOdometry->header.stamp, "/camera_init", "/robot"));

		// static tf::TransformBroadcaster bl_mapped;
		// tf::Transform transform_l;
		// tf::Quaternion q_l;
		// transform_l.setOrigin(tf::Vector3(t_w_new(0),t_w_new(1),t_w_new(2)));
		// q_l.setW(q_w_new.w());
		// q_l.setX(q_w_new.x());
		// q_l.setY(q_w_new.y());
		// q_l.setZ(q_w_new.z());
		// transform_l.setRotation(q_l);
		// bl_mapped.sendTransform(tf::StampedTransform(transform_l, laserOdometry->header.stamp, "/camera_init", "/aft_mapped"));

		if (1)
		{
			static tf::TransformBroadcaster br_laser_to_world;
			tf::Transform transform_laser_to_world;
			tf::Quaternion q_laser_to_world;
			transform_laser_to_world.setOrigin(tf::Vector3(0,0,0));
			q_laser_to_world.setW(1.0);
			q_laser_to_world.setX(0);
			q_laser_to_world.setY(0);
			q_laser_to_world.setZ(0);
			transform_laser_to_world.setRotation(q_laser_to_world);
			br_laser_to_world.sendTransform(tf::StampedTransform(transform_laser_to_world, laserOdometry->header.stamp, "/world", "/camera_init"));

			static tf::TransformBroadcaster br_world_to_odom;
			tf::Transform transform_world_to_odom;
			tf::Quaternion q_world_to_odom;
			transform_world_to_odom.setOrigin(tf::Vector3(0,0,0));
			q_world_to_odom.setW(1.0);
			q_world_to_odom.setX(0);
			q_world_to_odom.setY(0);
			q_world_to_odom.setZ(0);
			transform_world_to_odom.setRotation(q_world_to_odom);
			br_world_to_odom.sendTransform(tf::StampedTransform(transform_world_to_odom, laserOdometry->header.stamp, "/odom", "/world"));

			static tf::TransformBroadcaster br_odom_to_map;
			tf::Transform transform_odom_to_map;
			tf::Quaternion q_odom_to_map;
			transform_odom_to_map.setOrigin(tf::Vector3(0,0,0));
			q_odom_to_map.setW(1.0);
			q_odom_to_map.setX(0);
			q_odom_to_map.setY(0);
			q_odom_to_map.setZ(0);
			transform_odom_to_map.setRotation(q_odom_to_map);
			br_odom_to_map.sendTransform(tf::StampedTransform(transform_odom_to_map, laserOdometry->header.stamp, "/map", "/odom"));


			static tf::TransformBroadcaster br_mapping_to_robot;
			tf::Transform transform_mapping_to_robot;
			tf::Quaternion q_mapping_to_robot;
			transform_mapping_to_robot.setOrigin(tf::Vector3(t_rl.x(),0,t_rl.z()));
			q_mapping_to_robot.setW(1.0);
			q_mapping_to_robot.setX(0);
			q_mapping_to_robot.setY(0);
			q_mapping_to_robot.setZ(0);
			transform_mapping_to_robot.setRotation(q_mapping_to_robot);
			br_mapping_to_robot.sendTransform(tf::StampedTransform(transform_mapping_to_robot, laserOdometry->header.stamp, "/robot", "/aft_mapped"));

			static tf::TransformBroadcaster br_base_to_robot;
			tf::Transform transform_base_to_robot;
			tf::Quaternion q_base_to_robot;
			transform_base_to_robot.setOrigin(tf::Vector3(0,0,0));
			q_base_to_robot.setW(1.0);
			q_base_to_robot.setX(0);
			q_base_to_robot.setY(0);
			q_base_to_robot.setZ(0);
			transform_base_to_robot.setRotation(q_base_to_robot);
			br_base_to_robot.sendTransform(tf::StampedTransform(transform_base_to_robot, laserOdometry->header.stamp, "/robot", "/base_link"));

			static tf::TransformBroadcaster br_velodyne_to_robot;
			tf::Transform transform_velodyne_to_robot;
			tf::Quaternion q_velodyne_to_robot;
			transform_velodyne_to_robot.setOrigin(tf::Vector3(0,0,0));
			q_velodyne_to_robot.setW(1.0);
			q_velodyne_to_robot.setX(0);
			q_velodyne_to_robot.setY(0);
			q_velodyne_to_robot.setZ(0);
			transform_velodyne_to_robot.setRotation(q_velodyne_to_robot);
			br_velodyne_to_robot.sendTransform(tf::StampedTransform(transform_velodyne_to_robot, laserOdometry->header.stamp, "/aft_mapped", "/velodyne"));

			static tf::TransformBroadcaster br_laser_to_velodyne;
			tf::Transform transform_laser_to_velodyne;
			tf::Quaternion q_laser_to_velodyne;
			transform_laser_to_velodyne.setOrigin(tf::Vector3(0,0,0));
			q_laser_to_velodyne.setW(1.0);
			q_laser_to_velodyne.setX(0);
			q_laser_to_velodyne.setY(0);
			q_laser_to_velodyne.setZ(0);
			transform_laser_to_velodyne.setRotation(q_laser_to_velodyne);
			br_laser_to_velodyne.sendTransform(tf::StampedTransform(transform_laser_to_velodyne, laserOdometry->header.stamp, "/velodyne", "/base_laser"));

			static tf::TransformBroadcaster br_base_to_footprint;
			tf::Transform transform_base_to_footprint;
			tf::Quaternion q_base_to_footprint;
			transform_base_to_footprint.setOrigin(tf::Vector3(0,0,0));
			q_base_to_footprint.setW(1.0);
			q_base_to_footprint.setX(0);
			q_base_to_footprint.setY(0);
			q_base_to_footprint.setZ(0);
			transform_base_to_footprint.setRotation(q_base_to_footprint);
			br_base_to_footprint.sendTransform(tf::StampedTransform(transform_base_to_footprint, laserOdometry->header.stamp, "/base_link", "/base_footprint"));
		}

		geometry_msgs::PoseStamped laserAfterMappedPose;
		laserAfterMappedPose.header.stamp = laserOdometry->header.stamp;
		laserAfterMappedPose.header.frame_id = "camera_init";
		laserAfterMappedPose.pose.orientation.x = q_w_new.x();
		laserAfterMappedPose.pose.orientation.y = q_w_new.y();
		laserAfterMappedPose.pose.orientation.z = q_w_new.z();
		laserAfterMappedPose.pose.orientation.w = q_w_new.w();
		laserAfterMappedPose.pose.position.x = t_w_new.x();
		laserAfterMappedPose.pose.position.y = t_w_new.y();
		laserAfterMappedPose.pose.position.z = t_w_new.z();
		laserAfterMappedPath.header.stamp = laserOdometry->header.stamp;
		laserAfterMappedPath.header.frame_id = "camera_init";
		laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
		pubLaserAfterMappedPath.publish(laserAfterMappedPath);
		
		geometry_msgs::PoseStamped RobotPose;
		RobotPose.header.stamp = laserOdometry->header.stamp;
		RobotPose.header.frame_id = "world";
		RobotPose.pose.orientation.x = q_robot.x();
		RobotPose.pose.orientation.y = q_robot.y();
		RobotPose.pose.orientation.z = q_robot.z();
		RobotPose.pose.orientation.w = q_robot.w();
		RobotPose.pose.position.x = t_robot.x();
		RobotPose.pose.position.y = t_robot.y();
		RobotPose.pose.position.z = t_robot.z();
		aloam_robot_Path.header.stamp = laserOdometry->header.stamp;
		aloam_robot_Path.header.frame_id = "world";
		aloam_robot_Path.poses.push_back(RobotPose);
		pubaLoamRobotPath.publish(aloam_robot_Path);

		pubRobotPose.publish(RobotPose);

		PubRobotMarker(t_robot, q_robot);
		// PubRobotMarker(t_w_new, q_w_new);
	}

	void imu_callback(const sensor_msgs::Imu & imu_msg)
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

		// if(get_init_imu_bias(acc, gyr, IMU) != 1)
		// {
		// 	return;
		// }

		Eigen::Vector3d acc_new, gyr_new;
		acc_new = acc - IMU.ba;
		gyr_new = gyr - IMU.bg;

		IMU.t = t;
		IMU.ax = acc_new.x();
		IMU.ay = acc_new.y();
		IMU.az = acc_new.z();
		IMU.gx = gyr_new.x();
		IMU.gy = gyr_new.y();
		IMU.gz = gyr_new.z();
		IMU.count++;
		ComplementaryFilter(IMU);

		mBuf.lock();
		accBuf.push(std::make_pair(t, acc_new));
		gyrBuf.push(std::make_pair(t, gyr_new));
		IMUBuf.push(std::make_pair(t, IMU));
		mBuf.unlock();

		// printf("--imu_callback_%d | IMU_ypr: %.3f  %.3f  %.3f , t: %.3f \n", IMU.count, IMU.yaw*57.3, IMU.pitch*57.3, IMU.roll*57.3, t);
	}

	void lidar_scan_callback(const sensor_msgs::PointCloud2ConstPtr &lidarScan_msg)
	{
		mBuf.lock();
		scanBuf.push(lidarScan_msg);
		mBuf.unlock();
	}

	void PubRobotMarker(Eigen::Vector3d p, Eigen::Quaterniond q)
    {
        // uint8 ARROW=0
        // uint8 CUBE=1
        // uint8 SPHERE=2
        // uint8 CYLINDER=3
        // uint8 LINE_STRIP=4
        // uint8 LINE_LIST=5
        // uint8 CUBE_LIST=6
        // uint8 SPHERE_LIST=7
        // uint8 POINTS=8
        // uint8 TEXT_VIEW_FACING=9
        // uint8 MESH_RESOURCE=10
        // uint8 TRIANGLE_LIST=11

        visualization_msgs::Marker tempMarker;
        tempMarker.id = 0;

        tempMarker.header.frame_id = "world";
        tempMarker.header.stamp = ros::Time::now();
        tempMarker.id = 0;
        tempMarker.ns = "GR-loam";
        tempMarker.type = visualization_msgs::Marker::CUBE;
        tempMarker.action = visualization_msgs::Marker::ADD;

        tempMarker.pose.position.x = p(0);
        tempMarker.pose.position.y = p(1);
        tempMarker.pose.position.z = p(2);
        tempMarker.pose.orientation.x = q.x();
        tempMarker.pose.orientation.y = q.y();
        tempMarker.pose.orientation.z = q.z();
        tempMarker.pose.orientation.w = q.w();

        tempMarker.scale.x = 1.2;
        tempMarker.scale.y = 0.7;
        tempMarker.scale.z = 0.5;
 
        tempMarker.color.r = 1.0f;
        tempMarker.color.g = 0.0f;
        tempMarker.color.b = 0.0f;
        tempMarker.color.a = 1.0;
 
        tempMarker.lifetime = ros::Duration();

        pubMarker.publish(tempMarker);
    }
	
	// 一阶互补滤波
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

		// 对原始测量进行中值滤波，经测试，中值滤波比其他滤波效果好；
		t_imu.ax = accx_MF.MFilter(t_imu.ax);
		t_imu.ay = accy_MF.MFilter(t_imu.ay);
		t_imu.az = accz_MF.MFilter(t_imu.az);

		if (t_imu.count < 300) // 1s
        {
            t_imu.k = 0.9; // 快速收敛；
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
                    for (int i = 50; i < filter_size - 50; i++) // 除去前后100个数据；
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

					// 将机器人静止放置在尽量水平的地面，计算当前IMU的水平和俯仰角度；
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

	// Mapping线程
	void LaserMapping_thread()
	{
        printf("--LaserMapping_thread begain !!! \n");

		while (1)
		{
			while (!laserCornerBuf.empty() && !laserSurfBuf.empty() && !fullResBuf.empty() && !odometryBuf.empty() && !scanBuf.empty() && !groundBuf.empty())
			{
				mBuf.lock();

				while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < laserCornerBuf.front()->header.stamp.toSec())
					odometryBuf.pop();
				if (odometryBuf.empty())
				{
					mBuf.unlock();
					break;
				}

				while (!laserSurfBuf.empty() && laserSurfBuf.front()->header.stamp.toSec() < laserCornerBuf.front()->header.stamp.toSec())
					laserSurfBuf.pop();
				if (laserSurfBuf.empty())
				{
					mBuf.unlock();
					break;
				}

				while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < laserCornerBuf.front()->header.stamp.toSec())
					fullResBuf.pop();
				if (fullResBuf.empty())
				{
					mBuf.unlock();
					break;
				}

				while (!scanBuf.empty() && scanBuf.front()->header.stamp.toSec() < laserCornerBuf.front()->header.stamp.toSec())
					scanBuf.pop();
				if (scanBuf.empty())
				{
					mBuf.unlock();
					break;
				}

				while (!groundBuf.empty() && groundBuf.front().header.stamp.toSec() < laserCornerBuf.front()->header.stamp.toSec())
					groundBuf.pop();
				if (groundBuf.empty())
				{
					mBuf.unlock();
					break;
				}

				timeLaserCloudCorner = laserCornerBuf.front()->header.stamp.toSec();
				timeLaserCloudSurf = laserSurfBuf.front()->header.stamp.toSec();
				timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
				timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
				timeLaserground = groundBuf.front().header.stamp.toSec();
				timeScan = scanBuf.front()->header.stamp.toSec();

				if (timeLaserCloudCorner != timeLaserOdometry || timeLaserCloudSurf != timeLaserOdometry || timeLaserCloudFullRes != timeLaserOdometry || timeScan != timeLaserOdometry || timeLaserground != timeLaserOdometry)
				{
					printf("time corner %f surf %f full %f odom %f scan %f \n", timeLaserCloudCorner, timeLaserCloudSurf, timeLaserCloudFullRes, timeLaserOdometry, timeScan);
					printf("--LaserMapping_thread: unsync messeage!");
					mBuf.unlock();
					break;
				}

				// printf("--laserMapping: %d \n",mapping_count);

				TicToc t_whole;

				curTime = timeLaserOdometry;	
				if (laserCornerBuf.size() < initfirst) // 缓存一帧激光数据，使得imu数据完全充满上一帧；
				{
					mBuf.unlock();
					std::chrono::milliseconds time(10);
					std::this_thread::sleep_for(time);
					break;	
				}
				static int first_flag = 0;
				if (first_flag < initfirst)
				{
					first_flag++;
					laserCornerBuf.pop(); // 丢掉第一帧；
					prevTime = curTime;

					mBuf.unlock();
					std::chrono::milliseconds time(10);
					std::this_thread::sleep_for(time);
					break;
				}

				if(USE_IMU)
                {
                    if (!getIMUInterval(prevTime, curTime, accVector, gyrVector))
                    {
						laserCornerBuf.pop();
                        mBuf.unlock();
                        std::chrono::milliseconds time(10);
						std::this_thread::sleep_for(time);
                        break;
                    }
                    delta_q_imu = Eigen::Quaterniond::Identity();

                    for (size_t i = 0; i < accVector.size(); i++)
                    {
                        double dt = 0;
                        if (i == 0)
                            dt = accVector[i].first - prevTime;
                        else if (i == (accVector.size() - 1))
                            dt = curTime - accVector[i - 1].first;
                        else
                            dt = accVector[i].first - accVector[i - 1].first;
                        IMU_preintegration(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                    }
                }

				laserCloudCorner->clear();
				pcl::fromROSMsg(*laserCornerBuf.front(), *laserCloudCorner);
				laserCornerBuf.pop();

				laserCloudSurf->clear();
				pcl::fromROSMsg(*laserSurfBuf.front(), *laserCloudSurf);
				laserSurfBuf.pop();

				lidarScan->clear();
				pcl::fromROSMsg(*scanBuf.front(), *lidarScan);
				scanBuf.pop();

				ground_param = groundBuf.front();
                groundBuf.pop();
                ground_cur.vector_norm = Eigen::Vector3d(ground_param.normx,ground_param.normy,ground_param.normz);
                ground_cur.vector_1 = Eigen::Vector3d(ground_param.vector1x,ground_param.vector1y,ground_param.vector1z);
                ground_cur.vector_2 = Eigen::Vector3d(ground_param.vector2x,ground_param.vector2y,ground_param.vector2z);
                ground_cur.distance = ground_param.distance;
                ground_cur.source = ground_param.source;

				q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
				q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
				q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
				q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
				t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
				t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
				t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
				odometryBuf.pop();

				while (laserCornerBuf.size() > 2)
				{
					laserCornerBuf.pop();
					printf("--laserMapping: drop lidar frame in mapping for real time !!! \n\n\n");
				}

				mBuf.unlock();

				TicToc t_prepareMap;

				extractSurroundingKeyFramesAndMap();//构建局部关键帧和地图

				downsampleCurrentLaserCloud();
				// printf("--laserMapping: corner: %d,  surf: %d \n", laserCloudCornerDSNum, laserCloudSurfDSNum);

				// printf("map prepare time %f ms\n", t_prepareMap.toc());

				transformAssociateToMap(); // set initial guess from laserOdometry;

				groundidentify();

				if (laserCloudCornerDSNum > 10 && laserCloudSurfDSNum > 50 && laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 50)
				{
					TicToc t_opt;

					kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
					kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

					for (int iterCount = 0; iterCount < 2; iterCount++)
					{
						ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
						ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
						ceres::Problem::Options problem_options;

						ceres::Problem problem(problem_options);
						problem.AddParameterBlock(para_q, 4, q_parameterization);
						problem.AddParameterBlock(para_t, 3);
						problem.AddParameterBlock(para_q_last, 4, q_parameterization);
						problem.AddParameterBlock(para_t_last, 3);

						PointType2 pointOri, pointSel;
						std::vector<int> pointSearchInd;
						std::vector<float> pointSearchSqDis;

						int corner_num = 0;
						for (int i = 0; i < laserCloudCornerDSNum; i++)
						{
							pointOri = laserCloudCornerDS->points[i];
							pointAssociateToMap(&pointOri, &pointSel);
							kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

							if (pointSearchSqDis[4] < 1.0)
							{ 
								std::vector<Eigen::Vector3d> nearCorners;
								Eigen::Vector3d center(0, 0, 0);
								for (int j = 0; j < 5; j++)
								{
									Eigen::Vector3d tmp(laserCloudCornerFromMapDS->points[pointSearchInd[j]].x,
														laserCloudCornerFromMapDS->points[pointSearchInd[j]].y,
														laserCloudCornerFromMapDS->points[pointSearchInd[j]].z);
									center = center + tmp;
									nearCorners.push_back(tmp);
								}
								center = center / 5.0;

								Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
								for (int j = 0; j < 5; j++)
								{
									Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
									covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
								}

								Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

								// if is indeed line feature
								// note Eigen library sort eigenvalues in increasing order
								Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
								Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
								float edge_var = pointOri.normal_x;
								if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
								{ 
									Eigen::Vector3d point_on_line = center;
									Eigen::Vector3d point_a, point_b;
									point_a = 0.1 * unit_direction + point_on_line;
									point_b = -0.1 * unit_direction + point_on_line;

									ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, edge_var);
									problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
									corner_num++;	
								}							
							}
						}

						// 同时将上一帧也和地图对齐来优化上一帧位姿；
						int cornerLast_num = 0;
						for (int i = 0; i < laserCloudCornerLastDSNum; i++)
						{
							pointOri = laserCloudCornerLastDS->points[i];
							lastPointAssociateToMap(&pointOri, &pointSel);
							kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

							if (pointSearchSqDis[4] < 1.0)
							{ 
								std::vector<Eigen::Vector3d> nearCorners;
								Eigen::Vector3d center(0, 0, 0);
								for (int j = 0; j < 5; j++)
								{
									Eigen::Vector3d tmp(laserCloudCornerFromMapDS->points[pointSearchInd[j]].x,
														laserCloudCornerFromMapDS->points[pointSearchInd[j]].y,
														laserCloudCornerFromMapDS->points[pointSearchInd[j]].z);
									center = center + tmp;
									nearCorners.push_back(tmp);
								}
								center = center / 5.0;

								Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
								for (int j = 0; j < 5; j++)
								{
									Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
									covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
								}

								Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

								// if is indeed line feature
								// note Eigen library sort eigenvalues in increasing order
								Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
								Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
								float edge_var = pointOri.normal_x;
								if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
								{ 
									Eigen::Vector3d point_on_line = center;
									Eigen::Vector3d point_a, point_b;
									point_a = 0.1 * unit_direction + point_on_line;
									point_b = -0.1 * unit_direction + point_on_line;

									ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, edge_var);
									problem.AddResidualBlock(cost_function, loss_function, para_q_last, para_t_last);
									cornerLast_num++;	
								}							
							}
						}

						int surf_num = 0;
						for (int i = 0; i < laserCloudSurfDSNum; i++)
						{
							pointOri = laserCloudSurfDS->points[i];
							pointAssociateToMap(&pointOri, &pointSel);
							kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

							Eigen::Matrix<double, 5, 3> matA0;
							Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
							if (pointSearchSqDis[4] < 2.0) // 1.0
							{
								for (int j = 0; j < 5; j++)
								{
									matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
									matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
									matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
								}

								// find the norm of plane
								Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
								double negative_OA_dot_norm = 1 / norm.norm();
								norm.normalize();

								// Here n(pa, pb, pc) is unit norm of plane
								bool planeValid = true;
								for (int j = 0; j < 5; j++)
								{
									// if OX * n > 0.2, then plane is not fit well
									if (fabs(norm(0) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
											 norm(1) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
											 norm(2) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
									{
										planeValid = false;
										break;
									}
								}
								Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
								float plane_var = pointOri.normal_x;
								if (planeValid)
								{
									ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm, plane_var);
									problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
									surf_num++;
								}
							}
						}

						int surfLast_num = 0;
						for (int i = 0; i < laserCloudSurfLastDSNum; i++)
						{
							pointOri = laserCloudSurfLastDS->points[i];
							lastPointAssociateToMap(&pointOri, &pointSel);
							kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

							Eigen::Matrix<double, 5, 3> matA0;
							Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
							if (pointSearchSqDis[4] < 2.0) // 1.0
							{
								for (int j = 0; j < 5; j++)
								{
									matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
									matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
									matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
								}

								// find the norm of plane
								Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
								double negative_OA_dot_norm = 1 / norm.norm();
								norm.normalize();

								// Here n(pa, pb, pc) is unit norm of plane
								bool planeValid = true;
								for (int j = 0; j < 5; j++)
								{
									// if OX * n > 0.2, then plane is not fit well
									if (fabs(norm(0) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
											 norm(1) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
											 norm(2) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
									{
										planeValid = false;
										break;
									}
								}
								Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
								float plane_var = pointOri.normal_x;
								if (planeValid)
								{
									ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm, plane_var);
									problem.AddResidualBlock(cost_function, loss_function, para_q_last, para_t_last);
									surfLast_num++;
								}
							}
						}

						if (USE_IMU == 1 && map_update != 0)
                        {
                            // 添加相对约束： 此处将旋转方差设置为0.01；
							double imu_cov;
							Eigen::Vector3d d_ypr = Utility::R2ypr(delta_q_imu.toRotationMatrix());
							if (d_ypr.norm() > 0.6)
								imu_cov = 0.004;
							else
								imu_cov = 0.4;
							
                            ceres::CostFunction* imu_dq_factor = RelativeRFactor::Create(delta_q_imu.x(), delta_q_imu.y(), delta_q_imu.z(), delta_q_imu.w(), imu_cov);
                            problem.AddResidualBlock(imu_dq_factor, NULL, para_q_last, para_q);

							// 添加绝对重力约束：应该和重力对齐；减去IMU的安装相对于水平面的姿态后，雷达初始姿态应该等于IMU姿态；
							Eigen::Matrix3d Rwl = IMUTemp.Rwi * R_il;
							Eigen::Vector3d ypr = Utility::R2ypr(Rwl);
							double pl_tmp = ypr.y() * deg2rad;
							double rl_tmp = ypr.z() * deg2rad;
                            ceres::CostFunction* Gravity_factor = PitchRollFactor::Create(pl_tmp, rl_tmp, 0.02); // 全局约束：不能小于0.01
                            problem.AddResidualBlock(Gravity_factor, NULL, para_q);

							Eigen::Matrix3d Rwl_last = IMULast.Rwi * R_il;
							Eigen::Vector3d ypr_last = Utility::R2ypr(Rwl_last);
							double pl_last = ypr_last.y() * deg2rad;
							double rl_last = ypr_last.z() * deg2rad;
                            ceres::CostFunction* Gravity_factor_last = PitchRollFactor::Create(pl_last, rl_last, 0.02);
                            problem.AddResidualBlock(Gravity_factor_last, NULL, para_q_last);
                        }

						if (USE_GROUND &&  gflag == 0 &&map_update != 0 && mapping_count > 20)
                        {
							// 添加z轴位移约束: 位移方差设置为0.01；
							// double ground_cov = 40000;
							// Eigen::Vector3d ypr = Utility::R2ypr(Eigen::Matrix3d(q_w_curr));
							// if (fabs(ypr.y()) < 5.0) // 俯仰角度小于阈值时才添加约束;
							// {
							// 	ground_cov = 0.000001;
							// }
							// ceres::CostFunction* Ground_factor = GroundFactor::Create(ground_cov);
							// problem.AddResidualBlock(Ground_factor, NULL, para_t_last, para_t);
							double ground_cov = 0.2;
							// ROS_INFO("----------------------mapping now using ground factor---------------------");
							Eigen::Matrix<double, 4, 1> histoary_q{q_w_curr_f.x(),q_w_curr_f.y(),q_w_curr_f.z(),q_w_curr_f.w()};
							Eigen::Matrix<double, 4, 1> last_q_q{q_w_last.x(),q_w_last.y(),q_w_last.z(),q_w_last.w()};
							Eigen::Matrix<double, 3, 1> last_t_t{t_w_last.x(),t_w_last.y(),t_w_last.z()};
							ceres::CostFunction* ground_factor = Ground_DeltaFactor_goable::Create(ground_last, ground_cur, histoary_q, last_q_q, last_t_t, ground_cov);
							problem.AddResidualBlock(ground_factor, NULL, para_q, para_t);

							Eigen::Matrix<double, 4, 1> histoary_q2{q_w_curr_f2.x(),q_w_curr_f2.y(),q_w_curr_f2.z(),q_w_curr_f2.w()};
							Eigen::Matrix<double, 4, 1> last_q_q2{q_w_last2.x(),q_w_last2.y(),q_w_last2.z(),q_w_last2.w()};
							Eigen::Matrix<double, 3, 1> last_t_t2{t_w_last2.x(),t_w_last2.y(),t_w_last2.z()};
							ceres::CostFunction* ground_factor2 = Ground_DeltaFactor_goable::Create(ground_last2, ground_last, histoary_q2, last_q_q2, last_t_t2, ground_cov);
							problem.AddResidualBlock(ground_factor2, NULL, para_q_last, para_t_last);
							// ceres::CostFunction* ground_factor2 = Ground_DeltaFactor_goable::Create(g_w_curr_delta, ground_last, histoary_q, histoary_t, ground_cov);
							// problem.AddResidualBlock(ground_factor2, NULL, para_q_last, para_t_last);
                        }
						else
						{
							ROS_WARN("--------------mapping now ground unuse---------------");
						}

						TicToc t_solver;
						ceres::Solver::Options options;
						options.linear_solver_type = ceres::DENSE_QR;
						options.max_num_iterations = 6;
						options.minimizer_progress_to_stdout = false;
						options.check_gradients = false;
						options.gradient_check_relative_precision = 1e-4;
						ceres::Solver::Summary summary;
						ceres::Solve(options, &problem, &summary);
						
						// printf("corner factor: %d, %d; surf factor: %d, %d \n", corner_num, cornerLast_num, surf_num, surfLast_num);
						// printf("--laserMapping: ceres solving time %f \n", t_solver.toc());
					}

					// printf("--laserMapping: mapping optimization time %f \n", t_opt.toc());
				}
				else
				{
					if (laserCloudCornerDSNum < 10 || laserCloudSurfDSNum < 50)
					{
						
						ROS_WARN("--laserMapping: laserCloud corner and surf num are not enough!!!");
					}
					if (laserCloudCornerFromMapDSNum < 10 || laserCloudSurfFromMapDSNum < 50)
					{
						ROS_WARN("--laserMapping: map corner and surf num are not enough!!!");
					}
				}

				q_w_last.normalize();
				q_w_curr.normalize();
				
				static int frameCount = 0;
				if (USE_IMU && frameCount < initskipframe)
				{
					t_w_curr = Eigen::Vector3d::Zero();
					q_w_curr = IMU.Rwi * R_il;

					t_w_curr = Eigen::Vector3d(init_x, init_y, init_z);
					q_w_curr = Utility::ypr2R(Utility::R2ypr(Eigen::Matrix3d(q_w_curr)) + Eigen::Vector3d(init_yaw, 0, 0));
					q_w_curr.normalize();
					ROS_WARN("--laserMapping: 111111111!!!");
				}
				else if (frameCount < initskipframe)
				{
					t_w_curr = Eigen::Vector3d::Zero();
					q_w_curr = Eigen::Quaterniond::Identity();

				}
				frameCount++;

				transformUpdate(); // 求解最新的地图坐标系下的位姿后，更新地图与里程计坐标之间的变换；
				Eigen::Vector3d ypr_last = Utility::R2ypr(Eigen::Matrix3d(q_w_last)) * deg2rad;
				saveKeyframeAndOdomFactor(t_w_last, ypr_last, prevTime);
				q_w_last2 = q_w_last;
				t_w_last2 = t_w_last;
				ground_last2 = ground_last;
				q_w_last = q_w_curr;
				t_w_last = t_w_curr;
				IMULast = IMUTemp;
				prevTime = curTime;
				ground_last = ground_cur;

				if (frameCount % 2 == 0)
				{
					surroundingMapDS->clear();
					*surroundingMapDS += *laserCloudCornerFromMapDS;
					*surroundingMapDS += *laserCloudSurfFromMapDS;

					sensor_msgs::PointCloud2 laserCloudSurroundMsg;
					pcl::toROSMsg(*surroundingMapDS, laserCloudSurroundMsg);
					laserCloudSurroundMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
					laserCloudSurroundMsg.header.frame_id = "camera_init";
					pubLaserCloudSurround.publish(laserCloudSurroundMsg);
				}
				mapping_count++;

				nav_msgs::Odometry odomAftMapped;
				odomAftMapped.header.frame_id = "camera_init";
				odomAftMapped.child_frame_id = "aft_mapped";
				odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
				odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
				odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
				odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
				odomAftMapped.pose.pose.position.x = t_w_curr.x();
				odomAftMapped.pose.pose.position.y = t_w_curr.y();
				odomAftMapped.pose.pose.position.z = t_w_curr.z();
				pubOdomAftMapped.publish(odomAftMapped);
				

				// laserAfterMappedPose.header = odomAftMapped.header;
				// laserAfterMappedPose.pose = odomAftMapped.pose.pose;
				// laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
				// laserAfterMappedPath.header.frame_id = "world";
				// laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
				// pubLoamPath.publish(laserAfterMappedPath);

				cost_mapping = t_whole.toc();
				printf("--lasermapping: whole mapping time: %.1f ms \n", cost_mapping);
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
			printf("--laserMapping-getIMUInterval: not receive imu! \n");
			return false;
		}
		if (t0 <= accBuf.front().first && t1 <= accBuf.front().first)
		{
			printf("--laserMapping-getIMUInterval: wait for lidar... \n");
			return false;
		}

		if (t1 <= accBuf.back().first)
		{
			while (accBuf.front().first <= t0)
			{
				accBuf.pop();
				gyrBuf.pop();
				IMUBuf.pop();
			}
			while (accBuf.front().first < t1)
			{
				accVector.push_back(accBuf.front());
				accBuf.pop();
				gyrVector.push_back(gyrBuf.front());
				gyrBuf.pop();
				IMUBuf.pop();
			}
			accVector.push_back(accBuf.front());
			gyrVector.push_back(gyrBuf.front());
			IMUTemp = IMUBuf.front().second;
		}
		else
		{
			printf("--laserMapping-getIMUInterval: not enough imu... \n");
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

	void extractSurroundingKeyFramesAndMap()
    {
		if (cloudKeyPoses3D->points.empty() == true)
            return; 

		// 全局优化后更新最新的位姿，而且需要对优化期间新添加的关键帧进行矫正，维护局部一致性后才能进行地图构建优化；
		if (bKeyFramePoseGraphUpdated == true) 
		{
			// 对全局优化后的所有关键帧位姿进行更新；
			correctKeyFramePoseGraph();

			// 还要维护局部一致性;
			q_w_last = q_drift.cast<double>() * q_w_last;
			t_w_last = q_drift.cast<double>() * t_w_last + t_drift.cast<double>();
			q_w_last.normalize();

			q_wmap_wodom = q_drift.cast<double>() * q_wmap_wodom;
			t_wmap_wodom = q_drift.cast<double>() * t_wmap_wodom + t_drift.cast<double>();
			q_wmap_wodom.normalize();
			
			bKeyFramePoseGraphUpdated = false;
		}
        PointType currPos;
		currPos.x = t_w_curr.x();
		currPos.y = t_w_curr.y();
		currPos.z = t_w_curr.z();
		currPos.intensity = 0;
		pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
		std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

		// extract all the nearby key poses and downsample them
		kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
		kdtreeSurroundingKeyPoses->radiusSearch(currPos, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
		for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
			surroundingKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
        }

		// in each voxel, all the points present will be approximated with the closest point to the center of the voxel.
		//pcl::PointCloud<int> keypointIndices;
		pcl::UniformSampling<PointType> UniformSamplingSurroundingKeyPoses;
		UniformSamplingSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
		UniformSamplingSurroundingKeyPoses.setRadiusSearch(surroundingKeyframeDensity);
		UniformSamplingSurroundingKeyPoses.filter(*surroundingKeyPosesDS); 
        //pcl::copyPointCloud(*surroundingKeyPoses, keypointIndices.points, *surroundingKeyPosesDS);
		int numsurroundingKeyPosesDS = surroundingKeyPosesDS->points.size();
		for (int i = 0; i < (int)surroundingExistingKeyPoseID.size(); ++i)
		{
			bool existingflag = false;
			for (int j = 0; j < numsurroundingKeyPosesDS; ++j)
			{
				if (surroundingExistingKeyPoseID[i] == (int)surroundingKeyPosesDS->points[j].intensity)
				{
					existingflag = true;
					break;
				}
			}
			if (existingflag == false)
			{
				surroundingExistingKeyPoseID.erase(surroundingExistingKeyPoseID.begin() + i);
				surroundingCornerCloud.erase(surroundingCornerCloud.begin() + i);
				surroundingSurfCloud.erase(surroundingSurfCloud.begin() + i);
				i--;
			}
		}

		for (int i = 0; i < numsurroundingKeyPosesDS; ++i)
		{
			bool existingflag = false;
			for (int j = 0; j < (int)surroundingExistingKeyPoseID.size(); ++j)
			{
				if ((int)surroundingKeyPosesDS->points[i].intensity == surroundingExistingKeyPoseID[j])
				{
					existingflag = true;
                    break;
				}
			}
			if (existingflag == false)
			{
				int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                auto itCor = cornerCloudKeyFrames.find(thisKeyInd);
                auto itSur = surfCloudKeyFrames.find(thisKeyInd);
                if (itCor != cornerCloudKeyFrames.end() && itSur != surfCloudKeyFrames.end())
                {
                    surroundingExistingKeyPoseID.push_back(thisKeyInd);
                    surroundingCornerCloud.push_back(transformPointCloud(cornerCloudKeyFrames.at(thisKeyInd), &thisTransformation));
                    surroundingSurfCloud.push_back(transformPointCloud(surfCloudKeyFrames.at(thisKeyInd), &thisTransformation));
                }
			}
		}

		// fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
		laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();  
        for (int i = 0; i < (int)surroundingKeyPosesDS->size(); ++i)
        {
			*laserCloudCornerFromMap += *surroundingCornerCloud[i];
			*laserCloudSurfFromMap += *surroundingSurfCloud[i];
        }
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();

        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
    }

	void correctKeyFramePoseGraph()
	{
		// 如果发生了全局因子优化，需要更新整个位姿图；
		if (bLoopIsClosed == true || bGNSSIsAdded == true)
        {
            // clear map cache, because poses have changed;
            surroundingExistingKeyPoseID.clear();
            surroundingCornerCloud.clear();
            surroundingSurfCloud.clear();
			// clear global path and need update;
        	globalPoseGraphPath.poses.clear();

            int numPoses = cloudKeyPoses6D->size();
			mKeyframe.lock();
            for (int i = 0; i < numPoses; ++i)
            {
				int keyID = cloudKeyPoses6D->points[i].intensity; // 取出每个关键帧的ID；
				if(correctedKeyPose6DByLoop.find(keyID) != correctedKeyPose6DByLoop.end()) // 如果存在优化容器中；
				{
					PointTypePose correctedKeyPose6D = correctedKeyPose6DByLoop.at(keyID); // 取出优化后的帧；
					cloudKeyPoses3D->points[i].x = correctedKeyPose6D.x;
					cloudKeyPoses3D->points[i].y = correctedKeyPose6D.y;
					cloudKeyPoses3D->points[i].z = correctedKeyPose6D.z;
					cloudKeyPoses6D->points[i] = correctedKeyPose6D;
					updateGlobalPath(cloudKeyPoses6D->points[i]);
				}
            }

			// 因为有俩线程对优化期间新添加的关键帧进行矫正，维护局部一致性后才能进行地图构建优化；
			PointTypePose currKeyPose6D = correctedKeyPose6DByLoop.rbegin()->second; // 最后一个元素为位姿图优化中最新的元素；
			int key_curr = currKeyPose6D.intensity;
			for (int i = 0; i < numPoses; ++i)
			{
				PointTypePose tmpPose6D = cloudKeyPoses6D->points[i];
				int keyID = tmpPose6D.intensity; // 取出每个关键帧的ID；

				// 对于已经校正的关键帧跳过
				if (keyID <= key_curr)
					continue;

				Eigen::Affine3f TBefore = pcl::getTransformation(tmpPose6D.x, tmpPose6D.y, tmpPose6D.z, tmpPose6D.roll, tmpPose6D.pitch, tmpPose6D.yaw);
				Eigen::Affine3f TAfter = T_Drift * TBefore;
				pcl::getTranslationAndEulerAngles(TAfter, tmpPose6D.x, tmpPose6D.y, tmpPose6D.z, tmpPose6D.roll, tmpPose6D.pitch, tmpPose6D.yaw);
				
				cloudKeyPoses3D->points[i].x = tmpPose6D.x;
				cloudKeyPoses3D->points[i].y = tmpPose6D.y;
				cloudKeyPoses3D->points[i].z = tmpPose6D.z;
				cloudKeyPoses6D->points[i] = tmpPose6D;
				updateGlobalPath(cloudKeyPoses6D->points[i]);
			}
			mKeyframe.unlock();

			// 对KeyPose6D容器也更新；
			KeyPose6D.clear();
			int keySize = cloudKeyPoses6D->size();
			for (int i = 0; i < keySize; ++i)
			{
				PointTypePose thisPose6D = cloudKeyPoses6D->points[i];
				int keyID = thisPose6D.intensity;
				KeyPose6D[keyID] = thisPose6D;
			}

			// 位姿图更新后，重新绘制闭环约束可视化；
			visualizeLoopConstraintEdge();

            bLoopIsClosed = false;
            bGNSSIsAdded = false;
        }
	}

	void downsampleCurrentLaserCloud(void)
    {
		// save last cloud
		laserCloudCornerLastDS->clear();
		laserCloudSurfLastDS->clear();
		lidarScanLastDS->clear();
		pcl::copyPointCloud(*laserCloudCornerDS, *laserCloudCornerLastDS);
        pcl::copyPointCloud(*laserCloudSurfDS, *laserCloudSurfLastDS);
		pcl::copyPointCloud(*lidarScanDS, *lidarScanLastDS);
		laserCloudCornerLastDSNum = laserCloudCornerDSNum;
		laserCloudSurfLastDSNum = laserCloudSurfDSNum;

        // Downsample cloud from current scan
        laserCloudCornerDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCorner);
        downSizeFilterCorner.filter(*laserCloudCornerDS);
        laserCloudCornerDSNum = laserCloudCornerDS->size();

        laserCloudSurfDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurf);
        downSizeFilterSurf.filter(*laserCloudSurfDS);
        laserCloudSurfDSNum = laserCloudSurfDS->size();

		lidarScanDS->clear();
		downSizeFilterScan.setInputCloud(lidarScan);
        downSizeFilterScan.filter(*lidarScanDS);
    }

	void transformAssociateToMap()
	{
		q_w_curr = q_wmap_wodom * q_wodom_curr;
		t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
		q_w_curr.normalize();
	}

	void groundidentify()
	{
		if(mapping_count <=20)
		{
			ground_histoary = ground_last;
			q_histoary = q_w_last;
			t_histoary = t_w_last;
			histoary_pose.push_back(q_w_last);
			histoary_trans.push_back(t_w_last);
			histoary_plane.push_back(ground_last);
		}
		else
		{
			ground_histoary = ground_last;
			q_histoary = q_w_last;
			t_histoary = t_w_last;
			Eigen::Quaterniond q_last_curr_l = q_histoary.conjugate()*q_w_curr;
			Eigen::Vector3d t_last_curr_l = q_histoary.conjugate()*(t_w_curr - t_histoary);
			Eigen::Vector3d ground_norm_cur = q_last_curr_l*ground_cur.vector_norm;
			double ground_distance_cur = ground_cur.distance + ground_norm_cur.dot(t_last_curr_l);
			double ground_erro_1 = (ground_histoary.distance*ground_histoary.vector_norm - ground_distance_cur*ground_norm_cur).norm();
			double ground_erro_2 = abs(ground_histoary.vector_1.dot(ground_norm_cur));
			double ground_erro_3 = abs(ground_histoary.vector_2.dot(ground_norm_cur));
			Eigen::Vector3d d_ypr = Utility::R2ypr(delta_q_imu.toRotationMatrix());
			// std::cout<<"grounderro1:"<<ground_erro_1<<"grounderro2:"<<ground_erro_2<<"grounderro3:"<<ground_erro_3<<std::endl;
			if(ground_erro_1>=0.02&&ground_erro_2>=0.02&&fabs(d_ypr.y()) > 0.5)
                    {
                        ROS_ERROR("-------ground change-----");
                        changegroundflag=0;
                        gflag=1;
                        
                    }
                    if(gflag==1&&changegroundflag<25)
                    {                     
                      changegroundflag++;
                      if(changegroundflag==25)
                      {
                        Eigen::Quaterniond last_q;
						Eigen::Vector3d last_p;
						ground_s last_g;

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
								last_p = histoary_trans[i];
								last_g = histoary_plane[i];
                            }
                        }
                        std::cout<<"lasermapping---pr_erro----------------------"<<pr_erro<<"----------------------------"<<std::endl;
                        std::cout<<"lasermapping---pr_erro----------------------"<<pr_erro<<"----------------------------"<<std::endl;
                        if(pr_erro<6)
                        {
                            q_w_curr_delta = last_q;
							t_w_curr_delta = last_p;
							g_w_curr_delta = last_g;
                            gflag=0;
                            ROS_WARN("lasermapping---this ground can find same ground in histoary plane");
                        }
                        else
                        {
                            q_w_curr_delta = q_w_curr;
							t_w_curr_delta = t_w_curr;
							g_w_curr_delta = ground_cur;
                            histoary_pose.push_back(q_w_curr_delta);
							histoary_trans.push_back(t_w_curr_delta);
							histoary_plane.push_back(g_w_curr_delta);
                            gflag=0;
                        }
                      }
                    }                   
                    q_w_curr_f = q_w_curr_delta.conjugate()*q_w_last;
                    q_w_curr_f.normalize();
					q_w_curr_f2 = q_w_curr_delta.conjugate()*q_w_last2;
                    q_w_curr_f2.normalize();

		}
	}

	void pointAssociateToMap(PointType2 const *const pi, PointType2 *const po)
	{
		Eigen::Vector3d point(pi->x, pi->y, pi->z);
		Eigen::Vector3d point_out = q_w_curr * point + t_w_curr;
		po->x = point_out.x();
		po->y = point_out.y();
		po->z = point_out.z();
		po->intensity = pi->intensity;
		po->normal_x = pi->normal_x;
	}

	void lastPointAssociateToMap(PointType2 const *const pi, PointType2 *const po)
	{
		Eigen::Vector3d point(pi->x, pi->y, pi->z);
		Eigen::Vector3d point_out = q_w_last * point + t_w_last;
		po->x = point_out.x();
		po->y = point_out.y();
		po->z = point_out.z();
		po->intensity = pi->intensity;
		po->normal_x = pi->normal_x;
	}

	void transformUpdate()
	{
		q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
		t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
	}

	bool saveKeyframeAndOdomFactor(Eigen::Vector3d pos, Eigen::Vector3d rot, double time)
	{
		if (laserCloudCornerLastDSNum == 0 || laserCloudSurfLastDSNum == 0) return false;
		if (map_update == 0) return false;
		if (cloudKeyPoses3D->size() > 10)//太近不认为是关键帧
		{
			PointTypePose lastPose6D = cloudKeyPoses6D->back();
			float dx = pos.x() - lastPose6D.x;
			float dy = pos.y() - lastPose6D.y;
			float dz = pos.z() - lastPose6D.z;
			float dyaw = rot.x() - lastPose6D.yaw;
			float dpitch = rot.y() - lastPose6D.pitch;
			float droll = rot.z() - lastPose6D.roll;
			if (dyaw > M_PI) dyaw = dyaw - M_PI * 2; 
			if (dyaw < -M_PI) dyaw = dyaw + M_PI * 2; 

			if (abs(droll) < keyframeAddingAngle && 
                abs(dpitch) < keyframeAddingAngle && 
                abs(dyaw) < keyframeAddingAngle && 
                sqrt(dx * dx + dy * dy + dz * dz) < keyframeAddingDistance)
			{
				return false;
			}
		}

		// 拼接地图时流出重定位的时间
		static int initial_count = 0;
		
		bnewKeyFrame = true;
		
		// temp key poses
        PointType thisPose3D;
        PointTypePose thisPose6D, lastPose6D;

        thisPose3D.x = pos.x();
        thisPose3D.y = pos.y();
        thisPose3D.z = pos.z();
        thisPose3D.intensity = keyFrameNum; // this can be used as index
        
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
		thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.yaw = rot.x();
        thisPose6D.pitch = rot.y();
        thisPose6D.roll = rot.z();
        thisPose6D.time = time;

		// calculate travel distance;
		if (cloudKeyPoses6D->empty())
        {
            travel_distance[keyFrameNum] = 0;
			travel_angle[keyFrameNum] = 0;
        }
        else
        {
			lastPose6D = cloudKeyPoses6D->back();

            float dis_delta = sqrt((thisPose6D.x - lastPose6D.x) * (thisPose6D.x - lastPose6D.x) + 
								   (thisPose6D.y - lastPose6D.y) * (thisPose6D.y - lastPose6D.y) +
								   (thisPose6D.z - lastPose6D.z) * (thisPose6D.z - lastPose6D.z));
            float dis_temp = travel_distance.rbegin()->second + dis_delta;//累计路程
            travel_distance[keyFrameNum] = dis_temp;

			float rad_delta = thisPose6D.yaw - lastPose6D.yaw;
			if (rad_delta > M_PI) rad_delta = rad_delta - M_PI * 2; 
			if (rad_delta < -M_PI) rad_delta = rad_delta + M_PI * 2; 
            float rad_temp = travel_angle.rbegin()->second + abs(rad_delta);//累计旋转
            travel_angle[keyFrameNum] = rad_temp;
        }
		
		pcl::PointCloud<PointType2>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType2>());
		pcl::PointCloud<PointType2>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType2>());
		pcl::PointCloud<PointType>::Ptr thisScanFrame(new pcl::PointCloud<PointType>());
		pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
		pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
		pcl::copyPointCloud(*lidarScanLastDS, *thisScanFrame);
		bool usingRawCloud = true;
		mKeyframe.lock();
		cloudKeyPoses3D->push_back(thisPose3D);
		//printf("num:%d",cloudKeyPoses3D->size());
		cloudKeyPoses6D->push_back(thisPose6D);
		cornerCloudKeyFrames[keyFrameNum] = thisCornerKeyFrame;
		surfCloudKeyFrames[keyFrameNum] = thisSurfKeyFrame;
		scanCloudKeyFrames[keyFrameNum] = thisScanFrame;
		KeyPose6D[keyFrameNum] = thisPose6D;
        //sc处理
		// if( usingRawCloud ) { // v2 uses downsampled raw point cloud, more fruitful height information than using feature points (v1)
        //     scManager.makeAndSaveScancontextAndKeys(*thisScanFrame);
        // }
        // else { // v1 uses thisSurfKeyFrame, it also works. (empirically checked at Mulran dataset sequences)
        //     scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 
        // }
		mKeyframe.unlock();
		// 存储pcd文件
		boost::format fmt_pcd("%s/%s/%d.%s");
        pcl::io::savePCDFileASCII((fmt_pcd % saveDirectory % "cornercloud" % keyFrameNum % "pcd").str(), *thisCornerKeyFrame);
        pcl::io::savePCDFileASCII((fmt_pcd % saveDirectory % "surfcloud" % keyFrameNum % "pcd").str(), *thisSurfKeyFrame);

		keyFrameNum++;

		updateGlobalPath(thisPose6D);

		return true;
	}

	void updateGlobalPath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = "camera_init";
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();
        globalPoseGraphPath.poses.push_back(pose_stamped);
    }

	// 位姿图优化线程
	void poseGraphOptimizationThread()
    {
        ros::Rate rate(1);
        while (ros::ok())
        {
            rate.sleep();

			// 全局优化后位姿图已经更新，需要建图线程确认并维护局部一致性后才能进行下次全局优化；
			if(bKeyFramePoseGraphUpdated == true) continue;

			copyKeyPosesData();
			
            detectAndCalculateLoopFactor();

			PoseGraphOptimize4DoF();

			publishGlobalPoseGraph();

			publishGlobalMap();
        }
    }

	void copyKeyPosesData()
	{
		if (cloudKeyPoses3D->empty()) return;

		copy_cloudKeyPoses3D->clear();
        copy_cloudKeyPoses6D->clear();

		mKeyframe.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mKeyframe.unlock();

		correctedKeyPose6DByLoop.clear();
		int keyPoseNum = copy_cloudKeyPoses6D->size();
		for (int i = 0; i < keyPoseNum; ++i)
		{
			PointTypePose thisPose6D = copy_cloudKeyPoses6D->points[i];
			int keyI = thisPose6D.intensity;
			correctedKeyPose6DByLoop[keyI] = thisPose6D;
		}
	}

	void detectAndCalculateLoopFactor()
    {
		static int loopContinueCount = 0;
		static bool lowDriftFlag = false; 
		static float lastLoopDistance = -1000;

		if (LoopClosureEnable != 1) return;

		if (map_update != 1) return; // 定位模式下不进行回环检测

        if (cloudKeyPoses3D->points.empty()) return;

		// control loop in proper frequency;
		// 控制规则：
		// 1.根据距离上次闭环后的运动距离来决定连续闭环的状态；
		// 2.超过连续3次闭环，进入低漂移状态；距离上次闭环后的运动距离大于设定阈值，进入高漂移状态；
		// 3.在低漂移状态下，每隔10米接受一次闭环；高漂移状态每个周期都接受闭环；

		// 系统进入低漂移状态时，每隔10米进行一次闭环优化；
		PointTypePose latestKeyPose6D, closestHistoryKeyPose6D;
		latestKeyPose6D = copy_cloudKeyPoses6D->back();
        int latestKeyFrameID = latestKeyPose6D.intensity;
		if (lowDriftFlag == true)
		{
			if (abs(travel_distance.at(latestKeyFrameID) - lastLoopDistance) < 5)
				return;
		}
		if (abs(travel_distance.at(latestKeyFrameID) - lastLoopDistance) > 20)
		{
			lowDriftFlag = false;
		}

		if (bnewKeyFrame == false)
            return;
		bnewKeyFrame = false;

		// begin loop detection;
		TicToc t_loop;
        int latestFrameID = -1, closestHistoryFrameID = -1;
        if (detectLoopClosure(&latestFrameID, &closestHistoryFrameID, &closestHistoryKeyPose6D) == false)
		{
			return;
		}

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType2, PointType2> icp;
        icp.setMaxCorrespondenceDistance(poseGraphSearchRadius*2);//匹配最大距离
        icp.setMaximumIterations(100);//最大迭代次数
        icp.setTransformationEpsilon(1e-6);//收敛条件 两次迭代的转换矩阵的最大容差
        icp.setEuclideanFitnessEpsilon(1e-6);//收敛条件 均方误差和小于阈值
        icp.setRANSACIterations(0);// 设置RANSAC运行次数

        // Downsample map cloud
        pcl::PointCloud<PointType2>::Ptr cloud_temp(new pcl::PointCloud<PointType2>());
        downSizeFilterICP.setInputCloud(nearHistoryKeyFrameCloud);
        downSizeFilterICP.filter(*cloud_temp);
        *nearHistoryKeyFrameCloud = *cloud_temp;

        // Align clouds
        icp.setInputSource(latestKeyFrameCloud);
        icp.setInputTarget(nearHistoryKeyFrameCloud);
        pcl::PointCloud<PointType2>::Ptr unused_result(new pcl::PointCloud<PointType2>());
        icp.align(*unused_result);

		float alignDistance = icp.getFitnessScore();
        if (icp.hasConverged() == false || alignDistance > historyKeyframeFitnessScore)//参1为成功flag 参2为配准后的点云的最近点之间距离的均值
		{
			printf("\n--PoseGraph: false loop between : %d and %d , dis: %.2f and %.2f, score: %.3f, time: %.1f, return !!!!!!!!!!!!!!! \n\n", 
			latestFrameID, closestHistoryFrameID, travel_distance.at(latestFrameID), travel_distance.at(closestHistoryFrameID), alignDistance, t_loop.toc());
			return;
		}
		else
		{
			loopClosureCount++;
			printf("\n--PoseGraph: detected loop between : %d and %d , dis: %.2f and %.2f, score: %.3f, loop_count: %d, time: %.1f !!!!!!!!!!!!!!! \n\n", 
			latestFrameID, closestHistoryFrameID, travel_distance.at(latestFrameID), travel_distance.at(closestHistoryFrameID), alignDistance, loopClosureCount, t_loop.toc());
		}

		// ICP的估计结果为：当前位姿下的点云，乘以多少变换，就能与地图点云对齐，也就是得到了漂移量;
		// 计算漂移量；
		T_Drift = icp.getFinalTransformation();
		t_drift = T_Drift.translation();
		q_drift = T_Drift.rotation();
		q_drift.normalize();

		// transform from world to wrong pose;
        Eigen::Affine3f T_w_latest = pclPointToAffine3f(latestKeyPose6D); // 最新帧的原始位姿；
        // transform from world to corrected pose；
        Eigen::Affine3f T_w_correct = T_Drift * T_w_latest; // 通过漂移量校正后的最新帧位姿；
		// transform from world to loop pose;
        Eigen::Affine3f T_w_loop = pclPointToAffine3f(closestHistoryKeyPose6D); // 闭环帧的位姿；
		// transform from correct to loop pose;
        Eigen::Affine3f T_loop_correct = T_w_loop.inverse() * T_w_correct; 
	
		loopInfo loopInfoTmp;
		loopInfoTmp.key_curr = latestFrameID;
		loopInfoTmp.key_loop = closestHistoryFrameID;
		loopInfoTmp.keyPose6DCurr = latestKeyPose6D;
		loopInfoTmp.keyPose6DLoop = closestHistoryKeyPose6D;
		loopInfoTmp.t_loop_curr = T_loop_correct.translation();
		loopInfoTmp.q_loop_curr = T_loop_correct.rotation();
		loopInfoTmp.noise = alignDistance;
		loopInfoContainer[latestFrameID] = loopInfoTmp;

		if (loopInfoContainer.size() > 100)
		{
			loopInfoContainer.erase(loopInfoContainer.begin());
		}

        // save all loop constriant for visualize;
        currLoopKeyContainer.push_back(std::make_pair(latestFrameID, closestHistoryFrameID));

		visualizeLoopConstraintEdge();

		// 对系统的漂移状态进行判断，低漂移状态时限制闭环优化的频率；
		if (abs(travel_distance.at(latestFrameID) - lastLoopDistance) < 10) // 判断数据库当前帧与上个闭环帧之间的运动距离；
			loopContinueCount++;
		else
			loopContinueCount = 0; // reset
		
		if (loopContinueCount > 4)
			lowDriftFlag = true;
		
		// 闭环帧之间的距离，代表着这两帧之间的漂移被消除；
		DistanceByLoop = travel_distance.at(latestFrameID) - travel_distance.at(closestHistoryFrameID);
		if (DistanceByLoop < 0)
		{
			DistanceByLoop = 0;
			ROS_WARN("--Loopfactor: wrong DistanceByLoop !!!");
		}
		lastLoopDistance = travel_distance.at(latestFrameID);

        bLoopIsClosed = true;
    }

	bool detectLoopClosure(int *latestID, int *closestID, PointTypePose *closestKeyPose6D)
    {
		int latestFrameIndex = copy_cloudKeyPoses3D->size()-1;
        int latestFrameKeyID = copy_cloudKeyPoses3D->points[latestFrameIndex].intensity;
        int closestHistoryFrameID = -1, closestHistoryFrameIndex = -1;

		// 闭环优化后，SLAM在闭环区域内的漂移消除，因此减去这段区域的运动产生的漂移；
        poseGraphSearchRadius = historyKeyframeSearchRadius + (travel_distance.rbegin()->second - DistanceByLoop) * DRIFT_FACTOR;

        latestKeyFrameCloud->clear();
        nearHistoryKeyFrameCloud->clear();

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), poseGraphSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int index = pointSearchIndLoop[i];
			int keyID = copy_cloudKeyPoses3D->points[index].intensity;
			// 闭环帧之间的运行距离应该超过一定阈值，防止刚刚添加的相近的帧产生闭环；
            if (abs(travel_distance.at(keyID) - travel_distance.rbegin()->second) > (loopKeyframeDisDiff + poseGraphSearchRadius))
            {
				if (keyID < 10) continue;
                closestHistoryFrameID = keyID;
				closestHistoryFrameIndex = index;
				*closestKeyPose6D = copy_cloudKeyPoses6D->points[index];
                break;
            }
        }

        if (closestHistoryFrameID == -1 || closestHistoryFrameIndex == -1)
            return false;

        if (latestFrameKeyID == closestHistoryFrameID)
            return false;

        // save latest key frames
		PointTypePose latestKeyPose6D = copy_cloudKeyPoses6D->points[latestFrameIndex];

		auto itCor = cornerCloudKeyFrames.find(latestFrameKeyID);
		if (itCor != cornerCloudKeyFrames.end())
		{
			*latestKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames.at(latestFrameKeyID), &latestKeyPose6D);
		}
		auto itSur = surfCloudKeyFrames.find(latestFrameKeyID);
		if (itSur != surfCloudKeyFrames.end())
		{
			*latestKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames.at(latestFrameKeyID), &latestKeyPose6D);
		}
        
        // save history near key frames
        for (int i = -historyKeyframeSearchNum; i <= historyKeyframeSearchNum; ++i)
        {
			int thisIndex = closestHistoryFrameIndex + i;
			if (thisIndex < 0 || thisIndex >= latestFrameIndex) // 索引内存在一定范围内；
                continue;

			int thisKeyID = copy_cloudKeyPoses6D->points[thisIndex].intensity;
            if (thisKeyID < 0 || thisKeyID >= latestFrameKeyID) // id在一定范围内；
                continue;

			PointTypePose historyKeyPose6D = copy_cloudKeyPoses6D->points[thisIndex];
			auto itCor = cornerCloudKeyFrames.find(thisKeyID);
			if (itCor != cornerCloudKeyFrames.end())
			{
				*nearHistoryKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames.at(thisKeyID), &historyKeyPose6D);
			}
			auto itSur = surfCloudKeyFrames.find(thisKeyID);
			if (itSur != surfCloudKeyFrames.end())
			{
				*nearHistoryKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames.at(thisKeyID), &historyKeyPose6D);
			}
        }

        if (nearHistoryKeyFrameCloud->points.empty())
            return false;

        *latestID = latestFrameKeyID;
        *closestID = closestHistoryFrameID;

        return true;
    }

	void visualizeLoopConstraintEdge()
    {
        visualization_msgs::MarkerArray markerArray;

        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "camera_init";
        markerNode.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.4; markerNode.scale.y = 0.4; markerNode.scale.z = 0.4; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;

        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = "camera_init";
        markerEdge.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.2; markerEdge.scale.y = 0.2; markerEdge.scale.z = 0.2;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = currLoopKeyContainer.begin(); it != currLoopKeyContainer.end(); ++it)
        {
            int loopNewKeyID = it->first;
            int loopOldKeyID = it->second;
			PointTypePose loopNewPose6D, loopOldPose6D;

			auto itNew = KeyPose6D.find(loopNewKeyID);
			if (itNew != KeyPose6D.end())
			{
				loopNewPose6D = KeyPose6D.at(loopNewKeyID);
			}
			else
			{
				continue;
			}

			auto itOld = KeyPose6D.find(loopOldKeyID);
			if (itOld != KeyPose6D.end())
			{
				loopOldPose6D = KeyPose6D.at(loopOldKeyID);
			}
			else
			{
				continue;
			}

            geometry_msgs::Point p;
            p.x = loopNewPose6D.x;
            p.y = loopNewPose6D.y;
            p.z = loopNewPose6D.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = loopOldPose6D.x;
            p.y = loopOldPose6D.y;
            p.z = loopOldPose6D.z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);

        pubLoopConstraintEdge.publish(markerArray);
    }

	// 4自由度全局位姿图优化
	void PoseGraphOptimize4DoF()
	{
		if (cloudKeyPoses3D->empty()) return;

		// 如果没有闭环或者全局约束发生的话，则不进行位姿图优化；
		if (bLoopIsClosed == false && bGNSSIsAdded == false) return;

		// 将最新帧的key作为最大长度；
		int max_length = copy_cloudKeyPoses6D->back().intensity + 1; 

		double t_array[max_length][3];
		double euler_array[max_length][3];
		Eigen::Quaterniond q_array[max_length];

		double t_loop[3]={0};

		ceres::Problem problem;
		ceres::LocalParameterization* angle_local_parameterization = AngleLocalParameterization::Create();

		int odomFactorNum = 0, loopFactorNum = 0, oldestLoopKey = 100000;
		int keyPoseNum = copy_cloudKeyPoses6D->size();

		// 找到key值最小的闭环帧；
		for (int i = 0; i < keyPoseNum; ++i)
		{
			PointTypePose KeyPoses6D = copy_cloudKeyPoses6D->points[i]; // 取出点；
			int keyI = KeyPoses6D.intensity; // 取出key；

			if (loopInfoContainer.find(keyI) != loopInfoContainer.end()) // 当前帧存在闭环约束；
			{
				// 提取闭环信息；
				loopInfo loopInfoTmp = loopInfoContainer.at(keyI);
				int key_loop = loopInfoTmp.key_loop;

				// 判断闭环帧是否也存在于位姿图中，确保闭环约束正确性；
				if (correctedKeyPose6DByLoop.find(key_loop) != correctedKeyPose6DByLoop.end())
				{
					if (key_loop < oldestLoopKey)
						oldestLoopKey = key_loop;
				}			
			}
		}

		// 添加变量节点和约束因子；
		for (int i = 0; i < keyPoseNum; ++i)
		{
			PointTypePose KeyPoses6D = copy_cloudKeyPoses6D->points[i]; // 取出点；
			int keyI = KeyPoses6D.intensity; // 取出key；

			t_array[keyI][0] = KeyPoses6D.x;
			t_array[keyI][1] = KeyPoses6D.y;
			t_array[keyI][2] = KeyPoses6D.z;
			euler_array[keyI][0] = KeyPoses6D.yaw * rad2deg;
			euler_array[keyI][1] = KeyPoses6D.pitch * rad2deg;
			euler_array[keyI][2] = KeyPoses6D.roll * rad2deg;
			q_array[keyI] = Utility::ypr2R(Eigen::Vector3d(KeyPoses6D.yaw, KeyPoses6D.pitch, KeyPoses6D.roll) * rad2deg);
			
			// 添加所有变量节点；
			problem.AddParameterBlock(euler_array[keyI], 1, angle_local_parameterization); //1个角度
			problem.AddParameterBlock(t_array[keyI], 3); //3个平移

			if(i == 0) continue;

			// 添加里程计约束因子；
			PointTypePose KeyPoses6DFrom = copy_cloudKeyPoses6D->points[i-1]; // 取出上一帧；
			int keyFrom = KeyPoses6DFrom.intensity; // 取出key；
			Eigen::Vector3d t_w_ij(t_array[keyI][0] - t_array[keyFrom][0], t_array[keyI][1] - t_array[keyFrom][1], t_array[keyI][2] - t_array[keyFrom][2]);
			Eigen::Vector3d t_i_ij = q_array[keyFrom].inverse() * t_w_ij; // 当前帧相对于上一帧的位姿；局部增量；以上帧位姿为参考坐标系；
			double relative_yaw = euler_array[keyI][0] - euler_array[keyFrom][0];
			ceres::CostFunction* cost_function = FourDOFError::Create(t_i_ij.x(), t_i_ij.y(), t_i_ij.z(), relative_yaw, KeyPoses6DFrom.pitch * rad2deg, KeyPoses6DFrom.roll * rad2deg );
			problem.AddResidualBlock(cost_function, NULL, euler_array[keyFrom], t_array[keyFrom], euler_array[keyI], t_array[keyI]);

			// 添加闭环约束因子；
			if (loopInfoContainer.find(keyI) != loopInfoContainer.end()) // 当前帧存在闭环约束；
			{
				// 提取闭环信息；
				loopInfo loopInfoTmp = loopInfoContainer.at(keyI);
				int key_loop = loopInfoTmp.key_loop;
				PointTypePose keyPose6DLoop = loopInfoTmp.keyPose6DLoop;
				Eigen::Vector3d t_loop_curr = loopInfoTmp.t_loop_curr.cast<double>();
				Eigen::Quaterniond q_loop_curr = loopInfoTmp.q_loop_curr.cast<double>();

				// 提取闭环信息后，判断闭环帧是否也存在于位姿图中，确保闭环约束正确性；
				if (correctedKeyPose6DByLoop.find(key_loop) != correctedKeyPose6DByLoop.end()) 
				{
					// 添加闭环约束；
					Eigen::Vector3d euler_loop_curr = Utility::R2ypr(q_loop_curr.toRotationMatrix());
					double relative_yaw = euler_loop_curr.x();
					ceres::CostFunction* cost_function = FourDOFError::Create(t_loop_curr.x(), t_loop_curr.y(), t_loop_curr.z(), relative_yaw, keyPose6DLoop.pitch * rad2deg, keyPose6DLoop.roll * rad2deg);
					problem.AddResidualBlock(cost_function, NULL, euler_array[key_loop], t_array[key_loop], euler_array[keyI], t_array[keyI]);

					loopFactorNum++;
					printf("--PoseGraph: add Loop Factor between %d and %d, count: %d \n", keyI, key_loop, loopFactorNum );
				}
			}
			
			odomFactorNum++;
		}

		printf("--PoseGraph: add odom Factors num: %d \n", odomFactorNum );

		// 将最小闭环帧位设为固定；
		if (oldestLoopKey < 99999)
		{   
			//将闭环回环帧的位姿设置为固定；
			problem.SetParameterBlockConstant(euler_array[oldestLoopKey]);
			problem.SetParameterBlockConstant(t_array[oldestLoopKey]);
			printf("--PoseGraph: Set loop keyframe Constant: keyID: %d, pos: x: %.3f, y: %.3f, z: %.3f \n", oldestLoopKey, t_array[oldestLoopKey][0], t_array[oldestLoopKey][1], t_array[oldestLoopKey][2]);
			t_loop[0] = t_array[oldestLoopKey][0];
			t_loop[1] = t_array[oldestLoopKey][1];
			t_loop[2] = t_array[oldestLoopKey][2];
		}
		else
		{
			ROS_WARN("\n\n\n--PoseGraph: not find oldestLoopKey, pose grapgh not be optimized !!!");
			return;
		}

		// 开始求解；
		TicToc t_solve;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  
		options.max_num_iterations = 10;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		printf("--PoseGraph: ceres::Solve time: %.1f \n", t_solve.toc());

		// 将优化校正后的位姿保存起来，供建图节点用；
		correctedKeyPose6DByLoop.clear();
		for (int i = 0; i < keyPoseNum; ++i)
		{
			PointTypePose KeyPoses6DCorrected = copy_cloudKeyPoses6D->points[i]; // 取出点；
			int keyI = KeyPoses6DCorrected.intensity; // 取出key；

			if (keyI == oldestLoopKey)
			{
				// 确认闭环帧的位姿固定不变；
				// 测试中发现ceres一个bug：第一帧位姿为0，当设定第一帧位姿固定，优化后第一帧发生了变化，导致整个位姿图发生了移动；
				if(abs(t_loop[0] - t_array[keyI][0]) > 0.01 || abs(t_loop[1] - t_array[keyI][1]) > 0.01 || abs(t_loop[2] - t_array[keyI][2]) > 0.01)
				{
					ROS_WARN("\n\n\n--PoseGraph: loop pose changed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
					ROS_WARN("--PoseGraph: loop pose before: x: %.3f, y: %.3f, z: %.3f ", t_loop[0], t_loop[1], t_loop[2]);
					ROS_WARN("--PoseGraph: loop pose after : x: %.3f, y: %.3f, z: %.3f ", t_array[keyI][0], t_array[keyI][0], t_array[keyI][0] );
					return;
				}
			}

			KeyPoses6DCorrected.x = t_array[keyI][0];
			KeyPoses6DCorrected.y = t_array[keyI][1];
			KeyPoses6DCorrected.z = t_array[keyI][2];
			KeyPoses6DCorrected.yaw = euler_array[keyI][0] * deg2rad;
			KeyPoses6DCorrected.pitch = euler_array[keyI][1] * deg2rad;
			KeyPoses6DCorrected.roll = euler_array[keyI][2] * deg2rad;

			copy_cloudKeyPoses3D->points[i].x = KeyPoses6DCorrected.x;
			copy_cloudKeyPoses3D->points[i].y = KeyPoses6DCorrected.y;
			copy_cloudKeyPoses3D->points[i].z = KeyPoses6DCorrected.z;
			copy_cloudKeyPoses6D->points[i] = KeyPoses6DCorrected;

			correctedKeyPose6DByLoop[keyI] = KeyPoses6DCorrected; // 将校正后的位姿保存起来，供建图节点用；
		}

		bKeyFramePoseGraphUpdated = true;
	}

	void publishGlobalPoseGraph()
	{
		globalPoseGraphPath.header.stamp = ros::Time().fromSec( timeLaserOdometry );
		globalPoseGraphPath.header.frame_id = "camera_init";
		pubGlobalPoseGraphPath.publish(globalPoseGraphPath);

		sensor_msgs::PointCloud2 ROSCloudMsg;
		pcl::toROSMsg(*copy_cloudKeyPoses3D, ROSCloudMsg);
		ROSCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
		ROSCloudMsg.header.frame_id = "camera_init";
		pubGlobalPoseGraphPoint.publish(ROSCloudMsg);
	}

	void publishGlobalMap(void)
    {
        if (cloudKeyPoses3D->points.empty() == true) return;

		TicToc t_pubGlobalMap;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType2>::Ptr globalMap(new pcl::PointCloud<PointType2>());
        pcl::PointCloud<PointType2>::Ptr globalMapDS(new pcl::PointCloud<PointType2>());
		pcl::PointCloud<PointType>::Ptr globalMapScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapScanDS(new pcl::PointCloud<PointType>());

		int keyNum = copy_cloudKeyPoses3D->size();
        for (int i = 0; i < keyNum; ++i)
		{
			globalMapKeyPoses->push_back(copy_cloudKeyPoses3D->points[i]);
		}

		//pcl::PointCloud<int> keypointIndices;
		pcl::UniformSampling<PointType> UniformSamplingGlobalMapKeyPoses;
		UniformSamplingGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
		UniformSamplingGlobalMapKeyPoses.setRadiusSearch(globalMapVisualizationPoseDensity);
		// UniformSamplingGlobalMapKeyPoses.compute(keypointIndices);
		
        UniformSamplingGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
		//pcl::copyPointCloud(*globalMapKeyPoses, keypointIndices.points, *globalMapKeyPosesDS);
        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
		{
			PointTypePose keyPose6D = copy_cloudKeyPoses6D->points[globalMapKeyPosesDS->points[i].intensity];
            int thisKeyInd = (int)keyPose6D.intensity;

			auto itCor = cornerCloudKeyFrames.find(thisKeyInd);
			if (itCor != cornerCloudKeyFrames.end())
			{
				*globalMap += *transformPointCloud(cornerCloudKeyFrames.at(thisKeyInd), &keyPose6D);
			}
			auto itSur = surfCloudKeyFrames.find(thisKeyInd);
			if (itSur != surfCloudKeyFrames.end())
			{
				*globalMap += *transformPointCloud(surfCloudKeyFrames.at(thisKeyInd), &keyPose6D);
			}
			auto itScan = scanCloudKeyFrames.find(thisKeyInd);
			if (itScan != scanCloudKeyFrames.end())
			{
				*globalMapScan += *transformPointCloud(scanCloudKeyFrames.at(thisKeyInd), &keyPose6D);
			}
        }

        // downsample visualized points
        pcl::VoxelGrid<PointType2> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapDensity, globalMapDensity, globalMapDensity); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMap);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapDS);
		pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFramescan; // for global map visualization
		downSizeFilterGlobalMapKeyFramescan.setInputCloud(globalMapScan);
        downSizeFilterGlobalMapKeyFramescan.filter(*globalMapScanDS);

		sensor_msgs::PointCloud2 laserCloudMsg;
		pcl::toROSMsg(*globalMapDS, laserCloudMsg);
		laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
		laserCloudMsg.header.frame_id = "camera_init";
		pubLaserCloudMap.publish(laserCloudMsg);

		sensor_msgs::PointCloud2 laserCloudMsgScan;
		pcl::toROSMsg(*globalMapScanDS, laserCloudMsgScan);
		laserCloudMsgScan.header.stamp = ros::Time().fromSec(timeLaserOdometry);
		laserCloudMsgScan.header.frame_id = "camera_init";
		pubScanCloudMap.publish(laserCloudMsgScan);

		sensor_msgs::PointCloud2 KeyPosesMsg;
		pcl::toROSMsg(*globalMapKeyPoses, KeyPosesMsg);
		KeyPosesMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
		KeyPosesMsg.header.frame_id = "camera_init";
		pubGlobalMapKeyPose.publish(KeyPosesMsg);

		sensor_msgs::PointCloud2 KeyPosesDSMsg;
		pcl::toROSMsg(*globalMapKeyPosesDS, KeyPosesDSMsg);
		KeyPosesDSMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
		KeyPosesDSMsg.header.frame_id = "camera_init";
		pubGlobalMapKeyPoseDS.publish(KeyPosesDSMsg);

		int keyDSNum = globalMapKeyPosesDS->size();
		//printf("--GlobalMap: construct and pub GlobalMap time: %.1f, keyNum: %d, keyDSNum: %d \n", t_pubGlobalMap.toc(), keyNum, keyDSNum);
    }
	
	pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);
		Eigen::Quaterniond q_temp;
		Eigen::Vector3d t_temp(transformIn->x, transformIn->y, transformIn->z);
		q_temp = Utility::ypr2R(Eigen::Vector3d(transformIn->yaw, transformIn->pitch, transformIn->roll) * rad2deg);
		for (int i = 0; i < cloudSize; ++i)
		{
			pointFrom = &cloudIn->points[i];
			Eigen::Vector3d point_curr(pointFrom->x, pointFrom->y, pointFrom->z);
			Eigen::Vector3d point_w = q_temp * point_curr + t_temp;
			cloudOut->points[i].x = point_w.x();
			cloudOut->points[i].y = point_w.y();
			cloudOut->points[i].z = point_w.z();
			cloudOut->points[i].intensity = pointFrom->intensity;
		}
        return cloudOut;
    }

	pcl::PointCloud<PointType2>::Ptr transformPointCloud(pcl::PointCloud<PointType2>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType2>::Ptr cloudOut(new pcl::PointCloud<PointType2>());

        PointType2 *pointFrom;
        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);
		Eigen::Quaterniond q_temp;
		Eigen::Vector3d t_temp(transformIn->x, transformIn->y, transformIn->z);
		q_temp = Utility::ypr2R(Eigen::Vector3d(transformIn->yaw, transformIn->pitch, transformIn->roll) * rad2deg);
		for (int i = 0; i < cloudSize; ++i)
		{
			pointFrom = &cloudIn->points[i];
			Eigen::Vector3d point_curr(pointFrom->x, pointFrom->y, pointFrom->z);
			Eigen::Vector3d point_w = q_temp * point_curr + t_temp;
			cloudOut->points[i].x = point_w.x();
			cloudOut->points[i].y = point_w.y();
			cloudOut->points[i].z = point_w.z();
			cloudOut->points[i].intensity = pointFrom->intensity;
			cloudOut->points[i].normal_x = pointFrom->normal_x;
		}
        return cloudOut;
    }

	Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }
};


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");

	LaserMapping LaserMapper;

	ros::spin();

	return 0;
}
