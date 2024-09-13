#include "rgc_slam/utility.h"
#include "rgc_slam/tic_toc.h"

float cloudCurvature[30000];
float intensityCurvature[30000];
float cloudCurvature2[30000];
bool comp(int i, int j){return (cloudCurvature[i] < cloudCurvature[j]);}
bool comp_I(int i, int j){return (intensityCurvature[i] < intensityCurvature[j]);}

class ScanRegistration
{
  public:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;

    ros::Publisher pubLaserCloud;
    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubintenPointsSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubGroundPointsFlat;
    ros::Publisher pubScan;
    ros::Publisher pubgroundparam;
    ros::Publisher feature_source_plane;
    ros::Publisher feature_source_sharp;
    ros::Publisher feature_source_inten;
    
    
    int N_SCANS = 16;
    int USE_intensity = 1;
    double MINIMUM_RANGE = 0.5, MAXMUM_RANGE = 80;
    float lineResolution = 0.2, planeResolution = 0.4;

    const int groundScanInd = 7;
    const double scanPeriod = 0.1;
    const int systemDelay = 10;
    bool systemInited = false;
    int systemInitCount = 0;
    double laderH = 0.56;
    float Ground_scan_range[16] = {2.66, 3.04, 3.56, 4.30, 5.44, 7.41, 11.63, 27.12}; // h: 0.56m ; -15 至 -1 度； (pitch: 0.5deg)

    int cloudSortInd[30000];
    int intenSortInd[30000];
    int cloudNeighborPicked[30000];
    int intenNeighborPicked[30000];
    int cloudLabel[30000];
    int intenLabel[30000];
    int groundcloudMarked[30000];
    float range_vec[30000];
    float scan_angle[30000];
    float distance_source[30000];
    float other_source[30000];


    ScanRegistration()
    {
        nh.param<int>("scan_line", N_SCANS, 16);
        nh.param<int>("USE_intensity", USE_intensity, 1);
        nh.param<double>("minimum_range", MINIMUM_RANGE, 0.5);
        nh.param<double>("maxmum_range", MAXMUM_RANGE, 90);
        nh.param<float>("mapping_line_resolution", lineResolution, 0.2);
        nh.param<float>("mapping_plane_resolution", planeResolution, 0.4);
        
        printf("--scanRegistration: scan line number %d \n", N_SCANS);
        printf("--scanRegistration: use intensity feature%d \n", USE_intensity);
        printf("--scanRegistration: minimum_range: %.3f, maxmum_range: %.3f \n", MINIMUM_RANGE, MAXMUM_RANGE);
        printf("--scanRegistration: line resolution: %f, plane resolution: %f \n", lineResolution, planeResolution);

        if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
        {
            printf("only support velodyne with 16, 32 or 64 scan line!");
        }

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, &ScanRegistration::laserCloudHandler, this);//  /points_raw /kitti/velo/pointcloud

        pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2> ("/velodyne_cloud_2", 100);
        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_sharp", 100);
        pubintenPointsSharp = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_inten", 100);

        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_flat", 100);
        pubGroundPointsFlat = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_ground", 100);
        pubScan = nh.advertise<sensor_msgs::PointCloud2>("/cloud_scan", 100);
        pubgroundparam = nh.advertise<ground_msg::groundparam>("/ground_param",100);
        feature_source_plane = nh.advertise<std_msgs::Float64MultiArray>("/feature_source_plane", 100);
        feature_source_sharp = nh.advertise<std_msgs::Float64MultiArray>("/feature_source_sharp", 100);
        feature_source_inten = nh.advertise<std_msgs::Float64MultiArray>("/feature_source_inten", 100);
    }

    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        if (!systemInited)
        {
            systemInitCount++;
            if (systemInitCount > systemDelay)
            {
                systemInited = true;
            }
            else
            {
                return;
            }
        }

        TicToc t_whole;

       // 读取雷达点云
        pcl::PointCloud<PointType> laserCloudIn;
        pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

        // 去除无效点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE, MAXMUM_RANGE);

        // 雷达点云投影
        int cloudSize = laserCloudIn.points.size();
        float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }

        bool halfPassed = false;
        int count = cloudSize;
        PointType point;
        int point_intensity = 0;
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
        std::vector<std::deque<int>> intensityScans(N_SCANS);
        for (int i = 0; i < cloudSize; ++i)
        {
            point.x = laserCloudIn.points[i].x;
            point.y = laserCloudIn.points[i].y;
            point.z = laserCloudIn.points[i].z;
            point_intensity = laserCloudIn.points[i].intensity;
            //计算点的仰角(根据lidar文档垂直角计算公式),根据仰角排列激光线号，velodyne每两个scan之间间隔2度
            float verticalAngle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (N_SCANS == 16)
            {
                scanID = int ((verticalAngle + 15) / 2 + 0.5); //仰角四舍五入(加减0.5截断效果等于四舍五入)
                if (scanID > (N_SCANS - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (N_SCANS == 32)
            {
                scanID = int ((verticalAngle + 92.0 / 3.0) * 3.0 / 4.0);
                if (scanID > (N_SCANS - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (N_SCANS == 64)
            {
                if (verticalAngle >= -8.83)
                {
                    scanID = int((2 - verticalAngle) * 3.0 + 0.5);
                }
                else
                {
                    scanID = N_SCANS / 2 + int((-8.83 - verticalAngle) * 2.0 + 0.5);
                }
                if (verticalAngle > 2 || verticalAngle < -24.33 || scanID > 50 || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else
            {
                printf("wrong scan number\n");
                ROS_BREAK();
            }

            // 计算点水平方向的转角，并根据其转角占lidar扫描角度的比值计算其时间占比
            float ori = -atan2(point.y, point.x);
            if (!halfPassed)
            {
                if (ori < startOri - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > startOri + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - startOri > M_PI)
                    halfPassed = true;
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > endOri + M_PI / 2)
                    ori -= 2 * M_PI;
            }

            //-0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
            float relTime = (ori - startOri) / (endOri - startOri);
            // 点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）
            // 根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
            point.intensity = scanID + scanPeriod * relTime;
            laserCloudScans[scanID].push_back(point); //将每个点放入对应线号的容器；
            intensityScans[scanID].push_back(point_intensity);
        }

        cloudSize = count;

        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        std::deque<int> intensity_num;
        std::vector<int> scanStartInd(N_SCANS, 0);
        std::vector<int> scanEndInd(N_SCANS, 0);
        for (int i = 0; i < N_SCANS; ++i)
        {
            scanStartInd[i] = laserCloud->size() + 5;
            *laserCloud += laserCloudScans[i];
            for(int j = 0; j < intensityScans[i].size(); j++)
            {
                intensity_num.push_back(intensityScans[i][j]);
            }   
            scanEndInd[i] = laserCloud->size() - 5;
        }
        std::deque<int> intensity_num2=intensity_num;
        
        // 计算点的深度
        for (int i = 0; i < cloudSize; ++i)
        {
            range_vec[i] = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + laserCloud->points[i].y * laserCloud->points[i].y + laserCloud->points[i].z * laserCloud->points[i].z);
        }
        // 计算点的入射角余弦值,PCA求取法向量实现
        for (int i = 5; i < cloudSize - 5; ++i)
        {
            if(range_vec[i]<2)
            {                
                Eigen::Vector3d point_a(laserCloud->points[i + 5].x, laserCloud->points[i + 5].y, laserCloud->points[i + 5].z);
                Eigen::Vector3d point_b(laserCloud->points[i - 5].x, laserCloud->points[i - 5].y, laserCloud->points[i - 5].z);
                Eigen::Vector3d point_c = (point_a + point_b)/2;
                Eigen::Vector3d point_now(laserCloud->points[i].x, laserCloud->points[i].y, laserCloud->points[i].z);
                Eigen::Vector3d point_norm = (point_a - point_b).cross(point_now - point_c);
                
                scan_angle[i] = (point_norm.dot(point_now))/(point_norm.norm() * point_now.norm());
                if(scan_angle[i] < 0)//归一化入射角为0-pi/2
                {
                    scan_angle[i] = -scan_angle[i];
                }
            }
        }     
        //平滑近处的强度
        for(int i = 5; i < cloudSize - 5 ; ++i)
        {
            if(scan_angle[i]<0.07&&range_vec[i]<2)
            {
               intensity_num[i] = 0.9*intensity_num2[i];
               for(int j = -5; j < 6; ++j)
               {
                if(j!=0)
                 {intensity_num[i] = intensity_num[i] + 0.005*intensity_num2[i+j];};
               }                
            }
        }       
        // 计算点的曲率
        for (int i = 5; i < cloudSize - 5; ++i)
        {
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
            float diffI = intensity_num[i-5] + intensity_num[i-4] + intensity_num[i-3] + intensity_num[i-2] + intensity_num[i-1] - 10*intensity_num[i] + intensity_num[i+1] + intensity_num[i+2] + intensity_num[i+3] + intensity_num[i+4] + intensity_num[i+5];

            float dis_factor = 2.0 / (1.0 + range_vec[i] / 20.0); // 增加深度因子，根据点深度对粗糙度进行缩放，深度越大，距离比例应该越小；
            if (dis_factor < 0.2) dis_factor = 0.2;
            cloudCurvature[i] = (diffX * diffX + diffY * diffY + diffZ * diffZ) * dis_factor;
            distance_source[i] = 0.5 + dis_factor;
            float inten_factor = 1;
            if(scan_angle[i]<0.07&&range_vec[i]<2)
            {
               inten_factor = scan_angle[i]*10 +0.6;
               intensityCurvature[i] = (scan_angle[i]+0.3)*diffI;                  
            }
            else
            {
                inten_factor = 3;
                intensityCurvature[i] = diffI;
            }
            other_source[i] = inten_factor;
            float diff_range = range_vec[i-5] + range_vec[i-4] + range_vec[i-3] + range_vec[i-2] + range_vec[i-1] - 10.0 * range_vec[i] + range_vec[i+1] + range_vec[i+2] + range_vec[i+3] + range_vec[i+4] + range_vec[i+5];
            cloudCurvature2[i] = abs(diff_range * dis_factor);
            
            //欧氏距离
            cloudSortInd[i] = i;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            //强度
            intenSortInd[i] = i;
            intenNeighborPicked[i] = 0;
            intenLabel[i] = 0;
            //地面
            groundcloudMarked[i] = 0;
        }      
        // 标记地面点
        ground_msg::groundparam groundmsg;
        size_t scanStart_ind = 0;
        pcl::PointCloud<PointType>::Ptr GroundPoints(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr GroundPointsFlat(new pcl::PointCloud<PointType>);
        std::vector<Eigen::Vector3d> nearGround;
        std::vector<double> laserweight;
        Eigen::Vector3d center(0, 0, 0);
        double groundweights = 0;
        int groundsize=0;
        
        for (int i = 0; i < groundScanInd; ++i)
        {
            for (size_t col_ind = 5; col_ind < laserCloudScans[i].points.size() - 5; ++col_ind)
            {
                int cloudInd = scanStart_ind + col_ind;
                float diff_range_th = 0.8 * (1.0 + i /(groundScanInd-1));
                float diff_range = abs(range_vec[cloudInd] - Ground_scan_range[i]);
                double groundweight = 1.5-i/(groundScanInd-1);

                // 如果点深度与激光坐标原点到地面深度的差值小于一定阈值，则认为是地面点
                if (diff_range < diff_range_th)
                {
                    if (laserCloudScans[i].points[col_ind].z < 0.3)
                    {
                        groundcloudMarked[cloudInd] = 1;
                        for (int n = -5; n < 5; ++n)
                        {
                            if (abs(range_vec[cloudInd + n] - range_vec[cloudInd]) < diff_range_th / 2)
                            {
                                groundcloudMarked[cloudInd + n] = 1;
                                GroundPoints->push_back(laserCloud->points[cloudInd + n]);
                                Eigen::Vector3d tmp(laserCloud->points[cloudInd + n].x,
                                                    laserCloud->points[cloudInd + n].y,
                                                    laserCloud->points[cloudInd + n].z);
                                center = center + groundweight*tmp;
                                groundweights= groundweights + groundweight;
                                nearGround.push_back(tmp);
                                laserweight.push_back(groundweight);
                                groundsize = groundsize + 1;
                            }
                        }
                    }
                }
            }
            scanStart_ind += laserCloudScans[i].points.size();
        }
        if(groundsize==0)
        {
            ROS_ERROR("groundsize000000000000000000000000!!!!!");
        }
        if(groundsize!=0)
        {
            center = center/groundweights;
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            double distance = 0;
            double groundsource1 = 0;
            double groundsource2 = 0;
            for (int j = 0; j < groundsize; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearGround[j] - center;
                covMat = covMat + laserweight[j] * tmpZeroMean * tmpZeroMean.transpose();
            }
            covMat=covMat/groundweights;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(0);
            unit_direction = unit_direction.normalized();
            if(center.dot(unit_direction)<0)//统一法向量方向
            {
                unit_direction = -unit_direction;
            }
            if (saes.eigenvalues()[1] > 6 * saes.eigenvalues()[0])
            { 
                std::cout<<"groundseg success!!!!"<<std::endl;
            }
            else
            {
                std::cout<<"fales"<<std::endl;
            }
            double distanceweight=0;
            for (int j = 0; j < groundsize; j++)
            {
                if(unit_direction.dot(nearGround[j])<0)
                {
                    ROS_WARN("!!!!!!!!");
                }
                distanceweight = 1-100*(abs(unit_direction.dot((nearGround[j]-center).normalized())));
                if(distanceweight<0)
                {
                    distanceweight=0.1;
                }
                groundsource1 += distanceweight;//归一化后将source限定在0-1
                distance += distanceweight * unit_direction.dot(nearGround[j]);
                
            }
            distance = distance/groundsource1;

            groundsource1 = groundsource1/groundsize;
            if((distance/laderH)>1.1||(distance/laderH)<0.9)
            {
                // ROS_WARN("ground point may erro");
                distance = laderH;
            }
            if(groundsource1<0.9)
            {
                distance = 0.9*laderH+0.1*distance;
            }
            // if(distance>0.6)
            // {
            //     ROS_ERROR("distance too lager");
            // }
            // std::cout << "distance: " << distance << std::endl;
            // std::cout << "source: " << groundsource1 << std::endl;	
            groundmsg.normx = unit_direction.x();
            groundmsg.normy = unit_direction.y();
            groundmsg.normz = unit_direction.z();
            groundmsg.vector1x = saes.eigenvectors().col(1).x();
            groundmsg.vector1y = saes.eigenvectors().col(1).y();
            groundmsg.vector1z = saes.eigenvectors().col(1).z();
            groundmsg.vector2x = saes.eigenvectors().col(2).x();
            groundmsg.vector2y = saes.eigenvectors().col(2).y();
            groundmsg.vector2z = saes.eigenvectors().col(2).z();
            groundmsg.distance = distance;
            groundmsg.source = 1-groundsource1;
        }
        // 排除容易被前景挡住的点以及陡斜面上的点;
        for (int i = 5; i < cloudSize - 5; ++i)
        {
            float depth1 = range_vec[i];
            float depth2 = range_vec[i+1];

            if (depth1 - depth2 > 0.04 * depth2)
            {
                cloudNeighborPicked[i - 5] = 1;
                cloudNeighborPicked[i - 4] = 1;
                cloudNeighborPicked[i - 3] = 1;
                cloudNeighborPicked[i - 2] = 1;
                cloudNeighborPicked[i - 1] = 1;
                cloudNeighborPicked[i] = 1;
            }
            else if (depth2 - depth1 > 0.04 * depth1)
            {
                cloudNeighborPicked[i + 1] = 1;
                cloudNeighborPicked[i + 2] = 1;
                cloudNeighborPicked[i + 3] = 1;
                cloudNeighborPicked[i + 4] = 1;
                cloudNeighborPicked[i + 5] = 1;
                cloudNeighborPicked[i + 6] = 1;
            }
        }

        // 提取特征点
        pcl::PointCloud<PointType2>::Ptr intenPointsSharp(new pcl::PointCloud<PointType2>);
        pcl::PointCloud<PointType>::Ptr intenPointsLessSharp(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType2>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType2>);
        pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType2>::Ptr surfPointsFlat(new pcl::PointCloud<PointType2>);
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        std::vector<float> plane_source;
        std::vector<float> sharp_source;
        std::vector<float> inten_source;
        for (int i = 0; i < N_SCANS; ++i)
        {
            if (scanEndInd[i] - scanStartInd[i] < 10)
                continue;
            
            // 将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点
            for (int j = 0; j < 6; ++j)
            {
                // 六等份起点：sp
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
                // 六等份终点：ep
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);
                std::sort(intenSortInd + sp, intenSortInd + ep + 1, comp_I);

                int largestPickedNum = 0;
                // 从曲率最大的点开始循环判断
                for (int k = ep; k >= sp; --k)
                {
                    int ind = cloudSortInd[k];
                    if (cloudNeighborPicked[ind] == 0 && groundcloudMarked[ind] != 1 && cloudCurvature[ind] > 0.1 && cloudCurvature2[ind] > 0.3)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 2;
                            PointType2 point_temp;
                            point_temp.x = laserCloud->points[ind].x;
                            point_temp.y = laserCloud->points[ind].y;
                            point_temp.z = laserCloud->points[ind].z;
                            point_temp.intensity = laserCloud->points[ind].intensity;
                            point_temp.normal_x = distance_source[ind] + 1;
                            cornerPointsSharp->push_back(point_temp);
                            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 21)
                        {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; ++l)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; --l)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                // 从曲率最小的点开始循环判断选取平面点
                for (int k = sp; k <= ep; ++k)
                {
                    int ind = cloudSortInd[k];
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.3 && cloudCurvature2[ind] < 0.4)
                    {
                        smallestPickedNum++;
                        if (smallestPickedNum <= 40)
                        {
                            cloudLabel[ind] = -1;
                            PointType2 point_temp;
                            point_temp.x = laserCloud->points[ind].x;
                            point_temp.y = laserCloud->points[ind].y;
                            point_temp.z = laserCloud->points[ind].z;
                            point_temp.intensity = laserCloud->points[ind].intensity;
                            point_temp.normal_x = distance_source[ind];
                            surfPointsFlat->push_back(point_temp);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        
                        for (int l = 1; l <= 5; ++l)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; --l)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 将挑选剩余的点全部归入less flat类别中
                for (int k = sp; k <= ep; ++k)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                    }
                }

                int largestPickedNum2 = 0;
                for (int k = ep; k >= sp; --k)
                {
                    int ind = intenSortInd[k];
                    if (intenNeighborPicked[ind] == 0 && groundcloudMarked[ind] != 1 && intensityCurvature[ind] >65 && cloudLabel[ind] != 2 && cloudLabel[ind] != 1)
                    {
                        largestPickedNum2++;
                        if (largestPickedNum2 <= 20)
                        {
                            intenLabel[ind] = 2;
                            PointType2 point_temp;
                            point_temp.x = laserCloud->points[ind].x;
                            point_temp.y = laserCloud->points[ind].y;
                            point_temp.z = laserCloud->points[ind].z;
                            point_temp.intensity = laserCloud->points[ind].intensity;
                            point_temp.normal_x = other_source[ind];
                            intenPointsSharp->push_back(point_temp);
                            intenPointsLessSharp->push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum2 <= 21)
                        {
                            intenLabel[ind] = 1;
                            intenPointsLessSharp->push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        intenNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; ++l)
                        {
                            float diffI = intensity_num[ind + l] - intensity_num[ind + l-1];
                            if (abs(diffI) > 35)
                                break;
                            intenNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; --l)
                        {
                            float diffI = intensity_num[ind + l] - intensity_num[ind + l+1];
                            if(abs(diffI) > 35)
                                break;
                            intenNeighborPicked[ind + l] = 1;
                        }
                    }
                }

            }
        }
        if(USE_intensity)
        {
            double sharp = cornerPointsSharp->size();
            double plane = surfPointsFlat->size();
            double inten = intenPointsSharp->size();
            double plane_sharp = sharp/plane;
            std::cout<<"sharp:"<<sharp<<"plane"<<plane<<"inten"<<inten<<std::endl;
            std::cout<<"the ratio of sharp and plane:"<<plane_sharp<<std::endl;
            if(plane_sharp<0.3)
            {
                *cornerPointsSharp += *intenPointsSharp;
            }
            sharp = cornerPointsSharp->size();
            plane = surfPointsFlat->size();
            inten = intenPointsSharp->size();
            plane_sharp = sharp/plane;
            std::cout<<"sharp:"<<sharp<<"plane"<<plane<<"inten"<<inten<<std::endl;
            std::cout<<"the ratio of sharp and plane:"<<plane_sharp<<std::endl;
        }
        

        // pcl::PointCloud<PointType> cornerPointsSharpDS, surfPointsLessFlatScanDS,GroundPointsDS;
        // pcl::VoxelGrid<PointType> downSizeFilterCorner;
        // pcl::VoxelGrid<PointType> downSizeFilterSurf;
        // pcl::VoxelGrid<PointType> downSizeFilterGround;

        // downSizeFilterCorner.setLeafSize(lineResolution, lineResolution, lineResolution);
        // downSizeFilterCorner.setInputCloud(cornerPointsSharp);
        // downSizeFilterCorner.filter(cornerPointsSharpDS);
        // cornerPointsSharp->clear();
        // *cornerPointsSharp += cornerPointsSharpDS;

        // downSizeFilterSurf.setLeafSize(planeResolution, planeResolution, planeResolution);
        // downSizeFilterSurf.setInputCloud(surfPointsFlat);
        // downSizeFilterSurf.filter(surfPointsLessFlatScanDS);
        // *surfPointsLessFlat += surfPointsLessFlatScanDS;

        // downSizeFilterGround.setLeafSize(planeResolution, planeResolution, planeResolution);
        // downSizeFilterGround.setInputCloud(GroundPoints);
        // downSizeFilterGround.filter(GroundPointsDS);
        // *GroundPointsFlat += GroundPointsDS;

        // 发布特征点
        // 对点云进行下采样后存储
        sensor_msgs::PointCloud2 laserCloudOutMsg;
        pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
        laserCloudOutMsg.header.frame_id = "aft_mapped";
        pubLaserCloud.publish(laserCloudOutMsg);

        sensor_msgs::PointCloud2 cornerPointsSharpMsg;
        pcl::toROSMsg(*cornerPointsSharp, cornerPointsSharpMsg);
        cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsSharpMsg.header.frame_id = "aft_mapped";
        pubCornerPointsSharp.publish(cornerPointsSharpMsg);

        sensor_msgs::PointCloud2 cornerPointsintenMsg;
        pcl::toROSMsg(*intenPointsSharp, cornerPointsintenMsg);
        cornerPointsintenMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsintenMsg.header.frame_id = "aft_mapped";
        pubintenPointsSharp.publish(cornerPointsintenMsg);
        
        sensor_msgs::PointCloud2 surfPointsFlatMsg;
        pcl::toROSMsg(*surfPointsFlat, surfPointsFlatMsg);
        surfPointsFlatMsg.header.stamp = laserCloudMsg->header.stamp;
        surfPointsFlatMsg.header.frame_id = "aft_mapped";
        pubSurfPointsFlat.publish(surfPointsFlatMsg);

        sensor_msgs::PointCloud2 groundPointsFlatMsg;
        pcl::toROSMsg(*GroundPoints, groundPointsFlatMsg);
        groundPointsFlatMsg.header.stamp = laserCloudMsg->header.stamp;
        groundPointsFlatMsg.header.frame_id = "world";
        pubGroundPointsFlat.publish(groundPointsFlatMsg);

        sensor_msgs::PointCloud2 scanMsg;
        pcl::toROSMsg(laserCloudScans[8], scanMsg);
        scanMsg.header.stamp = laserCloudMsg->header.stamp;
        scanMsg.header.frame_id = "base_laser";
        pubScan.publish(scanMsg);

        groundmsg.header.stamp = laserCloudMsg->header.stamp;
        groundmsg.header.frame_id = "aft_mapped";
        pubgroundparam.publish(groundmsg);

        std::cout<<"timm::"<<t_whole.toc()<<std::endl;
    }

    void removeClosedPointCloud(const pcl::PointCloud<PointType> &cloud_in, pcl::PointCloud<PointType> &cloud_out, float th1,  float th2)
    {
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            float dis = cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z;
            if(dis < th1 * th1)
                continue;
            if(dis > th2 * th2)
                continue;
            if(cloud_in.points[i].x < 0 && abs(cloud_in.points[i].y) < 0.5)
                continue;
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }

        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "scanRegistration");

    ScanRegistration SR;

    ros::spin();
    return 0;
}
