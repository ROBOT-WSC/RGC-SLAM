#pragma once
#include "rgc_slam/utility.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_, float var_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_),var(var_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);//oab面积
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;//ab

		// 点到线的距离，d = |向量OA 叉乘 向量OB|/|AB|;
		residual[0] = (nu.x() / de.norm())*T(var);
		residual[1] = (nu.y() / de.norm())*T(var);
		residual[2] = (nu.z() / de.norm())*T(var);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_, const Eigen::Vector3d last_point_b_, const float var_)
	{
		return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(
//	                                				             ^  ^  ^
//				                                	             |  |  |
//			                                       残差的维度 ____|  |  |
//			                                   优化变量q的维度 _______|  |
//			                                   优化变量t的维度 __________|
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, var_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	float var;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_), last_point_m(last_point_m_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m); // 向量AB AC的向量积（即叉乘），得到的是法向量;
		ljm_norm.normalize(); // 归一化后的法向量；
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm); // 点到面的距离：向量OA与法向量的点积；

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_)
	{
		return (new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_, float var_)
		: curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_), var(var_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = (norm.dot(point_w) + T(negative_OA_dot_norm))*T(var);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_, const double negative_OA_dot_norm_, const float var_)
	{
		return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, var_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
	float var;
};

template <typename T> inline
void QuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[0] = q[0];
	q_inverse[1] = -q[1];
	q_inverse[2] = -q[2];
	q_inverse[3] = -q[3];
};
//IMU r
struct DeltaRFactor
{
	DeltaRFactor(double q_x, double q_y, double q_z, double q_w, double q_var)
		: q_x(q_x), q_y(q_y), q_z(q_z), q_w(q_w), q_var(q_var){}

	template <typename T>
	bool operator()(const T* q_i_j, T* residuals) const
	{
        T relative_q[4];
		relative_q[0] = T(q_w); // ceres in w, x, y, z order
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j_tmp[4];
		q_i_j_tmp[0] = q_i_j[3]; // ceres in w, x, y, z order
		q_i_j_tmp[1] = q_i_j[0];
		q_i_j_tmp[2] = q_i_j[1];
		q_i_j_tmp[3] = q_i_j[2];

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(relative_q_inv, q_i_j_tmp, error_q); 

		residuals[0] = T(2) * error_q[1] / T(q_var);
		residuals[1] = T(2) * error_q[2] / T(q_var);
		residuals[2] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double q_x, const double q_y, const double q_z, const double q_w, const double q_var) 
	{
	  return (new ceres::AutoDiffCostFunction<DeltaRFactor, 3, 4>(new DeltaRFactor(q_x, q_y, q_z, q_w, q_var)));
	}

	double q_x, q_y, q_z, q_w;
	double q_var;
};
//IMU r
struct RelativeRFactor
{
	RelativeRFactor(double q_x, double q_y, double q_z, double q_w, double q_var)
		: q_x(q_x), q_y(q_y), q_z(q_z), q_w(q_w), q_var(q_var) {}

	template <typename T>
	bool operator()(const T* const w_q_i, const T* w_q_j, T* residuals) const
	{
		T relative_q[4];
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T w_q_i_tmp[4];
		w_q_i_tmp[0] = w_q_i[3]; // ceres in w, x, y, z order
		w_q_i_tmp[1] = w_q_i[0];
		w_q_i_tmp[2] = w_q_i[1];
		w_q_i_tmp[3] = w_q_i[2];

		T w_q_j_tmp[4];
		w_q_j_tmp[0] = w_q_j[3]; // ceres in w, x, y, z order
		w_q_j_tmp[1] = w_q_j[0];
		w_q_j_tmp[2] = w_q_j[1];
		w_q_j_tmp[3] = w_q_j[2];

		T i_q_w[4];
		QuaternionInverse(w_q_i_tmp, i_q_w);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j_tmp, q_i_j);

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q); 

		residuals[0] = T(2) * error_q[1] / T(q_var);
		residuals[1] = T(2) * error_q[2] / T(q_var);
		residuals[2] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double q_x, const double q_y, const double q_z, const double q_w, const double q_var) 
	{
		return (new ceres::AutoDiffCostFunction<RelativeRFactor, 3, 4, 4>(new RelativeRFactor(q_x, q_y, q_z, q_w, q_var)));
	}

	double q_x, q_y, q_z, q_w;
	double q_var;
};
//LiDAR P
struct DeltaPFactor
{
	DeltaPFactor(double p_x_, double p_y_, double p_z_, double p_var_)
		: p_x(p_x_), p_y(p_y_), p_z(p_z_), p_var(p_var_){}

	template <typename T>
	bool operator()(const T* p_i_j, T* residuals) const
	{
        T delta_p[3];
		delta_p[0] = T(p_x); // ceres in x, y, z order
		delta_p[1] = T(p_y);
		delta_p[2] = T(p_z);

		T p_i_j_tmp[3];
		p_i_j_tmp[0] = p_i_j[0]; // ceres in x, y, z order
		p_i_j_tmp[1] = p_i_j[1];
		p_i_j_tmp[2] = p_i_j[2];
		
		T error_p[3];
		error_p[0] = p_i_j_tmp[0]  - delta_p[0];
		error_p[1] = p_i_j_tmp[1]  - delta_p[1];
		error_p[2] = p_i_j_tmp[2]  - delta_p[2];

		residuals[0] = error_p[0] / T(p_var);
		residuals[1] = error_p[1] / T(p_var);
		residuals[2] = error_p[2] / T(p_var);

		return true;
	}

	static ceres::CostFunction* Create(const double p_x_, const double p_y_, const double p_z_, const double p_var_) 
	{
	  return (new ceres::AutoDiffCostFunction<DeltaPFactor, 3, 3>(new DeltaPFactor(p_x_, p_y_, p_z_, p_var_)));
	}

	double p_x, p_y, p_z;
	double p_var;
};
//IMU P
struct IMU_DeltaPFactor
{
	IMU_DeltaPFactor(double p_x_, double p_y_, double p_z_, double p_x2_, double p_y2_, double p_z2_, double p_var_)
		: p_x(p_x_), p_y(p_y_), p_z(p_z_), p_x2(p_x2_), p_y2(p_y2_), p_z2(p_z2_), p_var(p_var_){}

	template <typename T>
	bool operator()(const T* p_i_j, T* residuals) const
	{
        T delta_p[3];
		delta_p[0] = T(p_x); // ceres in x, y, z order
		delta_p[1] = T(p_y);
		delta_p[2] = T(p_z);

		T relative_p[3];
		relative_p[0] = T(p_x2); // ceres in x, y, z order
		relative_p[1] = T(p_y2);
		relative_p[2] = T(p_z2);

		T p_i_j_tmp[3];
		p_i_j_tmp[0] = p_i_j[0]; // ceres in x, y, z order
		p_i_j_tmp[1] = p_i_j[1];
		p_i_j_tmp[2] = p_i_j[2];
		
		T error_p[3];
		error_p[0] = p_i_j_tmp[0] + relative_p[0] - delta_p[0];
		error_p[1] = p_i_j_tmp[1] + relative_p[1] - delta_p[1];
		error_p[2] = p_i_j_tmp[2] + relative_p[2] - delta_p[2];

		residuals[0] = error_p[0] / T(p_var);
		residuals[1] = error_p[1] / T(p_var);
		// residuals[2] = error_p[2] / T(p_var);

		return true;
	}

	static ceres::CostFunction* Create(const double p_x_, const double p_y_, const double p_z_, const double p_x2_, const double p_y2_, const double p_z2_,const double p_var_) 
	{
	  return (new ceres::AutoDiffCostFunction<IMU_DeltaPFactor, 2, 3>(new IMU_DeltaPFactor(p_x_, p_y_, p_z_, p_x2_, p_y2_, p_z2_, p_var_)));
	}

	double p_x, p_y, p_z, p_x2, p_y2, p_z2;
	double p_var;
};

struct Ground_DeltaFactor
{
	Ground_DeltaFactor(ground_s g_last_, ground_s g_cur_, Eigen::Matrix<double, 4, 1> q_curr_, double p_var_)
		: g_last(g_last_), g_cur(g_cur_), q_curr(q_curr_), p_var(p_var_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> g_last_vector1{T(g_last.vector_1.x()), T(g_last.vector_1.y()), T(g_last.vector_1.z())};
		Eigen::Matrix<T, 3, 1> g_last_vector2{T(g_last.vector_2.x()), T(g_last.vector_2.y()), T(g_last.vector_2.z())};
		Eigen::Matrix<T, 3, 1> g_last_norm{T(g_last.vector_norm.x()), T(g_last.vector_norm.y()), T(g_last.vector_norm.z())};
		Eigen::Matrix<T, 3, 1> g_curr_norm{T(g_cur.vector_norm.x()), T(g_cur.vector_norm.y()), T(g_cur.vector_norm.z())};
		T g_last_distance = T(g_last.distance);
		T g_curr_distance = T(g_cur.distance);
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_w_curr{T(q_curr(3,0)),T(q_curr(0,0)),T(q_curr(1,0)),T(q_curr(2,0))};
		Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

		Eigen::Matrix<T, 3, 1> ground_norm_cur = q_last_curr*g_curr_norm;
		Eigen::Matrix<T, 3, 1> delta_t = q_w_curr*t_last_curr;
		// T ground_distance_cur = g_curr_distance + ground_norm_cur.dot(t_last_curr);
		T ground_distance_cur = g_curr_distance + delta_t(2,0);

		// residual[0] = delta_t(2,0)/T(p_var*0.0001);
		residual[0] = (g_last_distance - ground_distance_cur)/T(p_var/1000);
		residual[1] =  abs(g_last_vector1.dot(ground_norm_cur))/T(p_var*10);
		residual[2] =  abs(g_last_vector2.dot(ground_norm_cur))/T(p_var*10);
		// residual[3] = delta_t(2,0)/T(p_var*0.0001);
		return true;
	}

	static ceres::CostFunction* Create(const ground_s g_last_, const ground_s g_cur_, const Eigen::Matrix<double, 4, 1> q_curr_, const double p_var_) 
	{
	  return (new ceres::AutoDiffCostFunction<Ground_DeltaFactor, 3, 4, 3>(new Ground_DeltaFactor(g_last_, g_cur_ , q_curr_, p_var_)));
	}

	ground_s g_last, g_cur;
	Eigen::Matrix<double, 4, 1> q_curr;
	double p_var;
};

struct Ground_DeltaFactor_goable
{
	Ground_DeltaFactor_goable(ground_s g_last_, ground_s g_cur_, Eigen::Matrix<double, 4, 1> q_curr_, Eigen::Matrix<double, 4, 1> q_curr2_, Eigen::Matrix<double, 3, 1> t_curr_,double p_var_)
		: g_last(g_last_), g_cur(g_cur_), q_histoary(q_curr_), last_q(q_curr2_), last_t(t_curr_), p_var(p_var_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> g_last_vector1{T(g_last.vector_1.x()), T(g_last.vector_1.y()), T(g_last.vector_1.z())};
		Eigen::Matrix<T, 3, 1> g_last_vector2{T(g_last.vector_2.x()), T(g_last.vector_2.y()), T(g_last.vector_2.z())};
		Eigen::Matrix<T, 3, 1> g_last_norm{T(g_last.vector_norm.x()), T(g_last.vector_norm.y()), T(g_last.vector_norm.z())};
		Eigen::Matrix<T, 3, 1> g_curr_norm{T(g_cur.vector_norm.x()), T(g_cur.vector_norm.y()), T(g_cur.vector_norm.z())};
		T g_last_distance = T(g_last.distance);
		T g_curr_distance = T(g_cur.distance);

		Eigen::Quaternion<T> q_cur{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_last{T(last_q(3,0)),T(last_q(0,0)),T(last_q(1,0)),T(last_q(2,0))};
		Eigen::Matrix<T, 3, 1> t_cur{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> t_last{T(last_t(0,0)),T(last_t(1,0)),T(last_t(2,0))};

		Eigen::Quaternion<T> q_last_curr;
		Eigen::Matrix<T, 3, 1> t_last_curr;
		q_last_curr = q_last.conjugate()*q_cur;
		t_last_curr = q_last.conjugate()*(t_cur - t_last);

		// Eigen::Quaternion<T> q_w_curr{T(q_curr(3,0)),T(q_curr(0,0)),T(q_curr(1,0)),T(q_curr(2,0))};
		Eigen::Quaternion<T> q_w_curr{T(q_histoary(3,0)),T(q_histoary(0,0)),T(q_histoary(1,0)),T(q_histoary(2,0))};

		Eigen::Matrix<T, 3, 1> ground_norm_cur = q_last_curr*g_curr_norm;
		Eigen::Matrix<T, 3, 1> delta_t = q_w_curr*t_last_curr;
		// T ground_distance_cur = g_curr_distance + ground_norm_cur.dot(t_last_curr);
		T ground_distance_cur = g_curr_distance + delta_t(2,0);

		// residual[0] = delta_t(2,0)/T(p_var*0.0001);
		residual[0] = (g_last_distance - ground_distance_cur)/T(p_var/1000);
		residual[1] =  abs(g_last_vector1.dot(ground_norm_cur))/T(p_var*10);
		residual[2] =  abs(g_last_vector2.dot(ground_norm_cur))/T(p_var*10);
		// residual[3] = delta_t(2,0)/T(p_var*0.0001);
		return true;
	}

	static ceres::CostFunction* Create(const ground_s g_last_, const ground_s g_cur_, const Eigen::Matrix<double, 4, 1> q_curr_, const Eigen::Matrix<double, 4, 1> q_curr2_,const Eigen::Matrix<double, 3, 1> t_curr_, const double p_var_) 
	{
	  return (new ceres::AutoDiffCostFunction<Ground_DeltaFactor_goable, 3, 4, 3>(new Ground_DeltaFactor_goable(g_last_, g_cur_ , q_curr_, q_curr2_, t_curr_, p_var_)));
	}

	ground_s g_last, g_cur;
	Eigen::Matrix<double, 4, 1> q_histoary;
	Eigen::Matrix<double, 4, 1> last_q;
	Eigen::Matrix<double, 3, 1> last_t;
	double p_var;
};

template <typename T> inline
void Quaternion2EulerAngle(const T q[4], T ypr[3])
{
	// roll (x-axis rotation)
	T sinr_cosp = T(2) * (q[0] * q[1] + q[2] * q[3]);
	T cosr_cosp = T(1) - T(2) * (q[1] * q[1] + q[2] * q[2]);
	ypr[2] = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	T sinp = T(2) * (q[0] * q[2] - q[1] * q[3]);
	if (sinp >= T(1))
	{
		ypr[1] = T(M_PI / 2); // use 90 degrees if out of range
	}
	else if (sinp <= T(-1))
	{
		ypr[1] = -T(M_PI / 2); // use 90 degrees if out of range
	}
	else
	{
		ypr[1] = asin(sinp);
	}
	
	// yaw (z-axis rotation)
	T siny_cosp = T(2) * (q[0] * q[3] + q[1] * q[2]);
	T cosy_cosp = T(1) - T(2) * (q[2] * q[2] + q[3] * q[3]);
	ypr[0] = atan2(siny_cosp, cosy_cosp);
};

struct PitchRollFactor
{
	PitchRollFactor(double p, double r, double q_var)
		: p(p), r(r), q_var(q_var) {}

	template <typename T>
	bool operator()(const T* const q_i, T* residuals) const
	{
		T q_i_tmp[4];
		q_i_tmp[0] = q_i[3]; // ceres in w, x, y, z order
		q_i_tmp[1] = q_i[0];
		q_i_tmp[2] = q_i[1];
		q_i_tmp[3] = q_i[2];

		T ypr[3];
		Quaternion2EulerAngle(q_i_tmp, ypr);

		T e[2];
		e[0] = ypr[1] - T(p);
		e[1] = ypr[2] - T(r);

		residuals[0] = T(2) * e[0] / T(q_var);
		residuals[1] = T(2) * e[1] / T(q_var);

		return true;
	}

	static ceres::CostFunction* Create(const double p, const double r, const double q_var) 
	{
		return (new ceres::AutoDiffCostFunction<PitchRollFactor, 2, 4>(new PitchRollFactor(p, r, q_var)));
	}

	double p, r;
	double q_var;
};

struct GroundFactor
{
	GroundFactor(double var): var(var){}

	template <typename T>
	bool operator()(const T* ti, const T* tj, T* residuals) const
	{
		residuals[0] = (ti[2] - tj[2]) / T(var);

		return true;
	}

	static ceres::CostFunction* Create(const double var) 
	{
		return (new ceres::AutoDiffCostFunction<GroundFactor, 1, 3, 3>(new GroundFactor(var)));
	}

	double var;
};

template <typename T>
T NormalizeAngle(const T& angle_degrees)
{
	if (angle_degrees > T(180.0))
		return angle_degrees - T(360.0);
	else if (angle_degrees < T(-180.0))
		return angle_degrees + T(360.0);
	else
		return angle_degrees;
};

class AngleLocalParameterization
{
  public:
	template <typename T>
	bool operator()(const T* theta_radians, const T* delta_theta_radians, T* theta_radians_plus_delta) const
	{
		*theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);
		return true;
	}

	static ceres::LocalParameterization* Create()
	{
		return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>);
	}
};

template <typename T> 
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{
	T y = yaw / T(180.0) * T(M_PI);
	T p = pitch / T(180.0) * T(M_PI);
	T r = roll / T(180.0) * T(M_PI);

	R[0] = cos(y) * cos(p);
	R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
	R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
	R[3] = sin(y) * cos(p);
	R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
	R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
	R[6] = -sin(p);
	R[7] = cos(p) * sin(r);
	R[8] = cos(p) * cos(r);
};

template <typename T> 
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
	inv_R[0] = R[0];
	inv_R[1] = R[3];
	inv_R[2] = R[6];
	inv_R[3] = R[1];
	inv_R[4] = R[4];
	inv_R[5] = R[7];
	inv_R[6] = R[2];
	inv_R[7] = R[5];
	inv_R[8] = R[8];
};

template <typename T> 
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct FourDOFError
{
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
		: t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){}

	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);

		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);//旋转矩阵是酉矩阵

		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z, const double relative_yaw, const double pitch_i, const double roll_i) 
	{
		return (new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
			new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
};

//IMU预积分
class IMUFactor : public ceres::SizedCostFunction<15, 3, 4, 9, 3, 4, 9>
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
    }
    //IMU对应的残差，对应ceres的结构，需要自己计算jacobian
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d Bai(parameters[2][3], parameters[2][4], parameters[2][5]);
        Eigen::Vector3d Bgi(parameters[2][6], parameters[2][7], parameters[2][8]);

        Eigen::Vector3d Pj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Quaterniond Qj(parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]);

        Eigen::Vector3d Vj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Eigen::Vector3d Baj(parameters[5][3], parameters[5][4], parameters[5][5]);
        Eigen::Vector3d Bgj(parameters[5][6], parameters[5][7], parameters[5][8]);

//Eigen::Matrix<double, 15, 15> Fd;
//Eigen::Matrix<double, 15, 12> Gd;

//Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
//Eigen::Quaterniond pQj = Qi * delta_q;
//Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
//Eigen::Vector3d pBaj = Bai;
//Eigen::Vector3d pBgj = Bgi;

//Vi + Qi * delta_v - g * sum_dt = Vj;
//Qi * delta_q = Qj;

//delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
//delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
//delta_q = Qi.inverse() * Qj;

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        // 在优化迭代的过程中, 预积分值是不变的, 输入的状态值会被不断的更新, 然后不断的调用evaluate()计算更新后的IMU残差;
        
        // 问题：在迭代优化的过程中，预积分值是不变的，相当于imu值固定，不断调整系统状态p,v,q，使得残差最小化；
        // 但是imu预积分值始终会有误差，使得系统优化后始终带有imu积分误差;
        // 这是不是vins长时间累积误差较大，依赖回环进行校正的原因？（suyun）
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                            Pj, Qj, Vj, Baj, Bgj);
		// std::cout<<"1"<<residual<<std::endl;
		// std::cout<<"2"<<residuals[0]<<std::endl;
        //因为 Ceres 只接受最小二乘优化，也就是 min(𝑒 𝑇 𝑒) , 所以把 𝑃 −1 做 LLT 分解，即 𝐿𝐿 𝑇 = 𝑃 −1 ,𝑑 = 𝑟 𝑇 𝐿𝐿 𝑇 𝑟 = (𝐿 𝑇 𝑟) 𝑇 (𝐿 𝑇 𝑟)  
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
        //sqrt_info.setIdentity();   // 若写成“sqrt_info.setIdentity()”相当于不加权

        //这里残差 residual 乘以 sqrt_info，这是因为真正的优化项其实是 Mahalanobis 距离: d = r^T P^{-1} r，其中 P 是协方差。
        //Mahalanobis距离 其实相当于一个残差加权，协方差大的加权小，协方差小的加权大，着重优化那些比较确定的残差。
        residual = sqrt_info * residual; //为了保证 IMU 和 视觉參差项在尺度上保持一致，一般会采用与量纲无关的马氏距离
		// std::cout<<"3"<<residual<<std::endl;
		// std::cout<<"4"<<residuals[0]<<std::endl;
        //迭代优化过程中会用到IMU测量残差对状态量的雅克比矩阵，但此处是对 误差状态量 求偏导，即采用扰动方式；
        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(pre_integration->O_P, pre_integration->O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(pre_integration->O_P, pre_integration->O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(pre_integration->O_R, pre_integration->O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(pre_integration->O_V, pre_integration->O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(pre_integration->O_V, pre_integration->O_BG);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }

            //下面对四部分误差状态量求取雅克比矩阵;
			if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(pre_integration->O_P, pre_integration->O_P) = -Qi.inverse().toRotationMatrix();

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    // ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
                jacobian_pose_i.setZero();

                // jacobian_pose_i.block<3, 3>(pre_integration->O_P, pre_integration->O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(pre_integration->O_P, pre_integration->O_P) = Utility::skewSymmetric(Qi.inverse() * (0.5 * pre_integration->G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_pose_i.block<3, 3>(pre_integration->O_R, pre_integration->O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(pre_integration->O_R, pre_integration->O_P) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_i.block<3, 3>(pre_integration->O_V, pre_integration->O_P) = Utility::skewSymmetric(Qi.inverse() * (pre_integration->G * sum_dt + Vj - Vi));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    // ROS_BREAK();
                }
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[2]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(pre_integration->O_P, pre_integration->O_V - pre_integration->O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(pre_integration->O_P, pre_integration->O_BA - pre_integration->O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(pre_integration->O_P, pre_integration->O_BG - pre_integration->O_V) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(pre_integration->O_R, pre_integration->O_BG - pre_integration->O_V) = -dq_dbg;
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_speedbias_i.block<3, 3>(pre_integration->O_R, pre_integration->O_BG - pre_integration->O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(pre_integration->O_V, pre_integration->O_V - pre_integration->O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(pre_integration->O_V, pre_integration->O_BA - pre_integration->O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(pre_integration->O_V, pre_integration->O_BG - pre_integration->O_V) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(pre_integration->O_BA, pre_integration->O_BA - pre_integration->O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(pre_integration->O_BG, pre_integration->O_BG - pre_integration->O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                // ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }
			if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_pose_j(jacobians[3]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(pre_integration->O_P, pre_integration->O_P) = Qi.inverse().toRotationMatrix();

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                // ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_pose_j(jacobians[4]);
                jacobian_pose_j.setZero();

                // jacobian_pose_j.block<3, 3>(pre_integration->O_P, pre_integration->O_P) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(pre_integration->O_R, pre_integration->O_R) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_j.block<3, 3>(pre_integration->O_R, pre_integration->O_P) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                // ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[5]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(pre_integration->O_V, pre_integration->O_V - pre_integration->O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(pre_integration->O_BA, pre_integration->O_BA - pre_integration->O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(pre_integration->O_BG, pre_integration->O_BG - pre_integration->O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                // ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    IntegrationBase* pre_integration;

};

