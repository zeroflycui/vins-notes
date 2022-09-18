#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
    bool state; // 是否三角化成功的标志
    int id;
    vector<pair<int,Vector2d>> observation; // 观测帧id和在帧中的坐标
    double position[3]; // 存放三角化后的坐标
    double depth;
};

// 使用ceres需要自己建立一个类，自己实现（）运算符的计算
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

        // ceres 要求自己定义重载（）运算符，计算残差
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p); // 旋转这个点（ceres中四元数的乘法）
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2]; // 其实就是Rcw * pw + tcw
        // 估计得到像机坐标系下的归一化坐标
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
        // 根现有观测建立残差
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}
    // 通过Create函数与constfuntion建立联系
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
        // 2是残差的维度，4，3，3分别是四元数、平移、三维点的维度
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y))); // 生成自动求导的对象
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};