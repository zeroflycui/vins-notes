#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

// 三角化共视特征点
// 出参：point_3d
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	// Ax = 0
	Matrix4d design_matrix = Matrix4d::Zero(); // A矩阵
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point; // 三角化后的三维齐次坐标向量
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>(); // 去矩阵奇异值分解后的右奇异矩阵V的最后一列
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

// 根据已经三角化的点和在当前帧中的匹配的2d点，通过pnp方法得到当前帧的位姿（R,T）
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
    // 遍历特征点，得到第滑窗中第i帧观测到已经三角化后的特征点，并得到在第i帧中的2d与3d特征点
	for (int j = 0; j < feature_num; j++)
	{
        // 因为Pnp是通过三维点和二位点来求解位姿的方法，故如果当前特征点没有被三角化，则跳过
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
        // 遍历观测到该特征点的帧，如果被当前帧i观测到，则保存该特征点的在该帧的2d点和sfm三角化后的地图点
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
                // 得到特征点在该帧中的坐标
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1)); // 将特征点坐标转换到opencv下
				pts_2_vector.push_back(pts_2);
                // 获取第i帧观测到特征点三角化后的点
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
    // 找到的3d、2d匹配点需要不小于15对
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
        // 3d、2d点对小于10对则认为pnp求解失败
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
    // 将eigen转换为opencv形势下
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec); // 将一个旋转矩阵转换为旋转向量
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1); // 内参矩阵
	bool pnp_succ;
    // 求解PNP
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
    // 将opencv下的值转化到eigen下
	cv::Rodrigues(rvec, r); // 将旋转向量转换为旋转矩阵
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
    // 得到当前帧的旋转和平移
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

/*
 * brief:三角化两帧共视的特征点
 * in：两帧中帧id、帧的位姿（Tcw）
 * out：三角化后的三维点
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1); // 不能对两帧同样的关键帧进行三角化
    // 遍历有特征点
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true) // 该特征点已经被三角化，sfm_f中的特征点状态默认是没有三角化的
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
        // 遍历观察到该特征点的所有关键帧，当两个目标帧同时观测到特征点时，获取该特征点在对应图像帧中的坐标点并跳出循环
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
            // 找到观察到当前特征点的关键帧frame0
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second; // 得到特征点的坐标
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}

			if(has_0 && has_1) break;
		}
        // fram0和frame1共同观察到该特征点，则进行三角化
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d); // 得到三角化点point_3d
            // 更新sfm中特征点的状态
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 根据已有的枢纽帧和最后一帧的位姿，得到滑动窗口中各帧位姿和3d点坐标，最后通过ceres进行优化，得到更加精确的位姿和3d坐标点
// 	out: q w_R_cam T w_R_cam 恢复出来的旋转和平移
//  c_rotation cam_R_w 
//   c_translation cam_R_w
// 	 relative_q[i][j]  j_q_i
// 	 relative_t[i][j]  j_t_ji  (j < i)
// 	out: sfm_tracked_points 三角化后的3D点
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size(); // 特征点数量
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
    // 枢纽帧设置为单位四元数可以理解为世界坐标系下的原点（即将枢纽帧作为参考帧）
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero(); // 平移设置为0
    // 通过枢纽帧到最后一帧的旋转平移，得到滑窗最后一帧到参考帧的旋转平移
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

    // 纯视觉slam处理的都是Tcw，而vins中维护的是Twc,因此需要转换为Tcw
	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

    // 将枢纽帧和最后一帧Twc转换成Tcw，包括四元数、旋转矩阵、平移向量和增广矩阵
    // 枢纽帧 Tlw
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // 最后一帧 Tjw
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 以上准备工作做好后，开始具体实现
	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
    // 三角化枢纽帧到最后一帧所有的共视点,得到3d点。
    // 根据得到的3d点通过PNP求解枢纽帧和最后一帧之间的关键帧的位姿Tiw，并根据求出的位姿三角化当前帧与最后一帧的共视点
	for (int i = l; i < frame_num - 1; i++)
	{
		// solve pnp
        // 通过pnp求解枢纽帧到最后一帧之间关键帧的位姿
		if (i > l)
		{
            // 将上一帧的R、T赋给当前帧作为R、T的初始值（Tli）
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
            // 通过Pnp求解当前帧到世界坐标系下的位姿(R_initial, P_initial)
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
            // 更新当前帧的位姿值
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
            // 更新当前帧的位姿 Tli(当前帧到参考帧的位姿)
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
        // 三角化当前关键帧与最后一帧关键帧所有共视的特征点
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	//3: triangulate l-----l+1 l+2 ... frame_num -2
    // 三角化枢纽帧和当前帧的共视点， 目的是为了三角化更多的特征点
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
    // 求解第零帧到枢纽帧前一帧的位姿和三角化点
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
        // 使用后一帧的R、T作为当前帧的初值
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
        // 通过pnp求解当前帧的位姿
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		// 三角化当前帧与枢纽帧的共视点
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
    // 三角化滑动窗口中没有被三角化的共视点
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
        // 当前特征点被观测到的帧不少于2帧
		if ((int)sfm_f[j].observation.size() >= 2)
		{
            // 三角化观察到当前特征点的初始帧和最后一帧的共视点
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
    // 以上通过pnp求得的位姿，三角化得到的3D点精度都不是很高
    // 下面通过使用ceres进行全局BA进行优化
	ceres::Problem problem;
    // 用于维护四元数的加法运算
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
        // ceres中所有坐标都是以二维数组构成的
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);

        // 将枢纽帧的R,T添加到problem，将最后一帧的R添加到problem中，但是不进行优化
        // 由于是单目视觉slam，会有尺度不固定的问题，因此可能会发生零空间漂移的问题
        // 因此将枢纽帧和最后一帧进行固定，增加漂移的约束
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

    // 视觉重投影构成的约束，因此遍历所有的特征点
	for (int i = 0; i < feature_num; i++)
	{
        // 需要特征点被三角化后才能继续
		if (sfm_f[i].state != true)
			continue;
        // 遍历观察到该特征点的帧，对这些帧建立约束
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first; // 帧索引
            // 计算代价函数（重投影误差）
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
            // 约束了该帧的位姿和3d地图点
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}
    // ceres进行求解
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR; // 线性求解的类型
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2; // 最大求解时间
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary); // 进行求解
	//std::cout << summary.BriefReport() << "\n";
    // 判断终止的条件是否是收敛结束的或者残差是否否足够小，否则认为求解失败
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
    // 优化结束需要把ceres下的double数组转换为正常的数值类型
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();  // 将旋转由Tcw -> Twc,转换为vins维护的坐标系下
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
    // 将平移由Tcw->Twc
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
    // 更新三角化后的三维点
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

