#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

// 标定IMU和相机之间的旋转外参，通过IMU和图像计算的旋转使用手眼标定计算获得
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    // 根据特征关联求解两个连续帧相机的旋转R（通过使用对极约束求解）
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix()); // imu坐标系下，有k+1到k（实际是两个图像帧的间隔）的旋转
    // ric是上一次求解的外参（由相机坐标系到惯性坐标系的转换）
    // 最后可以将IMU的旋转转换到相机坐标系下，当前帧到前一帧的旋转
    // TODO: Rc_g表示什么？？答：将在imu系下两个图像关键帧的旋转，变换到相机坐标系下
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);

    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]); // 相机坐标系下的旋转
        Quaterniond r2(Rc_g[i]); // IMU旋转通过外参转换到在相机坐标系中

        // 计算两者间的旋转差值，通过调用eigen中的augularDistance（）
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);
        // 构造一个简单的核函数，相差大于5°时所占权重越小，小于5度时权重为1
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R); // 构造完A矩阵
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV); // 对A矩阵进行奇异值分解
    Matrix<double, 4, 1> x = svd.matrixV().col(3); // 选择右奇异的最后一列（a3）作为结果
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse(); // 因为求得的是qc,b，而外参的形式为qb,c
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>(); // 得到奇异值奇异值的后三个，用于判断此次的奇异值分解是否正常
    // 倒数第二个奇异值需要大于设定的阈值，因为旋转是三自由度，因此前三个奇异值需要稍大于0，最后一个奇异值往往是趋近于0的，否则就会变为四自由度。
    // 同时奇异值是由大到小排列，因此只需要检测倒数第二个奇异值是否大于设定阈值即可
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric; // 将得到的旋转外参付给结果值
        return true;
    }
    else
        return false;
}

// 已知匹配的特征点对，通过对极几何约束求解外参R,T
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    // 至少要有九对匹配的特征点对儿，特征点对越多得到的外参置信度越高
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            // 得到相邻两帧的特征点坐标
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));  
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));  
        }
        // 这里使用的是相机坐标系下的归一化坐标，因此这个函数得到的是E矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2); // E矩阵奇异值分解，得到两组R,t

        // 旋转矩阵的行列式应该是1，如果是-1则取反
        // double或者float类型不能直接比较大小，应该相减然后和阈值比较，最终判断两个值是否相同
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2; // 得到置信度高的那个旋转矩阵

        // 以上步骤得到的旋转矩阵R21是从前一帧到当前帧的旋转，是以当前帧为参考帧
        // 下面需要转换到R12，从当前帧到上一帧的旋转
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); // R21的转置就是R12
        return ans_R_eigen; // 得到和IMU相同的旋转矩阵R12
    }
    return Matrix3d::Identity();
}

// 通过三角化来检测R,t是否合理
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud; // 4×N的矩阵
    // P和P1都表示在世界坐标系下的位姿
    // 其中一帧设置为单位矩阵
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // 第二帧设置为R,T对应的位姿
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    // 求得三角化后的三维坐标
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    // 遍历三角化后的三维坐标
    for (int i = 0; i < pointcloud.cols; i++)
    {
        // 取出三角化后的坐标的最后一维，用于后面将坐标转换为齐次坐标
        double normal_factor = pointcloud.col(i).at<float>(3);

        // 通过世界坐标系下的位姿和齐次坐标，最终转化为相机坐标系下的坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        // 通过判断深度是否大于零，来决定所求三角化点是否合理（因为通过奇异值分解会得到多对解，通过判断深度是否大于零可以进行简单的判断）
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols; // 返回当前外参R,T的置信度
}

void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    // 具体参考多视图几何
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
