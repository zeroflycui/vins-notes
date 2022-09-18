#include "initial_alignment.h"

// 求解陀螺仪零偏，同时利用求出来的零偏重新计算预积分
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        // 求最小值（qil * qlj * rij^-1(旋转的预积分)）
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R); // 得到下一帧到当前帧的旋转矩阵
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG); // 获得旋转对旋转零偏的雅克比矩阵（R对BG的偏导数）
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec(); // 下一帧预积分的逆与下一帧到当前帧的旋转
        // A△x = b 同时左乘雅克比矩阵的逆，乘矩阵的逆是为了保证矩阵的正定性为了后面的LDLT分解
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    // 求出陀螺仪的零偏
    delta_bg = A.ldlt().solve(b); // eigen的一种直接线性计算方法
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // 修正滑窗中的陀螺仪零偏
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;
    // 对all_image_frame中预积分量根据当前零偏重积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]); // 没有估计加速度零偏
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c; // 正切平面上的两个正交基
    Vector3d a = g0.normalized(); // 重力方向
    Vector3d tmp(0, 0, 1);
    // 确保a 和 tmp不同
    if(a == tmp)
        tmp << 1, 0, 0;

    /*
         ____b    
        |\
        | \tmp
        a
        (a.transpose() * tmp) 可以求出a 和tmp两个向量的夹角cos，其中a和tmp的模长为1
    */
    b = (tmp - a * (a.transpose() * tmp)).normalized(); // 求出与重力向量相垂直的向量
    c = a.cross(b); // 求出与重力向量和b向量互相垂直的向量
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

// 恢复重力
/*
@in: all_image_frame
@in: g，通过线性对齐求得的重力
@out: x
@out: g，修正后的重力
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1; // 由于重力的方向已经知道一维（指向地心方向），故状态向量的维度减少一维

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 最多迭代四次
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2); // B矩阵，表示正切平面的一对正交基
        lxly = TangentBasis(g0); // 得到重力的两个分量b、c
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9); // 重力只考虑两维
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b); // 求解Ax = b方程
            VectorXd dg = x.segment<2>(n_state - 3); // 求解出的W矩阵（表示b1\b2基所占的权重）
            g0 = (g0 + lxly * dg).normalized() * G.norm(); // 更新重力的方向
            //double s = x(n_state - 1);
    }   
    g = g0;
}

// 求解各帧的速度、枢纽帧的重力方向、以及尺度
// 得到更新方向后的重力g以及各帧速度、重力、尺度的解x
// out: x
// out: 返回线性对齐是否成功
// Xi = [vb1, vb2, ..., vbn, g, s]
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1; // 状态向量的维度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i; // 当前帧
    map<double, ImageFrame>::iterator frame_j; // 后一帧
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10); // 论文公式中的H矩阵
        tmp_A.setZero();
        VectorXd tmp_b(6); // 论文公式中的z值
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;  // s的系数除以100，相当于尺度s乘以100，故最终求出的尺度要除以一百   
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl; 

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 为了增加数值的稳定性
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b); // 求出速度、重力加速度、尺度
    double s = x(n_state - 1) / 100.0; // 得到尺度（因为尺度的系数除以100，为了保证数值稳定在计算过程中s需要乘100，故求出的尺度是真实尺度乘100后的值）
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4); // 得到重力
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    // 做一些检查
    // 如果求出的重力大小误差太大，或者尺度小于零，则认为求解失败
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }
    // 重力修复，得到更新重力方向后的重力，和速度，重力，尺度的新解x
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0; // 得到更新重力方向后的真实的尺度
    (x.tail<1>())(0) = s; // 更新为真实尺度
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// all_image_frame：每帧的位姿与对应的预积分量
// Bgs： 重新估计的陀螺仪的零偏
// g: 更新方向后的重力加速度
// x: 更新重力方向后的速度、重力、尺度的状态量
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    // 求解陀螺仪的零偏，通过零偏更新滑窗中的陀螺仪偏置以及更新所有帧的预积分量，因为在优化过程中是会估计零偏，如果零偏变化则要重新进行预积分计算
    // 为了避免反复预积分计算，将预积分使用一阶泰勒近似
    // 并没有求解加速度零偏，因为加速度零偏与重力加速耦合性很高，不易分解
    solveGyroscopeBias(all_image_frame, Bgs); 

    // 求解Xi = [vb1, vb2, ..., vbn, g, s]，最终得到的结果是修正重力方向后的结果
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
