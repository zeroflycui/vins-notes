#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}
// 外参、重投影置信度、延时设置
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i]; // 从参数列表读到的外参平移
        ric[i] = RIC[i]; // 从参数列表中读到的外参旋转矩阵
    }
    f_manager.setRic(ric);
    // 通过使用虚拟相机统一设置成重投影置信度（重投影的信息矩阵）
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();

    td = TD;
}

// 重置当前滑窗中的状态
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }
    // 外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    //solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

// 对imu数据进行处理，包括更新预积分量和提供优化状态量的初始值
/*
@in dt 两帧IMU的时间间隔
@in linear_acceleration imu测量的加速度
@in angular_velocity    imu测量的角速度
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // TODO: first_imu这个是在哪赋初值的？ 答：根据重启线程可知，first_imu初始值为false
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 滑窗中保留11帧，fram_count表示现在处理的第几帧，一般处于第11帧就保持不变了
    // 由于预积分是帧间约束，故第一帧预积分量是用不到的
    // 如果该帧没有被预积分，则新建预积分对象
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 会跳过第一帧数据进行处理
    if (frame_count != 0)
    {
        // 每来一帧imu数据都将会做一次预积分处理
        // TODO: 这求出的预积分在哪用到
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        // 主要用来做初始化用的，后面滑窗优化用不到了
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // 保存传感器数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 中值积分，更新滑动窗口中的状态量，本质是给非线性优化提供可信的初始值
        // 转换到世界坐标系下Twi
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; // 上一个时刻的加速度
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; // 通过中值定理得到当前时刻的角速度
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); // 更新旋转
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g; // 离散情景下当前时刻的加速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // 通过中值定理得到当前时刻的加速度值
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc; // 更新平移
        Vs[j] += dt * un_acc;   // 更新速度
    }
    // 更新IMU坐标系下的加速度和角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD; // 如果倒数第二帧是关键帧，则边缘化老的关键帧
    else
        marginalization_flag = MARGIN_SECOND_NEW; // 如果倒数第二帧不是关键帧，则移除

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // all_image_frame用来做初始化相关操作，它保留滑窗起始到当前的所有帧
    // 有一些会因为不是KF被MARGIN_SECOND_NEW,但是即使较新的关键帧被margin，他也会保留在这个容器中，因为初始化需要使用所有帧，而非只有关键帧
    ImageFrame imageframe(image, header.stamp.toSec()); // 创建一个图像帧对象
    imageframe.pre_integration = tmp_pre_integration; // tmp_pre_intergration 包含被边缘化关键帧的预积分信息
    // 这里就是简单的把图像和预积分绑定在一起，这里预积分量是两帧之间的值，滑动窗口中实际指的是两个关键帧之间的预积分量
    // 实际是用来做初始化的相关数据
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    // 帧间预积分的复位，为下面预积分做准备
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 没有外参初值，需要外参初始化
    // step2：外参初始化
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 这里标定IMU和相机的旋转外参的初值
            // 因为预积分是相邻的约束，因此这里得到的图像关联也是相邻的
            // 得到倒数第一帧和倒数第二帧两帧相邻关键帧观测到特征点在两帧中的坐标值集合
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;// 旋转外参
            // 进行外参标定，帧数至少不小于滑动窗口中维护的帧数
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1; // 标志位置成可信的外参初值
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // 保证有足够的帧数
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // 需要有可信的外参，同时距离上次初始化不成功至少相邻0.1s（因为如果上次初始化不成功，紧接着再次初始化很可能也会初始化失败）
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                // step3：VIO初始化，得到初始的速度、重力、尺度
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            // VIO初始化成功
            if(result)
            {
                solver_flag = NON_LINEAR; // 将标志设为非线性化，说明初始化完成
                //step4：非线性优化求解VIO
                solveOdometry();
                // step5: 滑动窗口
                slideWindow();
                // step6: 移除无效地图点
                f_manager.removeFailures(); // 移除一些无效的地图点
                ROS_INFO("Initialization finish!");

                // 记录此次优化后滑窗中第一帧和最后一帧的位姿，用于下次vio系统异常检测
                // 滑窗中最新帧的信息
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                // 滑窗中最老帧的信息
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else // 如果初始化失败则会将边缘化设置为最老帧（sfm步骤中发生）
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry(); // 非线性滑窗的求解
        ROS_DEBUG("solver costs: %fms", t_solve.toc());
        // 检测VIO是否正常
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            // 如果异常，重启VIO
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        // 记录此次优化后滑窗中第一帧和最后一帧的位姿，用于下次vio系统异常检测
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

// VIO初始化，通过sfm求得滑窗以及所有帧的位姿以及3d点，将滑动窗口中的P\V\Q恢复到世界坐标系，且和重力对齐
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //step1：check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g; 
        // 从第一帧开始（因为第零帧没有预积分值）
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt; // 两帧图像帧间的时间
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // 带重力加速度的加速度
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 因为第一帧的预积分没有计算
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 求方差
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 求标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 实际代码运行时并没有使用
        if(var < 0.25) // 如果标准差小于设定阈值则说明此段时间的运动比较集中，运动比较单调，认为运动激励不够不能完成初始化
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // step2：global sfm
    // 纯视觉的slam
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f; // 保存每个特征点的信息
    // 遍历所有特征点,得到初始的sfm所需的特征点信息（观测到特征点的帧id以及在每帧中的坐标）
    for (auto &it_per_id : f_manager.feature)
    {
        // 起始特征点对应帧的索引值（因为后面for循环有一个自增的运算）
        int imu_j = it_per_id.start_frame - 1; 
        SFMFeature tmp_feature; // sfm相关的特征点对象
        tmp_feature.state = false; // 没有被三角化成功
        tmp_feature.id = it_per_id.feature_id;
        // 遍历观察到当前特征点的所有帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point; // 相机平面内的归一化坐标（通过前端去畸变得到的结果）
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()})); // 观测帧id以及在帧观中的坐标
        }
        sfm_f.push_back(tmp_feature);
    }

    // 在滑窗中求出的旋转和平移
    // 两者都是出参
    Matrix3d relative_R;
    Vector3d relative_T;
    int l; // 枢纽帧
    // 确定枢纽帧以及最后一帧到枢纽帧之间的 R,T
    if (!relativePose(relative_R, relative_T, l)) 
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    GlobalSFM sfm;
    // sfm的求解滑窗中的位姿和三角化求解3D点（使用pnp、三角化以及局部BA优化，得到滑窗内更加精确的位姿以及3d地图点）
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points)) // 若视觉sfm求解失败，则边缘化最老帧
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    // stemp3:对所有关键帧进行pnp求解
    // 以上只是对滑动窗口中的关键帧进行求解，初始化需要对所有帧进行求解，因此下面通过KF来求解其他非KF的位姿
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    // 遍历所有帧，其中i是关键帧的索引
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // 当前帧为关键帧，可以直接求得R、T
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // 转化到相机坐标系下
            frame_it->second.T = T[i]; // 初始化不估计平移
            i++; // 更新为下一个图像帧的索引
            continue;
        }
        // 如果当前帧的时间戳大于当前帧的时间戳，则更新图像帧
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        // 最近的KF提供初值，Twc->Tcw
        // 并eigen -> opencv下
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector; // 3D点数组
        vector<cv::Point2f> pts_2_vector; // 2D点数组
        // 遍历当前非关键帧，得到该帧中三角化点的3d点和对应的2d点对儿
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first; // 特征点索引
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id); // 判断当前特征点，是否被三角化
                if(it != sfm_tracked_points.end()) // 当前特征点被三角化
                {
                    Vector3d world_pts = it->second; // 三角化后的3d点坐标
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>(); // 二维特征点坐标
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1); // 内参矩阵设为单位矩阵
        // 3D-2D点对数不可小于6组
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // 通过Pnp求解该帧的位姿
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        // 将opencv -> eigen
        // Tcw转到Twc
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose(); 
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // 更新当前帧的位姿
        frame_it->second.R = R_pnp * RIC[0].transpose(); // 考虑外参
        frame_it->second.T = T_pnp;
    }
    // 进行视觉惯性对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

// 求得滑窗内的陀螺仪的零偏、校正后的重力、每帧的速度、尺度s，并更新特征点带尺度的深度，最终对齐到世界坐标系下。
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // 得到新的陀螺仪零偏、更新方向后的重力加速度、包含速度、重力（切平面中正交基权重矩阵）、尺度的解x
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x); // 视觉惯性对齐
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 首先把对齐后的KF的位姿赋给滑窗中的值，Rwi，twc
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 根据有效的数据特征点数初始化这个动态向量
    VectorXd dep = f_manager.getDepthVector();  // 其实这里只是使用了深度的数量，因为下面全部将深度初始化为-1
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1; // 初始化地图点的深度置为-1
    f_manager.clearDepth(dep); // 赋值特征管理器把所有特征点的逆深度置

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero(); // 平移设置为0
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // 三角化所有特征点，此时得到的尺度依然是模糊的，因为还没有使用真实尺度
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0); // 取视觉惯性对齐时估计的真实尺度s
    // 将滑窗中的预积分重新进行计算（因为获得了修正重力方向后的重力、速度、以及尺度信息）Jk+1 = F * Jk   Qk+1 = F*Qk*F^T + V*n*V^T
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 枢纽帧的imu坐标系下把所有的平移状态对齐到枢纽帧
    // 已知T_l_c和Tic 得到T_l_i
    for (int i = frame_count; i >= 0; i--)
        // Ps[i] 表示第i帧相机到枢纽帧的平移，Rs[i]表示第i帧imu到枢纽帧的选转
        // (s * Ps[0] - Rs[0] * TIC[0]) 表示第0帧imu到枢纽帧的平移
        // Ps[i]最终表示在枢纽帧为参考系下第i帧imu到第0帧imu的平移
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); 
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    // 更新帧的速度
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            // 当时求得的速度是在IMU坐标系，现在转到世界坐标系（枢纽帧系）
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 把尺度模糊的3d点恢复到真实尺度下
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s; // 相机坐标系下的真实尺度
    }

    // 以上只是对齐到枢纽帧
    // 所有的P V Q全部对齐到第0帧，同时和对齐到重力方向
    Matrix3d R0 = Utility::g2R(g); // g是枢纽帧下的重力方向，得到枢纽帧到世界坐标系下的旋转R0 = R_w_l
    double yaw = Utility::R2ypr(R0 * Rs[0]).x(); // Rs[0] 表示imu帧到枢纽帧的旋转R_l_bk，得到第0帧imu坐标系到世界系下的yaw  
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 得到枢纽帧到世界坐标系下的旋转，同时yaw角变化到枢纽帧。为了保证yaw角为0是对应的是第一帧
    g = R0 * g; // 转换到世界坐标系下 g_w_bk
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0; // R w-l 
    for (int i = 0; i <= frame_count; i++)
    {
        // 全部与重力方向对齐，同时对齐到第零帧，并转换到世界坐标系下
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

// 通过共视点，找到枢纽帧，并计算出滑窗最后一帧到枢纽帧的R,T
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 从滑动窗口中的第零帧开始遍历
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); // 得到第i帧与最后一帧的共视点对儿在两帧中的坐标组合
        // 共视点对大于20
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                // 相机坐标系下的坐标
                Vector2d pts_0(corres[j].first(0), corres[j].first(1)); // 第i帧中的坐标
                Vector2d pts_1(corres[j].second(0), corres[j].second(1)); // 最后一帧中的坐标
                // 计算视差
                double parallax = (pts_0 - pts_1).norm(); // 计算两点之间的直线距离即：[(x1 - x2)^2 + (y1 - y2)^2]^(1/2)
                sum_parallax = sum_parallax + parallax;

            }
            // 计算平均视差
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 有足够的视差，再通过对极几何（本质矩阵）恢复最后一帧到第i帧的R,T
            // 460是虚拟焦距，使用虚拟相机为了达到统一的判断标准
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i; // 找到枢纽帧的索引
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}
// 求解里程计
void Estimator::solveOdometry()
{
    // 保证滑窗中帧数满了
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        // 先把可以三角化的特征点三角化，因为在视觉惯性初始化时得到了转到世界坐标系的P\V\Q
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization(); // 滑动窗口中的非线性优化，这里使用的是ceres的解析求导，因为里程计对实时性要求较高，解析求导比自动求导要快
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    // 传感器时延的变化量
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// double -> eigen 同时fix第一帧的yaw和平移，固定四自由度的零空间（为了防止零空间漂移）
void Estimator::double2vector()
{
    // 去除优化前第一帧的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    // 优化后的第一帧的位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
          
    // yaw角差                                            para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0)); // 要补偿的yaw角转换为旋转矩阵
    // 防止万向锁死锁的问题，如果俯仰角的绝对值与90°相差1°之外，则需要重新计算yaw角的补偿值
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        //（遍历滑动窗口中的每一帧，都乘上yaw角的补偿）
        // 保持第一帧的yaw不变
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        // 保持第一帧的位移不变
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);
        // 认为在滑窗内加速度和角速度的偏置是不变的
        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }
    // 外参值是不变的
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }
    // 重新设置各个特征点的逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info) // 对回环进行的处理
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        // 乘rot_diff目的是为了fix做的补偿
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();// 与上一次回环的的yaw角差
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        // T_loop_cur = T_loop_w * T_w_cur
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}
// 失败检测
bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2) // 追踪的特征点少于2个，则认为失败，需要重启
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5) // IMU加速度偏置大于 2.5，则认为失败，需要重启
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0) // IMU角速度偏置大于 1弧度，则认为失败，需要重启
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
   // last_P是只初始化完成后第一次对滑窗进行非线性优化后滑窗最后一帧的平移
   // 在进行边缘化后最后一帧和倒数第二帧的位姿是一致的
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    // 相邻帧间的运动大于5m，也需要重启
    if ((tmp_P - last_P).norm() > 5) // TODO: 为什么是相邻帧？？？答：边缘化后经过滑窗移动后倒数第一帧和倒数第二帧的位姿是相同的
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1) // 相邻帧在重力方向运动大于1m，也需重启
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R; // 最后两帧间的旋转
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0; // acos()返回的是弧度值【0-PI】要转换为角度值
    if (delta_angle > 50) // 两帧的旋转超过50度，也需重启
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

// 滑动窗口中进行非线性优化
void Estimator::optimization()
{
    // 借助ceres进行非线性优化
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0); // 核函数
    // step1: 定义待优化的参数块，类似g2o的顶点
    // 参数快1：滑窗中每帧的PQ（1*7） Vbabg（1*9）
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 由于姿态不满足正常的加法，也就是李群上没有加法，因此需要自己定义加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); 
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 参数块 2 ： 相机imu间的外参Tic
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            // 如果不需要优化外参就设置为fix，防止漂移
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 需要估计传感器的时延
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    // 实际上还有地图点，其实正常的参数块（可以实现正常加法的参数）不需要调用AddParameterBlock, 增加残差块接口时会自动绑定
    TicToc t_whole, t_prepare;
    // eigen -> double
    vector2double();
    // step2： 通过残差约束来添加残差块，类似g2o的边
    // 上一次的边缘化结果作为这一次的先验(如果当前滑窗中的参数块值与被边缘化的参数块差值变化与残差变化成正比，
    // 如果过大需要调整状态量进行调整，保证残差尽可能的小)
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    // imu预积分的约束
    // imu残差相关的变量 k和k+1时刻的{t,q,v,ba,bg}其中q是四元数形式
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        // 两帧关键帧之间预积分的时间过长就认为当前的预积分约束不可信
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1; // 滑窗中有效的特征
    // 视觉重投影的约束
    // 遍历每一个特征点，计算这个点与观察到这个特征点帧中所有特征点的重投影
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 进行特征点有效性的检查
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;
        // 观测到这个特征点的起始帧idx
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        // 特征点在第一帧下的归一化相机坐标系坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        // 遍历看到这个特征点的所有帧，将每帧的点与观测到第一帧中点的重投影残差添加到残差块中
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)// 自己和自己不能形成重投影
            {
                continue;
            }
            // 取出当前帧的归一化坐标
            Vector3d pts_j = it_per_frame.point;
            // 考虑传感器时间延时的是另一种重投影误差形式
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j); // 构造函数就是同一个特征点在不同帧的观测
                // 约束的变量是该特征点的第一个观测帧以及其他一个观测帧，加上外参和特征点逆深度
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());
    // 回环检测的约束，添加与回环帧形成的重投影约束
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            // 至少观测到该特征点的关键帧数大于等于2且观测到该特征点的起始关键帧要在倒数第二帧之前
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;// 更新特征点的id
            int start = it_per_id.start_frame; // 观测到该特征点的起始帧Id
            if(start <= relo_frame_local_index) // 起始帧小于等于回环帧在滑窗中的id，保证回环帧被观测到
            {   
                // 确保回环帧能有看到的地图点的可能性
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id) // TODO: 为什么用匹配点的z坐标和feature_id比较
                {
                    retrive_feature_index++;
                }
                // 该地图点也能够被回环帧观测到
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    // 构造一个重投影误差约束
                    // pts_j 表示回环帧， pst_i表示起始帧。
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0); // 回环帧对应的归一化点
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 特征点在观测到该特征点第一帧中的坐标
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }
    // step3 ceres优化求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR; // 稠密矩阵
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG; // 置信区间，求解下降梯度的方法选择DOGLEG
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD) // 边缘化老帧
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME; // 边缘化新帧操作较少，给他的优化时间多一些
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary); // ceres优化求解
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    // double -> eigen, 同时fix第一帧的yaw和平移，固定了四自由度的零空间
    // 因为IMU和图像都是提供的帧间约束，有可能会发生零偏
    double2vector();

    // step4 舒尔补（边缘化）
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        // 一个用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 这里类似手写高斯牛顿，因此也需要都转成double数组
        vector2double();
        // 关于边缘化有几点注意的地方
        // 1、找到需要边缘化的参数块，边缘化最老帧的参数包括：地图点，第0帧位姿，第0帧速度零偏
        // 2、找到构造高斯牛顿下降是跟这些待边缘化相关的参数块有关的残差约束。其中包含第0帧与第一帧间的预积分约束；与第0帧相关的重投影约束，以及上一次边缘化约束
        // 3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏

        // 上一次的边缘化结果
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            // last_marginalization_parameter_blocks是上一次边缘化对哪些当前参数块有约束
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
                // 将最老帧在残差块中的索引添加到边缘化容器中
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // 处理方式和其他残差块相同
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        // 预积分残差约束
        {
            // 如果两帧间预积分的时间跨度超过10则认为没有约束作用
            // 第0帧的预积分约束在实际计算中没有意义
            // 这里只考虑第0帧和第1帧的约束
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 跟构建ceres约束问题一样，这里也需要得到残差和雅克比
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); // 这里就是第0和1个参数块是需要被边缘化的
                marginalization_info->addResidualBlockInfo(residual_block_info); // 添加到残差块中
            }
        }
        // 遍历视觉重投影的约束
        {
            int feature_index = -1; // 符合条件的特征点
            // 遍历每个特征点
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size(); // 当前特征点被关键帧观察到的次数
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // i帧表示当前帧
                // 如果当前特征点不能够被第0帧看到，则说明该特征点不参与边缘化计算
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 该特征点在观察到该特征点帧中的坐标
                // 遍历观察到该特征点的所有关键帧，通过这个特征点建立和第0帧的约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j) // 如果相等，说明是同一帧，相同帧之间不能构成重投影约束
                        continue;

                    Vector3d pts_j = it_per_frame.point; // 取出当前关键帧中的特征点
                    // 根据是否约束延时确定残差阵
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        // loss_function是核函数，为了避免一些误匹配
                        // 状态量：i帧的位姿和第j帧的位姿，相机到IMU的外参，逆深度
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3}); // 这里表示第0帧和地图点被边缘化
                        marginalization_info->addResidualBlockInfo(residual_block_info); // 添加到残差块中
                    }
                }
            }
        }
        // 所有的残差信息都收集完成
        TicToc t_pre_margin;
        // 进行边缘化预处理，计算个参数块的残差和雅克比同时备份参数块的内容
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        // 进行边缘化处理
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());
        // 即将滑窗操作，因此需要记录新地址和对应的老地址
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 位姿和速度都要滑窗移动，
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        // 外参和时间延时不变
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // parameter_blocks实际上就是addr_shift的索引及搬进去的新地址。即与边缘化帧相关的参数块在滑窗中地址信息
        // 获得边缘化后的参数块大小，参数索引，参数块值
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        // last_marginalization_info是上一个边缘化的信息
        // 如果有上一次边缘化的信息，则将其删除掉
        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info; // 更新边缘化信息
        // 更新边缘化后剩余参数块的地址信息
        last_marginalization_parameter_blocks = parameter_blocks; // 代表该次边缘化对某些参数块形成的约束，这些参数块在滑窗滑动之后的地址
        
    }
    else // 边缘化最新的倒数第二帧（）
    {
        // 要求有上一次边缘化的结果的同时倒数第二帧被观察到
        // 有上一次边缘化结果且倒数第二帧与上次边缘化帧相关
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set; // 存放被边缘化参数块id的容器
                // 遍历上一次边缘化相关的参数块
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    // 速度零偏只会边缘第一个，不可能出现倒数第二个
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]); 
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1]) // 如果是最新倒数第二帧，添加到drop_set容器中
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 这里只会更新一下margin factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            // 开始移动滑窗
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1) // 跳过倒数第二最新帧
                    continue;
                else if (i == WINDOW_SIZE) // 将倒数第一帧地址对应倒数第二帧的参数，（相当于去掉倒数第二帧，倒数第一帧移动到倒数第二帧的位置上）
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else // 滑窗内其他帧的位置不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
     // 边缘化最老帧
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec(); // 最老帧的时间戳
        // 备份最老帧的位姿
        back_R0 = Rs[0]; 
        back_P0 = Ps[0];
        // 关键帧数量达到滑动窗口设定值
        if (frame_count == WINDOW_SIZE)
        {
            // 将滑动窗口中的变量向前滑动替代最老帧
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]); // 将滑窗内的旋转向前移动，将最老帧移动到最后

                std::swap(pre_integrations[i], pre_integrations[i + 1]); // 将滑窗内的预积分量向前移动

                dt_buf[i].swap(dt_buf[i + 1]); // 时间戳向前移动
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]); // 线性加速度向前移动
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]); // 角速度向前移动

                Headers[i] = Headers[i + 1]; // 更新滑动窗口的关键帧的头信息
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            // 处理最老帧的信息
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1]; // 将最老帧的头更新为倒数第二帧的头信息
            // 将最老帧的位姿和偏置更新为倒数第二帧的值
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
            // 删除最老帧的预积分值，重新创建预积分对象
            delete pre_integrations[WINDOW_SIZE]; // pre_integrations中存放的是预积分指针
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

             // 清空最老帧的时间戳、线性加速度、角速度
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0); // 根据时间戳找到最老帧的迭代器
                delete it_0->second.pre_integration; // 将其删除
                it_0->second.pre_integration = nullptr; // 删除后记得置空，否则会造成野指针
                // 遍历所有关键帧，将边缘化最老帧之前帧的预积分删除
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }
                // 删除最老帧以及之前的帧
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld(); // 将被边缘化帧的相关约束
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE) // 满足关键帧的数量和滑窗大小相匹配
        {
            // 将最后两个预积分观测合并成一个。因为要将次新帧边缘化
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
                // 倒数第二帧进行预积分
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                // 更新倒数第二帧的△t、线性加速度以及角速度的值为倒数第一帧的值
                dt_buf[frame_count - 1].push_back(tmp_dt); // 更新△t的值
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration); // 更新线性加速度的值
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }
            // 滑窗内，最后一帧的header、位姿、偏置移到倒数第二帧的位置
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count); // 去除倒数与第二帧相关的约束信息（删除倒数第二帧）
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth) // 非线性求解，此时已经完成初始化操作
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // back_R0 back_P0 是被边缘化帧的旋转和平移
        R0 = back_R0 * ric[0]; // 被移除原始时刻在相机坐标系下的旋转  a b c d ,此时指的是原始a时刻的情况
        R1 = Rs[0] * ric[0]; // 被移除帧位置当前时刻在相机坐标系下的旋转  b c d a ,此时指的是当前b时刻的情况
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        // 把被移除帧看到的地图点的管理权交给当前的最老帧
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

// 找到当前回环帧在滑窗内相同时间戳的关键帧，如果找到则认为当前回环帧是有效的，且将滑窗中帧的位姿赋值为回环帧的位姿。
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    // TODO 为什么要在滑窗中搜索回环相关的内容
    // 在滑窗中寻找当前帧，因为VIO送给回环结点的是倒数第三帧，因此，很有可能这个当前帧还在滑窗中
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        // TODO 为什么回环帧的时间戳会与滑动窗口中的回环帧相同
        // 假如回环帧的时间戳与滑动窗口中的时间戳相等，说明当前帧还没有被边缘化掉
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i; // 对应滑窗中的第i帧
            relocalization_info = 1; // 这是一个有效的回环信息
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j]; // 借助VIO优化回环帧位姿，初值先设为当前帧位姿
        }
    }
}

