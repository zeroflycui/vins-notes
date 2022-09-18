#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}
// 设置外参
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

// 获得可以可能被计算视差的特征点
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size(); // 观测到该特征点的帧数
        // 该特征点至少被两帧观测到，且至少被滑动窗口中的后两帧观察到
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

//在滑窗中判断倒数第二帧和倒数第三帧的视差，从而判断倒数第二帧是否为关键帧
// 
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0; 
    last_track_num = 0;
    /*
     * VINS中的特征点是通过list链表来管理的
     * 每个链表节点中都包含：特征点的id、起始帧的id、深度信息、求解状态、每帧中的属性
     * 每帧中的属性中的帧ID并不一定是从0开始的
     */
    // 遍历传入的帧信息，该帧中的特征点（没有在特征点链表中的点）添加到特征点链表中，同时将该特征点的相关帧信息，添加到特征点链表中特征点相关的帧信息中
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 创造一个帧属性对象

        int feature_id = id_pts.first; // 当前帧中的特征点id
        // 遍历特征点list容器，找到与对应特征点索引，使用lamda表达式完成条件判断
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
        // 特征点list容器没有当前特征点id
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); // 将当前特征点id和起始帧添加到特征点链表中
            feature.back().feature_per_frame.push_back(f_per_fra);  // 将特征点所属当前帧的相关属性（像素坐标、归一化相机坐标、特征点速度等）添加到帧属性链表的末尾
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra); // 将当前帧的帧属性添加到当前特征点对应的帧属性容器中
            last_track_num++; // 追踪到特征点数量数加一
        }
    }

    // 前两帧都设置为KF; 追踪特征点过少20个，认为当前帧与之前关键帧的关联性变弱，也认为是KF
    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        //  * * * * * *
        //        |<>|
        // 滑动窗口中实际上是判断滑窗中倒数第二帧是否为关键帧，而不是当前帧
        // 首先保证，特征点的起始帧至少能够被倒数第3帧看到，同时保证观测到该特征点的帧数能够覆盖到倒数第二帧，这样才能计算两帧间的视差
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // 计算符合条件特征点的视差和
            parallax_num++; // 更新满足计算视差条件的特征点个数
        }
    }
    // 如果没有满足计算视差的特征点，则直接认为倒数第二帧为KF，因为倒数第二帧和倒数第三帧没有关联，则会认为倒数第二帧为关键帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;// 平均视差大于等于最小视差值时，才认为倒数第二帧为关键帧，否则不是关键帧
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

// 得到同时被frame_count_l 和frame_cont_r帧看到的特征点在各自帧中的的坐标组合（匹配对集合）
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    //      |frame_count_l     frame_count_r|
    //  |start                              end|   特征点被观测到的图像帧的范围
    // 遍历特征点，找到能够被两帧同时观测到的特征点，并将特征点在两帧中的坐标维护到corres容器中
    for (auto &it : feature)
    {
        // 保证两帧需要观测到当前特征点
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame; // fram_count_l在特征点帧属性容器中的绝对索引值
            int idx_r = frame_count_r - it.start_frame; // fram_count_r在特征点帧属性容器中的绝对索引值

            a = it.feature_per_frame[idx_l].point; // fram_count_l在特征点帧属性容器中索引值对应的坐标

            b = it.feature_per_frame[idx_r].point; // fram_count_r在特征点帧属性容器中的索引值对应的坐标点
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2; // 特征点的深度求解失败
        }
        else
            it_per_id.solve_flag = 1; // 特征点的深度求解成功
    }
}
// 移除一些不能被三角化的点
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2) // 求解失败的点，去除
            feature.erase(it);
    }
}

// 将符合条件的特征点赋值逆深度
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

// 得到特征点的逆深度
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // 获得符合条件特征点的个数
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 利用观测到该特征点的所有位姿来三角化特征点
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历所有特征点，将符合条件的特征点三角化
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 只有符合条件才可以进行三角化
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0) // 因为所有初始深度都置为-1了，不为-1表示已经完成三角化
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        // Twi->Twc0 ，第一个观察到这个特征点的KF的位姿
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();
        // 遍历所有看到这个特征点的KF，得到相对于起始帧的位姿，然后构造，用A矩阵于三角化
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 得到该KF的相机位姿 Tw_cj
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            // 已知Tw_c0和T_w_cj -> T_c0_cj
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            // 因为视觉slam中维护的是Tcw
            // T_c0_cj -> T_cj_c0  c0相当于参考帧
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 构造超定方程中的其中两个方程 p_c = P * p_w
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        // 求解SVD，取V矩阵最后一维作为最终求解结果（世界坐标系下路标点的值）
        // 计算观测到当前特征点所有帧与第一次观察到的帧进行SVD求解，最终得到世界坐标系下的非齐次坐标（x,y,z,1）
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        // 求解齐次坐标系下的深度
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];
        // 得到的深度值实际上就是第一个观测到这个特征点的相机坐标系下的深度值
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;
        // 特征点的深度不应该太小，如果求得的深度过小，则设置为默认值
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH; // 深度太小就设定为默认值
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}
// 将特征点在被移除帧观测到的特征信息删除，同时更新该征点在边缘化后第一帧中的特征点
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // 遍历所有特征点，计算起始帧为第零帧且观测帧数超过两帧的特征点深度
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0) // 如果不是被移除帧看到，则直接将该地图点对应的起始帧id减一
            it->start_frame--;
        else // 被移除帧看到
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  // 取出观察到该特征点起始帧的归一化相机坐标
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 该点不再被原来的第一帧看到，因此删除该特征点在移除帧中的坐标、速度、时间间隔
            if (it->feature_per_frame.size() < 2) // 如果该地图点被观察到的帧小于两帧，则直接将其删除
            {
                feature.erase(it);
                continue;
            }
            else // 进行管辖权交接
            {
                // 保存该特征点的深度
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;  // 恢复移除帧在相机坐标系下观测到该特征点的坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // 特征点转到世界坐标系下。marg_R边缘化帧在由相机系到世界坐标系的旋转，marg_P在相机系到世界坐标系下的平移
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // 将世界坐标系下的点转到第j帧坐标系下。已知的是世界坐标系到第j帧的转换
                double dep_j = pts_j(2);
                if (dep_j > 0) // 检查深度是否有效
                    it->estimated_depth = dep_j; // 有效的话就得到在现在最老帧下的深度值
                else
                    it->estimated_depth = INIT_DEPTH; // 无效就设置为默认值
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}
// 这个还没有初始化结束，因此相比初始化结束是，不需要进行地图点新的深度换算，因为此时还没有进行视觉惯性对齐
// TODO: 优化for循环
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0) // 如果不是被边缘化帧看到，直接将该地图点的起始帧id减一，因为帧id是按顺序排
            it->start_frame--;
        else // 如果被边缘化帧看到，但是此时没有初始化完成，因此直接将地图点去掉
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 删除观察到该特征点的关键帧
            // 当没有帧观察到当前特征点，则将该特征点删除掉
            if (it->feature_per_frame.size() == 0) 
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count) // 如果当前地图点被滑窗中最后一个关键帧观测到，则将观测到其的起始帧id减一（因为最后一帧移动到倒数第二帧的位姿，则起始帧为原始最后一帧的情况需要减一）
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame; // 倒数第二帧在观测到该地图点到观察到该特征点起始帧的绝对距离
            if (it->endFrame() < frame_count - 1) // 如果观测到该特正点的最后一帧都小于滑窗中的倒数第二帧，则不需要任何操作（因为没有移动位置）
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j); // 从观测到该地图点的帧数组中删除被边缘化帧
            if (it->feature_per_frame.size() == 0) // 如果没有观测关键帧观测到当前地图点，则将其删除
                feature.erase(it);
        }
    }
}
// 计算倒数第三帧和倒数第二帧的视差
/*
@brief: 
@in it_per_id 特征点的信息
@in frame_cout 帧的id
@out 计算当前特征点在倒数第3帧和倒数第2帧间的坐标距离，作为视差
*/
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // 倒数第三帧
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // 倒数第二帧

    double ans = 0;
    Vector3d p_j = frame_j.point; // 归一化相机坐标系下的坐标

    // 归一化坐标的前两维：x y
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j; // 归一化相机坐标系的坐标差
    // 都是归一化坐标时，他们两个都是一样的。因为point中存的点就是归一化后的值
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 选取归一化前和归一化后两点间最小的距离作为新的视差，与旧的视差选最大值
    // 实际du、dv与du_comp、dv_comp都是相同的，因为存储的本身就是归一化后的坐标
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}