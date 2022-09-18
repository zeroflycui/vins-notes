#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf; // 存放imu消息的队列
queue<sensor_msgs::PointCloudConstPtr> feature_buf; // 存放特征消息的队列
queue<sensor_msgs::PointCloudConstPtr> relo_buf; // 存放回环信息的队列
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
// 偏差
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0; // 上一时刻的加速度测量值
Eigen::Vector3d gyr_0; // 上一时刻的加速度测量值
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// 通过相邻两帧imu的测量值通过中值定理以及通过物理方法得到临时的P、V、Q
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    // 如果是第一次接受IMU信息，则只记录时间戳即可，不做任何操作
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time; // 当前imu帧与上一帧imu帧的时间间隔
    latest_time = t;

    // 得到IMU的测量加速度值
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    // 得到IMU的测量角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    // 上一时刻世界坐标系下的加速度, tmp_Q（Rwi）acc_0表示imu坐标系下的加速度
    // TODO: tmp_Q的转换方向是什么？？？答：imu坐标系转到世界坐标系下
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g; // 加速度的真实值
    // 中值定理得到当前时刻IMU帧的角速度
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 更新当前IMU帧的旋转
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    // 当前时刻世界坐标系下IMU的加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
    // 世界坐标系下的加速度中值积分的值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 经典物理中位置和速度的更新方程 y = y0 + v*t + 1/2*a*t^2   v = v0 + a*t
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    // 更新IMU帧的加速度和角速度，作为下一次的加速度和角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}
// 用最新VIO结果更新最新IMU对应的位姿
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf; // 遗留的IMU的buffer，因为下面需要pop，所以copy了一份（因为是多线程，为了线程安全所以进行了copy）
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front()); // 更新最新IMU位姿、偏置信息

}
// 获得匹配好的图像IMU组（将IMU数据和图像数据进行对齐）
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        // 加入存储imu或图像ros信息的容器为空，则返回。因为这个函数是为了获取一组imu和img信息
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        
        // imu_msg 通过使用queue容器存储
        // imu ******
        // img         ****
        // imu数据最末端的时间戳小于图像前端的时间戳，说明imu数据还没与到，imu数据和图像数据还没有对齐（如上简图所示）
        // estimator.td表示估计的传感器时延
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // imu       *****
        // img   *********
        // imu的前端时间戳大于或等于图像前端的时间戳，此时会将一部分图像帧丢弃。没有IMU数据的图像帧没有操作的意义，故把当前图像帧剔除
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop(); // 将多余的图像帧删除与IMU对齐
            continue;
        }

        // 此时就保证了图像数据前一定有imu数据
        // imu  ****************
        // img       **    **
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front(); // 取出一个图像消息
        feature_buf.pop(); // 读出后删除该图像消息

        // 一般第一帧不会严格对齐，在后面的图像帧会和IMU数据严格对齐；在后续计算的过程中不会使用到一帧图像与IMU对齐的数据
        std::vector<sensor_msgs::ImuConstPtr> IMUs; // 存放图像帧之前的imu消息作为两个图像帧间的imu数据
        // 将图像数据前端时间戳之前的IMU数据从imu_buf中剔除，转进IMUs容器中作为与图像对齐的IMU数据
        // 因为IMU的频率要高于图像的频率，两帧图像之间会有多帧imu数据
        // 故一组对齐的imu和img数据中，会有一帧图像帧和多帧imu帧数据。
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front()); // 向IMUS中添加imu消息
            imu_buf.pop();
        }

        // 因为IMU数据是离散的，会发生图像数据正巧在IMU数据间隔，这样需要将图像数据后最近的一个IMU数据(^所指的位置)传入IMUs容器中
        // imu ****   *****
        // img      * ^     *
        IMUs.emplace_back(imu_buf.front()); //  取出imu数据，但是并没有删除imu_buf中的信息
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg); // 此时取到了一组对齐的IMU和图像数据
    }
    return measurements;
}
// imu消息存进队列imu_buf中，同时按照IMU频率以及后端当前的位姿估计下一时刻的位姿，同时还可以提升里程计的频率（与imu的频率相当）
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 若当前IMU信息的时间戳小于上次的时间戳，则认为IMU信息紊乱，需要重新接受新的imu信息
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    // 线程锁和条件变量参考https://www.jianshu.com/p/cldfald40f53
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one(); // 条件变量，触发VIO线程

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state); // 互斥锁，RAII机制，构造上锁，析构解锁
        predict(imu_msg); // 通过传来的imu信息得到当前IMU帧的P、V、Q的预测值
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        // 只有初始化完成才发送当前结果
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// 将前端信息存进buffer
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    // 线程安全的一些操作
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one(); // 条件变量，触发VIO线程
}
// 将vins估计器复位；应对前端跟踪失败的问题
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        // 清空特征点和IMU数据
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// 把收到的消息全部放到回环buf中去
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
// VIO线程
void process()
{
    while (true)  // 这个进程会一直循环下去
    {
        // 存放IMU和图像帧对齐的数据的容器
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        // 等待被唤醒（imu回调函数、特征回调函数，通过条件变量唤醒该线程）
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0; // 获取一组对齐的IMU和图像帧对齐的数据
                 }); // 条件变量
        lk.unlock();

        m_estimator.lock(); // 进行后端求解，不能和复位重启冲突
        // 遍历对齐的IMU和图像对齐容器（图像只有一帧数据，IMU有很多数据）
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历IMUs容器，对每帧imu数据进行预积分处理（除了第零帧）
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // estimtor.td是指计算的传感器的时延，用于降低传感器传输数据过程中的影响
                // 如果图像帧前面有IMU数据
                // imu ****  ****
                // img     * ^
                if (t <= img_t)
                { 
                    if (current_time < 0) // current_time 初始值为-1
                        current_time = t;
                    double dt = t - current_time; // 相邻两帧IMU数据间的时间间隔
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    // 获取imu的加速度和角速度
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else // 这就是IMU帧最后一个imu数据（图像帧后面的一个较近的IMU数据），需要做一个简单的线性插值
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // TODO: 为什么要这样构造。线性插值，(x-x0)/(y-y0) = (x1-x0)/(y1-y0) 最后整理成
                    // y = (x1-x)/(x1-x0)*y0 + (x-x0)/(x1-x0)*y1
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            // 回环检测
            sensor_msgs::PointCloudConstPtr relo_msg = NULL; // 前三维是平移，接下来的四维是旋转（四元数），最后一维是帧id
            // 判断是否有重定位信息,如果有则除取出回环队列中的最后一个回环信息
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec(); // 回环帧时间戳
                //遍历取出的回环帧的特征点
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    // 取出最新回环帧，将回环帧中的特征点放入匹配点集合中
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r); // 将回环信息更新到estimator类中的回环信息中
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // image容器中参数依次为特征点id、相机id、归一化坐标像素坐标速度构成的一维矩阵xyz_uv_velocity
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            // 遍历前端传过来的特征信息，重新计算得到特征点id和相机id，归一化坐标、像素坐标和速度存入到image容器中
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5; // 进行一个四舍五入的操作
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                // 去畸变后归一化相机坐标
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                // 特征点像素坐标
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                // 特征点速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1); // 检查是否为归一化后的坐标
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            // 处理图像的数据
            estimator.processImage(image, img_msg->header);

            // 一些打印以及topic的发送
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL) // 如果有回环帧
                pubRelocalization(estimator); 
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // 当前状态是非线性求解，说明已经初始化完成，处在正常vio的状态
            update(); // 进行状态更新
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter(); // 设置参数外参
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // 注册一些publisher
    registerPub(n);

    // 订阅IMU话题接受imu信息，没帧imu到来都会调用回调函数
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // 接受前端视觉光流结果
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // 接受前端重启命令
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // 回环检测的fast relocalization响应
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    // 核心处理线程  
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
