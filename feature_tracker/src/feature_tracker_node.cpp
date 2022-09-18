#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time; 
int pub_count = 1; // 发布的帧数
bool first_image_flag = true; // 第一帧的时间戳
double last_image_time = 0; // 最新帧的时间戳
bool init_pub = 0;

// 图像回调函数
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 如果是第一帧只更新图片帧的时间戳
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    // 检查当前帧的时间与第一帧时间差过大或这当前帧时间小于前一帧的时间，认为视频流异常
    // 因为视觉前端是否的是光流跟踪法，如果时间差过大会导致光流跟踪失败。由于没有描述子匹配，因此对时间戳要求较高
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }

    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    // 控制向后端发送图像的频率= 发送的次数/时间间隔   round()函数表示对小数取整，即四舍五入
    // 如果发送给后端的频率过快，可能会造成后端来不及处理图像数据，从而导致系统崩溃
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ) // FREQ默认是10HZ
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 当这段时间的频率和预设频率相似时，就认为这段时间很棒，需要重置时间间隔和发布帧数，避免delta太大
        //（即在同样的时间内进入了更多帧数据，但是依旧没有超过预定频率，而是无限接近预定频率）
        // 如果不进行这样的处理，可能会导致在短时间内有大量帧数据进入后端，造成后端不能及时处理数据，导致系统崩溃
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    // 即使当前帧不发布也要进行正常的光流跟踪，因为光流跟踪需要前后两针图像之间的移动要尽可能的小

    // 因为opencv并不能处理ros的图像，需要通过cv_bridge将图像进行转化为CV::MAT类型
    // 使用了copy的方式将ros图像转为opencv格式，这样可以对返回的cv图像进行自由修改
    cv_bridge::CvImageConstPtr ptr; // ros cv_bridge格式的指针
    if (img_msg->encoding == "8UC1") // 灰度图
    {
        // ros图像类型
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";

        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    // 光流跟踪，得到去畸变后的特征点坐标
    cv::Mat show_img = ptr->image; // 得到cv格式的图像,用于可视化展示
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK) // 单目
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec()); // 参数为图像的信息和对应的时间戳信息
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // 为新提取的特征点赋上id值
    // 直到i大于等于特征点的最大值时，跳出for循环
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i); // 更新新提取特征点的id（id值是全部特征点中的位置）
        if (!completed)
            break;
    }

    // 给后端发送数据
   if (PUB_THIS_FRAME)
   {
        pub_count++; // 发布数加一
        // 使用ros信息的格式进行信息存储
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud); // 声明一个PointCloud话题
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            // 往后端发送的数据
            auto &un_pts = trackerData[i].cur_un_pts;// 去畸变的归一化相机坐标
            auto &cur_pts = trackerData[i].cur_pts;  // 像素坐标
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity; // 速度
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                // 只发布追踪次数大于1的数据，因为等于1的特征点没办法构成重投影约束，也没办法进行三角化
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    // 将去畸变后的相机坐标发送给后端
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i); // 为了区分单目和双目相机特征点id
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points); // 将前端里程计得到的信息（特征点坐标、特征点速度）通过publisher发布出去

            // 可视化操作
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    // 通过opencv的函数绘制以特征点为中心的点，点的颜色与跟踪次数相关
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker"); // ros初始化
    ros::NodeHandle n("~"); // ros初始化句柄，创建当前的命名空间为~
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);  // 设置ros 日志的级别
    readParameters(n); // 读取配置文件函数

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); // 读取相机的相关内参

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // 向roscore注册订阅这topic，收到一次图像的message就执行一次回调函数
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback); // 订阅原图像节点，接受原始图像信息的传入

    // 注册一些publisher，将处理后的消息发送出去
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    // spin函数代表这节点开始循环查询是否有图像话题进入，如果没有则阻塞。
    // 只有对应的话题进入，一系列的订阅、回调和发布才会开始进行
    ros::spin(); 
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?