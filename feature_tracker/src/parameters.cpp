#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES; // 配置文件对应的路径
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file"); // 读取对应的launch文件中的config_file对应的路径
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ); // 通过opencv读取config_file（.yaml文件）中的内容
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    // 特征点相关参数
    MAX_CNT = fsSettings["max_cnt"]; // 单针中的最大特征点数，为了控制系统的计算
    MIN_DIST = fsSettings["min_dist"]; // 两个特征点中的最短像素距离，为了保证提取特征点的均匀分布
    // 提取图像的分辨率
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    // 发送图像的频率
    FREQ = fsSettings["freq"];
    // 特征点提取算法的相关设置
    F_THRESHOLD = fsSettings["F_threshold"];  // 堆积约束ransac算法的阈值
    SHOW_TRACK = fsSettings["show_track"];   //
    EQUALIZE = fsSettings["equalize"];  // 是否进行均衡化

    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20; // 滑窗大小
    STEREO_TRACK = false; // 非双目
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();  // 释放打开的文件句柄


}
