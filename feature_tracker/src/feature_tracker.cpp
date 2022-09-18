#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// 判断点是否在图像内
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 通过点的状态，对点集的大小进行重新设计
// 通过使用双指针，避免了从新开辟空间
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}
// 给现有的特征点设置mask，目的是为了特征点均衡化，得到均衡化后的特征点坐标、id、被追踪次数
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone(); 
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // 将MASK设置为相同大小的白色灰度图
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // 将当前帧中的特征点信息、id以及被跟踪的次数，存入特征点追踪容器中
    // track_cnt[i]表示对应特征点被追踪到的次数，forw_pts[i]表示特征点的位置，ids[i]当前特征点的id
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 排序算法，将cnt_pts_id vector按照 特征点被跟踪的次数由大到小进行排序。
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    // 清空当前帧中的数据信息：特征点信息、id、被追踪次数
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        // 将符合条件的特征点坐标信息、id、被追踪次数，存入对应容器中
        // 通过检查特征点对应mask图像的位置是否为白色点，来判断是否将选用该特征点。（类似辅助动态标记，规定特征点的提取范围）
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // 在图像mask上，以it.second.first特征点为圆心，像素的最短距离MIN_DIST为半径的圆内全部设为0，-1是把圈内全部涂满的意思
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); // 将特征点周围设置为不可取特征点的区域
        }
    }
}

// 将新提取的特征点维护起来，新特征点的id统一设置为-1
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1); // 将新特征点的id统一置为-1
        track_cnt.push_back(1); // 因为是当前帧中提取的特征点，因此追踪次数为1
    }
}
/*
 * brief: 读取图像参数
 *
 * in： _img 输入图像
 * in： _cur_time 图像的时间戳
 * 1、图像均衡化预处理
 * 2、光流跟踪
 * 3、提取新的特征点（如果发布）
 * 4、所有特征点取畸变、计算速度
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // 图像预处理，均衡化
    if (EQUALIZE)
    {
        // 如果图像太暗或者太亮，提取特征点会比较难，所以均衡化，提升对比度，方便提取角点
        // 调用opencv中的直方图均衡化算法，目的是增强图像的对比度同时抑制噪声
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // 3.0是颜色对比阈值，第二个参数为进行均衡化网格的大小
        TicToc t_c;
        clahe->apply(_img, img); // 得到CLAHE之后的图像img
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // 这里forw_img表示当前帧，cur_img表示前一帧
    if (forw_img.empty())   
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear(); // 清空当前特征点信息，防止保留前一帧的特征点信息

    // 光流跟踪
    // 第一帧图像不进行光流跟踪
    // 当上一帧图像有特征点信息时，说名就可以进行光流跟踪了。
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // step1 通过调用opencv光流追踪给的状态位，剔除outlier
        // 使用图像金字塔进行图像光流跟踪，从图像金字塔的顶层数从开始作为初始值直到最底层；
        // status为状态位，前一帧到当前帧特征点是否跟踪成功，为0时表示光流跟踪失败
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        // 遍历当前帧的特征点
        for (int i = 0; i < int(forw_pts.size()); i++)
            // step2 通过图像边缘剔除outlier
            if (status[i] && !inBorder(forw_pts[i])) // 光流跟踪成功但不在图像内的点，将跟踪状态置为0
                status[i] = 0;
        reduceVector(prev_pts, status); // 没有用到
        // 根据状态位进行瘦身，即只保留状态位不为0的特征数据进行保留（使用双指针，完成将状态为零的特征信息去除，并替换为非零特征信息）
        reduceVector(cur_pts, status); // 前一帧特征点
        reduceVector(forw_pts, status); // 当前帧特征点
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status); // 成功追踪的次数的数组
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    // 将跟踪成功的特征点的追踪次数加1
    for (auto &n : track_cnt)
        n++;

    // 只有要发布该帧，才会进行如下操作，避免占用过多的计算资源
    if (PUB_THIS_FRAME)
    {
        // step3 通过对极约束去除外点outliner，得到去除外点后的数据
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();  // 通过设置区域，使特征点均衡化，
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        // 检测特征点
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // MAX_CNT tum数据中为150
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size()); // 需要新提取的特征点的数目
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 调用opencv的特征点提取函数。out = n_pts，提取角点的像素坐标;
            // 只有发布才会提取更多特征点，保证输送给后端的特征点数量
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints(); // 将新提取出的特征点添加到当前帧中
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    prev_img = cur_img; 
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts; // 以上三个值没有用到，因为在特征点提取和追踪过程中，cur_img表示上一帧，forw_img表示当前帧 forward

    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints(); // 当前帧所有点统一去畸变，同时计算特征点速度、用来后续时间戳标定
    prev_time = cur_time; // 更新上一帧的时间
}

// 使用对极几何去除光流跟踪的外点，对维护的特征点集进行瘦身
void FeatureTracker::rejectWithF()
{
    // 保证当前光流跟踪的点至少为8个
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        // 遍历特征点，得到去畸变后的归一化相机坐标，然后投影到虚拟相机，维护虚拟相机下前一帧和当前帧的归一化坐标
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // 得到相机归一化坐标系数据，去畸变
            // 出参：tmp_p（相机归一化坐标系下的坐标）
            // 入参：前一帧特征点像素坐标
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 这里使用一个虚拟相机；FOCAL_LENGTH表示虚拟焦距，已经被写死
            // 这里有个好处就是F_THRESHOLD和相机无关
            // 投影到虚拟相机的像素坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            // 得到当前帧去畸变后的归一化相机坐标系
            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // 通过opencv接口计算本质矩阵，某种意义上也是一种对极几何约束的outlier剔除（通过使用F_THRESHOLD阈值进行去除部分外点）
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

// 为新的特征点更新id值
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1) // 当前特征点为新提取的特征点
            ids[i] = n_id++; // 将所有特征点id进行排序
        return true;
    }
    else
        return false;
}

// 读取对应配置文件中的相机内参
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    // 从config_file文件中读取相机的内参，
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

// 当前帧所有特征点统一去畸变，同时计算特征点速度、用来后续时间戳标定
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    // cur_pts指的是当前帧，因为调用该函数之前，已经将cur_pts=forw_pts
    // 如果没有发布该关键帧，则需要将特征点进行去畸变；如果发布关键帧，则也可能会提取新的特征点，则需要进行去畸变
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        //之前所有的点已经做过去畸变了，但是因为有新帧特征点，故所有点全部进行去畸变处理
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y); // 像素坐标
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b); // 得到归一化相机坐标系下的去畸变的原始坐标b
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z())); // 将相机归一化的坐标的前两维添加到当前帧归一化容器中
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z()))); // 将特征点的id和归一化后的相机坐标添加进map中
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    // 计算特征点的速度
    // 计算特征点的速度，需要知道上一帧是否存在，因为需要上一帧和当前帧的距离才能求取特征点的速度
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        //  遍历当前帧的特征点
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1) // 在addpoint()函数中，将新提取特征点的id设为-1.新提取的特征点只有一个点无法计算速度。
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end()) // 在map中成功找到特征点id
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt; // x方向上的速度
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt; // y方向上的速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y)); // 将速度添加进速度容器中
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0)); // 当前特征点没有在上一帧中没有维护，将该特征点速度预设为0
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0)); // 将新提取的特征点速度设置为0
            }
        }
    }
    else // 如果该帧为第一帧，因为只有一个帧是没法计算特征点的速度的
    {
        // 如果是第一帧，将特征点的速度预设为0
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map; // 更新前一帧的特征点的map，用于与下一帧做比较
}
