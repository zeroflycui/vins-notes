#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix(); // z方向上的旋转
    double yaw = Utility::R2ypr(R0).x(); // 对齐到z轴后的yaw角变量
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // yaw角对齐到z轴，此时的yaw角偏移为0
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
