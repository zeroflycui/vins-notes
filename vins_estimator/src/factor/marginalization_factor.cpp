#include "marginalization_factor.h"

// 计算待边缘化的各个残差块的残差和雅克比，同时处理核函数的case
void ResidualBlockInfo::Evaluate()
{
    // 残差是一个动态向量
    residuals.resize(cost_function->num_residuals()); // 确定残差块的维数

    std::vector<int> block_sizes = cost_function->parameter_block_sizes(); // 确定相关参数块的数目
    raw_jacobians = new double *[block_sizes.size()]; // ceres接口都是double数组，因此这里给雅克比准备double数组，创建了block_size()个double数组
    jacobians.resize(block_sizes.size()); // 
    // 这里实际上把jacobians中的每一个matrix地址赋给raw_jacobians，然后把raw_jacobians传递给ceres的接口，这样计算结果直接放进了这个matrix
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]); // 雅克比矩阵维度大小 残差×变量
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    // 调用各自重载的接口计算残差和雅克比
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians); // 这里实际上结果放在了jacobians

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);
    // 如果是视觉重投影会有核函数，目的是为了消除部分外点，那么就对残差进行相关调整
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm(); // 获得残差的模
        loss_function->Evaluate(sq_norm, rho); // 得到rho的值。rho[0]是核函数在这个点的值，rho[1]是这个点的导数，rho[2]是这个点的二阶导
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]); // 残差模的一阶导数的平方根

        // 如果二阶导数小于零则认为是outlier region
        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) // 柯西核 p = log(s+1)的二阶导 <= 0始终成立，一般函数二阶导都是小于0的
        {
            residual_scaling_ = sqrt_rho1_; // 更新残差的尺度
            alpha_sq_norm_ = 0.0; // 更新残差模的变化量
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }
        // 这里相当于残差雅克比都乘上sprt_rho1_，及核函数所在的点的一阶导数，基本都是小于1
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_; // 为每一个残差都乘上残差尺度
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

// brief: 将生成的残差块信息添加到残差块中，记录每个参数块的首地址以及对应参数块的大小，待边缘化的参数块首地址对应的大小置为0
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info); // 收集各个残差块

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks; // 这个是和该约束相关的参数块
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes(); // 各个参数块的大小

    // 遍历每一个参数快，得到每个参数块对应的地址以及参数块大小
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];// 参数快的首地址
        int size = parameter_block_sizes[i]; // 参数块的大小
        // 这里是个unorded map（哈希表），避免重复添加
        parameter_block_size[reinterpret_cast<long>(addr)] = size; // 地址->global size
    }
    // 待边缘化的参数块，将待边缘化参数块首地址对应参数块大小设置为0
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        // 先准备好待边缘化的参数块的unordered_map
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

// brief : 将各个残差块计算残差和雅克比，同时备份所有相关的参数块内容
void MarginalizationInfo::preMarginalize()
{
    // 遍历各个残差块
    for (auto it : factors)
    {
        it->Evaluate(); // 调用这个接口，通过自定义的方法计算残差块的残差和雅克比

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes(); // 得到每个残差块的参数块大小
        // 遍历残差块的参数块，对没有存储的参数块进行备份
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]); // 得到参数块的首地址
            int size = block_sizes[i]; // 参数块的大小
            // 如果在unordered_map中没有参数块的首地址，则将参数块的地址备份到unordered_map中
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                // 深拷贝
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size); // 将参数块备份到data中
                parameter_block_data[addr] = data; // 存储到unordred_map中
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

// 子线程构造 Ax = b
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    // 遍历分配的子任务，构造每个子任务的H矩阵和g矩阵
    for (auto it : p->sub_factors)
    {
        // 遍历参数块，构造H矩阵和g矩阵
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])]; // 在大矩阵中的id，
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            // 确保是local size
            if (size_i == 7)
                size_i = 6;
            // 之前边缘化预处理已经算好了各个残差和雅克比，这里取出来
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            // 遍历参数块
            // 和本身以及其他雅克比块构造H矩阵
            // i：当前参数块 j：另一个参数块
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])]; // 在大矩阵中的id
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                // 对角线上的参数块为J^TJ
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else // 非对角线上的参数块关于对角线互为转置
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            // 构造误差矩阵g= -J^Te, e是残差
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

// 边缘化操作， 并将结果转化为残差和雅克比的形式
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    // 创建参数块的索引
    // parameter_block_idx的key值是待边缘化的参数块的地址，value值都预设为0
    for (auto &it : parameter_block_idx)
    {
        it.second = pos; // 这就是在所有参数块中排序的idx，待边缘化的排在前面
        // 小参数块的起始索引（参数块的维度是残差维度×变量的维度，残差中旋转是三维，而变量中的旋转是用李代数表示）
        pos += localSize(parameter_block_size[it.first]);  // 每次偏移一个参数块大小
    }

    m = pos; // 总共待边缘化的参数块总大小（不是个数）
    // 遍历参数块，在待边缘化参数块unordered_map中添加不存在的参数块大小
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }
    // pos指的是H矩阵参数块的总大小
    n = pos - m; // 新增的待边缘化的参数块大小

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    // 构造增量方程
    Eigen::MatrixXd A(pos, pos); // Ax = b
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread

    // 往A矩阵和b矩阵中添加东西，利用多线程加速
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it); // 每个子线程均匀分配任务
        i++;
        i = i % NUM_THREADS; // i以NUM_THREADS为周期进行累加赋值
    }
    // 每个线程构造一个A矩阵和b矩阵，最后大家加起来
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        // A矩阵和b矩阵大小都一样，预设均为0
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        // 多线程访问会带来冲突，因此每个线程备份以下要查询的map
        threadsstruct[i].parameter_block_size = parameter_block_size; // 参数块大小
        threadsstruct[i].parameter_block_idx = parameter_block_idx; // 待边缘化参数块索引
        // 产生若干线程，分别计算各自的H矩阵和g矩阵
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        pthread_join( tids[i], NULL );  // 等待各线程完成各自的任务
        // 把各个子模块拼起来，就是最终的Hx = g的矩阵了
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    
    /*
        | Amm Amr | 
    A = | Arm Arr |  Amm 表示待边缘化的变量

        | bmm |
    b = | brr |  bmm表示待边缘化的变量
    */
    // 进行舒尔补操作，（边缘化地图点和第零帧，待确定待边缘化的内容？？？？）
    //TODO
    // Amm矩阵的构造是为了保证其正定性，保证Amm求逆有意义
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose()); 
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm); // 对Amm矩阵进行特征值分解

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    // 一个逆矩阵的特征值是原矩阵特征值的倒数，特征向量相同 select 类似c++中 ？ ： 运算符。（a > b).select(10, 0)
    // 利用特征值取逆来构造其逆矩阵
    // eps表示无穷小
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    Eigen::VectorXd bmm = b.segment(0, m); // 待边缘化的大小
    // 对应的四块矩阵
    Eigen::MatrixXd Amr = A.block(0, m, m, n); 
    Eigen::MatrixXd Arm = A.block(m, 0, n, m); // 
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n); // 剩下的参数

    A = Arr - Arm * Amm_inv * Amr; // 边缘化后的A矩阵
    b = brr - Arm * Amm_inv * bmm; // 边缘化后的b矩阵

    // 求出雅克比J和误差e
    // 这个地方根据Ax = b => J^T*J = -J^T*e
    // 对A做特征值分解 A = V * S * V^T，其中S是特征值构成的对角矩阵。 这里把S = S^(1/2)^T * S^(1/2)，为了方便分解
    // 因为A = J^T * J, A = V * S * V^T  ==> J = S^(1/2) * V^T
    // 因b = -J^T * e, 故,e = -(J^T)^-1 * b = -((S^(1/2)*V^T)^-1) * b
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    // 对A矩阵取逆
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt(); // 这个求得的是S^(1/2)，不过这里是向量不是矩阵
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose(); // J = S^(1/2) * V^T
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b; // e = ((S^(1/2)*V^T)^-1) * b, 这里没有考虑负号
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}
 // 更新边缘化后参数块的size、idx、data、addr。返回边缘化参数块的地址数组
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx) // 遍历边缘化相关的参数块。维护边缘化剩下的参数块大小、id、数值以及地址
    {
        if (it.second >= m) // 如果是留下来的，说明后续会对其形成约束。这里的m指的是边缘化参数块Amm的维度
        {
            keep_block_size.push_back(parameter_block_size[it.first]); // 留下来的参数块大小 global size
            keep_block_idx.push_back(parameter_block_idx[it.first]); // 留下来的在原向量中排序
            keep_block_data.push_back(parameter_block_data[it.first]); // 边缘化前各个参数块的值的备份
            keep_block_addr.push_back(addr_shift[it.first]); // 滑窗内移动后对应的新地址
        }
    }
    // 剩下所有参数块的大小
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
     // 遍历边缘化后剩余参数块
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it); // 调用ceres接口，添加参数块大小信息
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n); // 残差维数就是所有剩余状态量的维数和，这里是local size
};

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n; // 上一次边缘化保留的残差块的local size的和，也就是残差维数
    int m = marginalization_info->m; // 上次边缘化中被待边缘化的残差维数
    Eigen::VectorXd dx(n); // 用来存储残差
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m; // 保留参数的索引，因为keep_block_idx中保存的idx值是含有被边缘化掉参数块时的排序数组，这里将idx从0开始
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size); // parameters是当前滑窗中的参数。这里得到的是滑窗中参数块的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size); // 这里是当时被边缘化参数块的值
        if (size != 7) // 说名当前参数块不是位姿（位姿的维数是7）
            dx.segment(idx, size) = x - x0; // 获得当前参数块和边缘化后保留参数块维数的差值
        else // 当前参数块是位姿
        {
            // 平移可以直接作差
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            // 旋转作差，需要转为四元数操作，不能直接做广义上的减法
            // 取四元数的虚部
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            
            // 确保四元数实部大于零
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }

    // 更新残差 边缘后的先验误差 e = e0 + J * dx
    // VINS根据FEJ认为雅克比是不变的，只是通过雅克比更新残差值
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {
        
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m; // 边缘化完成之后参数块完成归一化操作
                // 按行存储
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size); // middleCols 从idx处开始去，取local_size个
            }
        }
    }
    return true;
}
