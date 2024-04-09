/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3
computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs,
                   bool *clamped) {
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 pos = means[idx]; // 从给定的means数组中获取第idx个高斯的位置
    glm::vec3 dir = pos - campos; // 计算从相机位置(campos)指向当前高斯位置(pos)的方向向量
    dir = dir / glm::length(dir); // 将方向向量归一化，以确保它是单位长度的

    // 将球谐系数的指针shs强制转换为glm::vec3*类型，并且根据当前高斯的索引idx以及最大系数数量max_coeffs找到对应的球谐系数数组
    glm::vec3 *sh = ((glm::vec3 *) shs) + idx * max_coeffs;
    // 根据球谐系数数组的第一个系数（通常是基础系数），用权重SH_C0乘以它来计算初始的RGB颜色。这里假设SH_C0是一个常量，表示基础系数的权重。
    glm::vec3 result = SH_C0 * sh[0];

    // 这段代码是根据球谐系数的阶数对RGB颜色进行更多的修正，包括考虑到二阶和三阶系数的影响
    if (deg > 0) { // 检查球谐系数的阶数是否大于0，如果大于0，则表示有球谐系数需要考虑
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        /*
         * 根据一阶球谐系数修正RGB颜色。
         * 这里假设SH_C1是一个常量，表示一阶球谐系数的权重。
         * 球谐系数sh[1]、sh[2]和sh[3]分别表示一阶球谐系数的分量。
         * 这一行代码对RGB颜色进行了修正，以考虑一阶球谐系数对颜色的影响
         * */
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        //检查球谐系数的阶数是否大于1，如果大于1，则表示还需要考虑二阶球谐系数的影响
        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                     SH_C2[0] * xy * sh[4] +
                     SH_C2[1] * yz * sh[5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                     SH_C2[3] * xz * sh[7] +
                     SH_C2[4] * (xx - yy) * sh[8];

            //检查球谐系数的阶数是否大于2，如果大于2，则表示还需要考虑三阶球谐系数的影响
            if (deg > 2) {
                result = result +
                         SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                         SH_C3[1] * xy * z * sh[10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                         SH_C3[5] * z * (xx - yy) * sh[14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;// 这一行代码是为了给 RGB 颜色添加一个偏移量，以调整颜色的亮度

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    //这三行代码用于检查 RGB 颜色的各个分量是否小于 0，如果小于 0，则将对应的 clamped 数组中的值设置为 true，表示该颜色分量被截断了
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    /*
     * 最后，这一行代码确保 RGB 颜色的每个分量都不会小于 0，即将小于 0 的分量都设置为 0，以确保 RGB 颜色的值都为正值。
     * 这样做的目的是为了避免颜色出现负值，因为 RGB 颜色的值通常在 0 到 1 之间
     * */
    return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3
computeCov2D(const float3 &mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float *cov3D,
             const float *viewmatrix) {
    // The following models the steps outlined by equations 29
    // and 31 in "EWA Splatting" (Zwicker et al., 2002).
    // Additionally considers aspect / scaling of viewport.
    // Transposes used to account for row-/column-major conventions.
    /*
     * 这里调用了一个函数 transformPoint4x3，将点三维高斯中心点 mean 使用视图矩阵 viewmatrix 进行变换，得到了新的点 t。
     * 这个新的点 t 是在相机坐标系中的位置
     * */
    float3 t = transformPoint4x3(mean, viewmatrix);

    /*
     * 这两行代码计算了水平方向和垂直方向上的限制范围。
     * tan_fovx 和 tan_fovy 是水平和垂直方向上的视场角的正切值，通过它们可以确定屏幕空间中的范围
     * */
    const float limx = 1.3f * tan_fovx;// 计算水平方向上的限制范围
    const float limy = 1.3f * tan_fovy;// 计算垂直方向上的限制范围
    /*
     * 这两行代码计算了点 t 在屏幕空间中的坐标，具体来说是计算了点 t 在视平面上的投影坐标
     * */
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    /*
     * 这两行代码将点 t 的水平和垂直位置限制在屏幕空间内的范围内。
     * 通过取最大值和最小值来确保点在给定范围内，然后乘以 t.z 以将点带回相机坐标系的深度
     * */
    t.x = min(limx, max(-limx, txtz)) * t.z;// 对水平位置进行限制
    t.y = min(limy, max(-limy, tytz)) * t.z;// 对垂直位置进行限制


    /*
     * 这里计算了焦距矩阵 J，焦距矩阵是一个 3x3 的矩阵，用于将相机坐标系中的点投影到屏幕空间。
     * 其中 focal_x 和 focal_y 是 x 和 y 方向上的焦距，t.z 是点在相机坐标系中的深度。
     * 这里的计算基于投影方程，将焦距和点的深度考虑在内
     * */
    glm::mat3 J = glm::mat3(
            focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
            0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0, 0, 0);

    /*
     * 这里构建了一个 3x3 的视图矩阵 W。视图矩阵用于将世界坐标系中的点变换到相机坐标系中。
     * 视图矩阵是由 viewmatrix 提供的，它是一个 4x4 的矩阵，通常表示相机的位置和朝向
     * */
    glm::mat3 W = glm::mat3(
            viewmatrix[0], viewmatrix[4], viewmatrix[8],
            viewmatrix[1], viewmatrix[5], viewmatrix[9],
            viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    /*
     * 这行代码计算了一个转换矩阵 T，通过将焦距矩阵 J 应用于视图矩阵 W 得到。
     * 这个转换矩阵 T 结合了视图变换和投影变换，用于将世界坐标系中的点投影到屏幕空间
     * */
    glm::mat3 T = W * J;
    /*
     * 这里构建了一个 3x3 的三维协方差矩阵 Vrk。
     * 这个协方差矩阵是根据输入的 cov3D 构建的，cov3D 包含了三维空间中点的协方差信息
     * */
    glm::mat3 Vrk = glm::mat3(
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]);

    /*
     * 这一行代码计算了二维协方差矩阵 cov，即将三维协方差矩阵转换为了二维屏幕空间的协方差矩阵
     * 首先，将三维协方差矩阵 Vrk 进行转置并左乘转置后的视图变换矩阵 T，然后再次对结果进行转置并右乘原始的视图变换矩阵 T。
     * */
    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    /*
     * 这两行代码对二维协方差矩阵的对角元素进行了修正，增加了一个小的值（0.3f）。
     * 这个操作相当于对协方差矩阵施加了一个低通滤波，确保每个高斯函数至少有一个像素的宽度和高度
     * */
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    //最后，函数返回了修正后的二维协方差矩阵的对角元素
    return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float *cov3D) {
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);// 首先，创建了一个3x3的单位矩阵 S，表示初始时没有进行任何缩放操作
    /*
     * 然后，将缩放向量 scale 的每个分量分别乘以修正因子 mod，并将结果分别赋值给缩放矩阵 S 的对角线元素。
     * 这个操作会在每个轴上对高斯函数进行相应比例的缩放，从而调整高斯函数的形状
     * */
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    /*
     * 将缩放矩阵 S 与旋转矩阵 R 相乘，得到最终的变换矩阵 M。
     * 这个变换矩阵将旋转和缩放操作组合在一起，用于在后续步骤中构建三维协方差矩阵
     * */
    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    /*
     * 首先，计算了变换矩阵 M 的转置与其自身的乘积，得到了三维世界协方差矩阵 Sigma。
     * 这一步是基于线性代数中的性质，用于将变换矩阵转换为对应的协方差矩阵
     * */
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    /*
     * 然后，将协方差矩阵 Sigma 的上三角元素分别存储到数组 cov3D 中。
     * 由于协方差矩阵是对称矩阵，所以只需存储上三角部分即可
     * */
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,//高斯数量
                               int D,//球谐函数的最高阶数
                               int M,//球谐函数的最大系数数量
                               const float *orig_points,//原始点的坐标数组
                               const glm::vec3 *scales,//每个高斯的尺度向量数组
                               const float scale_modifier,//尺度修正因子
                               const glm::vec4 *rotations,//每个高斯的旋转四元数数组
                               const float *opacities,//每个高斯的不透明度数组
                               const float *shs,//每个高斯的球谐系数数组
                               bool *clamped,//用于记录RGB颜色是否被截断的布尔数组
                               const float *cov3D_precomp,//预计算的每个高斯的三维协方差矩阵数组
                               const float *colors_precomp,//预计算的每个高斯的颜色数组
                               const float *viewmatrix,//视图矩阵
                               const float *projmatrix,//投影矩阵
                               const glm::vec3 *cam_pos,//摄像机位置
                               const int W, int H,//输出图像的宽度和高度
                               const float tan_fovx, float tan_fovy,//水平和垂直方向上的视场角的正切值
                               const float focal_x, float focal_y,//焦距
                               int *radii,//每个高斯的半径数组
                               float2 *points_xy_image,//每个高斯在图像上的坐标数组
                               float *depths,//每个高斯的深度数组
                               float *cov3Ds,//每个高斯的三维协方差矩阵数组
                               float *rgb,//RGB颜色数组
                               float4 *conic_opacity,//用于光栅化的椎体和不透明度数组
                               const dim3 grid,//二维线程块数量
                               uint32_t *tiles_touched,//记录每个高斯覆盖的图像块数量的数组
                               bool prefiltered//指示是否对输入进行了预过滤的布尔值
                               ) {
    /*
     * 这部分代码使用了CUDA Thrust库中的cg::this_grid().thread_rank()函数来获取当前线程在网格中的排名（即线程在二维网格中的索引）。
     * 然后，它检查当前线程的排名是否超出了高斯的数量P。如果超出了，则直接返回，不执行任何操作。
     * 这样做是为了确保只有分配到高斯的线程才会执行后续的操作，避免不必要的计算和内存访问
     * */
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    /*
     * 初始化变量 radii 和 tiles_touched，将它们的值都设为0。
     * 这些变量用于存储高斯的半径和接触的瓦片数
     * */
    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    /*
     * 调用 in_frustum 函数，对当前高斯进行近裁剪操作。如果高斯在视锥体之外，则直接返回，不再处理该高斯
     * */
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    //首先，从 orig_points 数组中获取当前点的坐标，存储在 p_orig 变量中
    float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
    //然后，通过使用 projmatrix 对点进行投影变换，得到齐次坐标系下的结果，存储在 p_hom 变量中
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    //接着，计算齐次坐标系下的 w 分量的倒数，存储在 p_w 变量中。这一步是为了进行透视除法
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    //最后，通过将齐次坐标系下的结果除以 w 分量，得到投影后的点的坐标，存储在 p_proj 变量中
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

    // If 3D covariance matrix is precomputed, use it, otherwise compute
    // from scaling and rotation parameters.
    const float *cov3D;
    if (cov3D_precomp != nullptr) {
        //如果预先计算了3D协方差矩阵，则将 cov3D 指向预先计算的矩阵，其中 idx * 6 用于定位当前高斯的矩阵数据
        cov3D = cov3D_precomp + idx * 6;
    } else {// 如果没有预先计算的矩阵，则需要根据尺度和旋转参数计算3D协方差矩阵
        /*
         * 调用 computeCov3D 函数计算3D协方差矩阵，并将结果存储在 cov3Ds 数组中，同样使用 idx * 6 来定位当前高斯的矩阵数据
         * */
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        //将 cov3D 指向刚刚计算的3D协方差矩阵
        cov3D = cov3Ds + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    //计算2D屏幕空间的协方差矩阵，并将结果存储在 cov 变量中
    float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);// 计算协方差矩阵的行列式
    if (det == 0.0f)
        return;// 如果行列式为0，说明协方差矩阵不可逆，直接返回
    float det_inv = 1.f / det;// 计算行列式的倒数，用于后续计算
    // 使用行列式的倒数将协方差矩阵转换为一个二次曲面(conic)，这是EWA算法中使用的一种表示方法
    float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute a bounding rectangle
    // of screen-space tiles that this Gaussian overlaps with. Quit if
    // rectangle covers 0 tiles.
    float mid = 0.5f * (cov.x + cov.z);// 计算二维协方差矩阵的中心点的平均值
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));// 计算第一个特征值
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));// 计算第二个特征值
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));// 计算覆盖当前高斯分布的屏幕空间的半径
    float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};// 将当前点转换为屏幕空间坐标
    uint2 rect_min, rect_max;
    // 根据半径和屏幕空间坐标计算覆盖当前高斯分布的屏幕空间矩形的最小和最大坐标
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return; // 如果矩形的面积为0，说明该高斯分布不覆盖任何屏幕空间的矩形，直接返回

    // If colors have been precomputed, use them, otherwise convert
    // spherical harmonics coefficients to RGB color.
    if (colors_precomp == nullptr) {
        // 如果颜色预先计算了，则使用它们，否则将球谐系数转换为RGB颜色
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *) orig_points, *cam_pos, shs, clamped);
        // 将颜色存储到RGB数组中
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Store some useful helper data for the next steps.
    depths[idx] = p_view.z;// 存储视图坐标中的z分量
    radii[idx] = my_radius;// 存储半径
    points_xy_image[idx] = point_image;// 存储屏幕空间坐标
    // Inverse 2D covariance and opacity neatly pack into one float4
    conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};// 存储逆2D协方差和不透明度
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);// 存储受影响的瓷砖数
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 声明了一个名为renderCUDA的CUDA核函数，具有模板参数CHANNELS，代表输出的颜色通道数
template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
        const uint2 *__restrict__ ranges,// 表示每个线程块要处理的点的范围
        const uint32_t *__restrict__ point_list,// 表示点的列表
        int W, int H,// 图像的宽度和高度
        const float2 *__restrict__ points_xy_image,// 表示点在图像上的位置坐标
        const float *__restrict__ features,// 表示点的特征
        const float4 *__restrict__ conic_opacity,// 表示点的锥形和不透明度
        float *__restrict__ final_T,// 最终的T值
        uint32_t *__restrict__ n_contrib, // 贡献数
        const float *__restrict__ bg_color,// 背景颜色
        float *__restrict__ out_color// 输出的颜色
        ) {
    // Identify current tile and associated min/max pixel range.
    // 标识当前的瓦片和相关的最小/最大像素范围
    auto block = cg::this_thread_block();
    // 获取水平方向上的线程块数量
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    // 计算当前线程块的最小像素坐标
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    // 计算当前线程块的最大像素坐标，不能超出图像的宽度和高度
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    // 计算当前线程处理的像素坐标
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    // 计算当前像素在图像中的索引
    uint32_t pix_id = W * pix.y + pix.x;
    // 将像素坐标转换为浮点数格式
    float2 pixf = {(float) pix.x, (float) pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);// 计算处理范围需要的轮数
    int toDo = range.y - range.x;// 计算要处理的总数量

    // Allocate storage for batches of collectively fetched data.
    // 为批量收集的数据分配存储空间
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Initialize helper variables
    float T = 1.0f;// 初始化累积的透射率为1
    uint32_t contributor = 0;// 初始化贡献者数量为0
    uint32_t last_contributor = 0;// 初始化上一个贡献者数量为0
    float C[CHANNELS] = {0};// 初始化累积的颜色为0向量

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        // 如果整个块都标记为完成渲染，则结束循环
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        // 从全局内存收集每个高斯数据到共享内存
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];// 获取当前批次的高斯ID
            collected_id[block.thread_rank()] = coll_id;// 将ID存储到共享内存
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];// 存储高斯的坐标到共享内存
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // 存储高斯的圆锥和不透明度到共享内存
        }
        block.sync(); // 等待所有线程完成数据收集

        // Iterate over current batch
        // 在当前批次上迭代
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current position in range
            // 跟踪范围中的当前位置
            contributor++;

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            // 使用圆锥矩阵重新采样
            float2 xy = collected_xy[j];// 获取高斯的坐标
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};// 计算高斯点相对于像素的偏移
            float4 con_o = collected_conic_opacity[j];// 获取高斯的圆锥和不透明度
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)// 如果功率为正，跳过此高斯
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix).
            // 通过与高斯不透明度和距均值的指数衰减相乘来获得α。
            // 避免数值不稳定性（请参阅论文附录）。
            float alpha = min(0.99f, con_o.w * exp(power));// 计算像素处的alpha值
            if (alpha < 1.0f / 255.0f)// 如果alpha小于最小值，跳过此高斯
                continue;
            float test_T = T * (1 - alpha);// 更新T值
            if (test_T < 0.0001f) {// 如果T小于阈值，标记为已完成
                done = true;
                continue;
            }

            // Eq. (3) from 3D Gaussian splatting paper.
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;// 计算像素处的颜色值

            T = test_T;// 更新T值

            // Keep track of last range entry to update this pixel.
            // 跟踪最后一个范围条目以更新此像素。
            last_contributor = contributor;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside) {
        final_T[pix_id] = T;//将变量T（表示透明度）写入到final_T数组中的对应像素位置pix_id。这个变量用于记录像素的最终透明度
        n_contrib[pix_id] = last_contributor;//将变量last_contributor（表示最后一个对当前像素有贡献的高斯数量）写入到n_contrib数组中的对应像素位置pix_id
        /*
         * 对于每个颜色通道（ch循环），计算最终的像素颜色并将其写入到out_color数组中的对应位置。
         * 最终颜色是通过将像素在每个颜色通道上的贡献与背景颜色相加得到的，并且还考虑了像素的透明度T。
         * */
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    }
}

void FORWARD::render(
        const dim3 grid, dim3 block,
        const uint2 *ranges,
        const uint32_t *point_list,
        int W, int H,
        const float2 *means2D,
        const float *colors,
        const float4 *conic_opacity,
        float *final_T,
        uint32_t *n_contrib,
        const float *bg_color,
        float *out_color) {
    renderCUDA<NUM_CHANNELS> << < grid, block >> > (
            ranges,
                    point_list,
                    W, H,
                    means2D,
                    colors,
                    conic_opacity,
                    final_T,
                    n_contrib,
                    bg_color,
                    out_color);
}

void FORWARD::preprocess(int P, int D, int M,
                         const float *means3D,
                         const glm::vec3 *scales,
                         const float scale_modifier,
                         const glm::vec4 *rotations,
                         const float *opacities,
                         const float *shs,
                         bool *clamped,
                         const float *cov3D_precomp,
                         const float *colors_precomp,
                         const float *viewmatrix,
                         const float *projmatrix,
                         const glm::vec3 *cam_pos,
                         const int W, int H,
                         const float focal_x, float focal_y,
                         const float tan_fovx, float tan_fovy,
                         int *radii,
                         float2 *means2D,
                         float *depths,
                         float *cov3Ds,
                         float *rgb,
                         float4 *conic_opacity,
                         const dim3 grid,
                         uint32_t *tiles_touched,
                         bool prefiltered) {
    preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
            P, D, M,
                    means3D,
                    scales,
                    scale_modifier,
                    rotations,
                    opacities,
                    shs,
                    clamped,
                    cov3D_precomp,
                    colors_precomp,
                    viewmatrix,
                    projmatrix,
                    cam_pos,
                    W, H,
                    tan_fovx, tan_fovy,
                    focal_x, focal_y,
                    radii,
                    means2D,
                    depths,
                    cov3Ds,
                    rgb,
                    conic_opacity,
                    grid,
                    tiles_touched,
                    prefiltered
    );
}