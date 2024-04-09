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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer {
/*
 * 用于从内存块中获取特定类型的数据指针，并更新内存块的指针位置
 * @param chunk: 指向内存块的指针，通过引用传递，表示内存块的起始位置
 * @param ptr:通过引用传递，表示要获取的特定类型 T 的数据指针的目标地址
 * @param count:表示要获取的数据块的数量
 * @param alignment:表示数据的对齐要求，通常为字节对齐的值
 * */
    template<typename T>
    static void obtain(char *&chunk, T *&ptr, std::size_t count, std::size_t alignment) {
        // 1.计算偏移量，使得指针满足对齐要求
        std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        //2. 将偏移后的指针转换为 T 类型指针并赋值给 ptr
        ptr = reinterpret_cast<T *>(offset);
        //3. 更新 chunk 指针，指向下一个数据块的起始位置
        chunk = reinterpret_cast<char *>(ptr + count);
    }

    struct GeometryState {
        size_t scan_size;
        float *depths;
        char *scanning_space;
        bool *clamped;
        int *internal_radii;
        float2 *means2D;
        float *cov3D;
        float4 *conic_opacity;
        float *rgb;
        uint32_t *point_offsets;
        uint32_t *tiles_touched;

        static GeometryState fromChunk(char *&chunk, size_t P);
    };

    struct ImageState {
        uint2 *ranges;
        uint32_t *n_contrib;
        float *accum_alpha;

        static ImageState fromChunk(char *&chunk, size_t N);
    };

    struct BinningState {
        size_t sorting_size;
        uint64_t *point_list_keys_unsorted;
        uint64_t *point_list_keys;
        uint32_t *point_list_unsorted;
        uint32_t *point_list;
        char *list_sorting_space;

        static BinningState fromChunk(char *&chunk, size_t P);
    };

    template<typename T>
    size_t required(size_t P) {
        char *size = nullptr;
        T::fromChunk(size, P);
        return ((size_t) size) + 128;
    }
};