/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

Changes in comparison to original at commit 3bf1d43. Apply to both header and definitions.
 - Changed include path
 - Removed unneeded includes
 - Removed add_bias_act.* code
 - Removed code related to float16 / half
 - Added FINAL_MASK define
 - Added eps option to layer_norm

 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>

namespace fastertransformer
{

#define FINAL_MASK 0xffffffff

template <typename T>
void add_bias_input_layernorm_kernelLauncher(T *out, const T *input_tensor,
                                             const T *bias, const T *gamma,
                                             const T *beta, int m, int n,
                                             cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_2_kernelLauncher(const T *from_tensor, const T *gamma,
                                               const T *beta, const T *bias,
                                               T *output, T *norm_output_buf_,
                                               const int m, const int n, cudaStream_t stream);

template <typename T>
void add_bias_input_kernelLauncher(T *output, const T *bias, const T *input, const int m, const int n, cudaStream_t stream);

template <typename T>
void layer_norm(const T *from_tensor, const T *gamma,
                const T *beta, T eps, T *norm_from_tensor_buf_, const int m, const int n, cudaStream_t stream);

} // namespace fastertransformer
