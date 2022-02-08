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
*/

#include <nestedtensor/csrc/cuda/transformer_kernels.h>
#include <c10/util/Half.h>

namespace fastertransformer 
{


template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename T>
__global__ 
void add_bias_gelu(T* out, const T* __restrict bias, int m, int n)
{
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    T reg_bias = __ldg(&bias[id % n]);
    T val = out[id] + reg_bias;
    out[id] = (T)(gelu(val));
  }
}

template <typename T>
__global__ 
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] + __ldg(&bias[tid]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  out[blockIdx.x * n + tid] = 
        (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

template <typename T>
__global__ 
void add_bias_input_layernorm_v2(T* out, const T* __restrict input, const T* __restrict bias, 
                                const T* __restrict gamma, const T* __restrict beta, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float local_out[ite];

  float sum = 0.0f;
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid; 
    int id = bid * n + col_id; 
    local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  mean = blockReduceSum<float>(sum);
  if(tid == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    float diff = local_out[i] - s_mean;
    var += diff * diff;
  }

  variance = blockReduceSum<float>(var);
  if(tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid; 
    int id = bid * n + col_id; 
    out[id] = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
  }
}

template<typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input, const T* bias, 
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  if(n == 768 || n == 1024)
    add_bias_input_layernorm_v2<T><<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, n);
  else
    add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template <typename T>
__global__
void add_bias_input_layernorm_2(const T* __restrict input, 
                          const T* __restrict gamma, 
                          const T* __restrict beta, 
                          const T* __restrict bias, 
                          T* output, T* norm_output, 
                          int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f; 
  for(int i = tid; i < n; i+= blockDim.x)
  {
    float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
    local_out += (float)(output[blockIdx.x * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[blockIdx.x * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for(int i = tid; i < n; i+= blockDim.x)
  {
    float diff = (float)(__ldg(&output[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);
  
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  for(int i = tid; i < n; i+= blockDim.x)
  {
    norm_output[blockIdx.x * n + i] = 
      (T)((( (float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
  }
}

template<typename T>
void add_bias_input_layernorm_2_kernelLauncher(
  const T* input,
  const T* gamma,
  const T* beta,
  const T* bias,
  T* output,
  T* norm_output,
  int m, int n, 
  cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  
  if(n % 32 != 0)
    block.x = 1024;
  
  block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  add_bias_input_layernorm_2<T><<<grid, block, 0, stream>>>(input, gamma, beta, bias, output, norm_output, m, n); // For gpt-3 
}

template <typename T>
__global__ 
void add_bias_input(T* output, const T* input, const T* bias, const int m, const int n)
{
  // This kernel can run with any block size and grid size
  // Since the hidden dimension of GPT-3 would be larger than 1024
  const int bid = blockIdx.x;
  const int blocks_per_row = n / blockDim.x;
  const int col_index = (bid % blocks_per_row) * blockDim.x + threadIdx.x;
  T bias_val = __ldg(&bias[col_index]);
  for(int index = bid * blockDim.x + threadIdx.x; index < m * n; index += blockDim.x * gridDim.x)
  {
    output[index] = output[index] + input[index] + bias_val; 
  }
}

template<typename T>
void add_bias_input_kernelLauncher(T* output, const T* bias, const T* input, const int m, const int n, cudaStream_t stream)
{
  dim3 grid(min(m, 65536));
  dim3 block(min(n, 1024));
  
  add_bias_input<<<grid, block, 0, stream>>>(output, input, bias, m, n);
}

template <typename T>
__global__
void layer_norm_kernel_generalize(const T* __restrict input, 
                          const T* __restrict gamma, 
                          const T* __restrict beta, 
                          T eps,
                          T* output, 
                          int m, int n)
{
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f; 
  for(int i = tid; i < n; i+= blockDim.x)
  {
    local_sum += (float)(__ldg(&input[blockIdx.x * n + i]));
  }

  mean = blockReduceSum<float>(local_sum);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for(int i = tid; i < n; i+= blockDim.x)
  {
    float diff = (float)(__ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + eps);

  __syncthreads();

  for(int i = tid; i < n; i+= blockDim.x)
  {
    output[blockIdx.x * n + i] = 
      (T)((( (float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
  }
}

template<typename T>
void layer_norm(
  const T* input,
  const T* gamma,
  const T* beta,
  T eps,
  T* output,
  int m, int n,
  cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if(n % 32 != 0)
    block.x = 1024;

  block.x = block.x / (4 / sizeof(T)); // if using half, only need half of block.x
  // Note that this cannot be less than 32 because blockReduceSum above
  // uses (threadIdx.x < blockDim.x >> 5), which is true if blockDim.x is 16
  // which happens if n is 32 and we're using half.
  block.x = max(32, block.x);

  /* should pay attention to the rsqrt precision*/
  layer_norm_kernel_generalize<T><<<grid, block, 0, stream>>>(input, gamma, beta, eps, output, m, n); // For gpt-3
}

template void add_bias_input_layernorm_kernelLauncher<float>(
  float* out, const float* input, const float* bias, const float* gamma, const float* beta, 
  int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_2_kernelLauncher<float>(
  const float* input,
  const float* gamma,
  const float* beta,
  const float* bias,
  float* output,
  float* norm_output,
  int m, int n, cudaStream_t stream);

template void add_bias_input_kernelLauncher<float>(
  float* output,
  const float* bias,
  const float* input,
  const int m,
  const int n,
  cudaStream_t stream);

template void layer_norm<float>(
  const float* input,
  const float* gamma,
  const float* beta,
  float eps,
  float* output,
  int m, int n,
  cudaStream_t stream);

template void layer_norm<c10::Half>(
  const c10::Half* input,
  const c10::Half* gamma,
  const c10::Half* beta,
  c10::Half eps,
  c10::Half* output,
  int m, int n,
  cudaStream_t stream);

} // namespace fastertransformer
