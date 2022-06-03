/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/* Reduction kernels used determine if any contiguous matches were generated */
// The functions here are based on reduce7 from
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu

#ifndef _REDUCTION_KERNELS_CUH_
#define _REDUCTION_KERNELS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"

//----------------------------------------------------------------------------//
//! Sum reduction within a warp
//! @param  s_fullMatch  any nonzero value means we found a match
//! @param  tid  thread id within warp
//----------------------------------------------------------------------------//
template <const unsigned int blockSize>
__device__ void warpVecAnyMatch(volatile unsigned int * s_fullMatch, 
    unsigned int tid) 
{

  if (blockSize>=64) s_fullMatch[tid] += s_fullMatch[tid+32];
  if (blockSize>=32) s_fullMatch[tid] += s_fullMatch[tid+16];
  if (blockSize>=16) s_fullMatch[tid] += s_fullMatch[tid+8 ];
  if (blockSize>=8 ) s_fullMatch[tid] += s_fullMatch[tid+4 ];
  if (blockSize>=4 ) s_fullMatch[tid] += s_fullMatch[tid+2 ];
  if (blockSize>=2 ) s_fullMatch[tid] += s_fullMatch[tid+1 ];

}

//----------------------------------------------------------------------------//
//! Sum reduction on an array
//! @param  g_idata  input data: any nonzero value means we found a match
//! @param  g_odata  will alternate between reading from d_Amatch and writing to
//!                  d_Bmatch, and vice versa
//! @param  n  how many elements to be read from input array
//! @param  writeRes  whether or not to write to u_foundMatch
//! @param  u_foundMatch  if true, we found a nonzero value
//----------------------------------------------------------------------------//
template <const unsigned int blockSize>
__global__ void kernVecAnyMatch(const bool * __restrict__ g_idata,
    bool * g_odata, 
    const unsigned int n,
    const bool writeRes,
    bool * __restrict__ u_foundMatch)
{

  const int global_tid = threadIdx.x+blockIdx.x*blockDim.x;
  const int tid = threadIdx.x;
  volatile __shared__ unsigned int s_fullMatch[SHMEM_SIZE+1];

#if REDUCTION_VEC_SIZE == 16
    if (16*global_tid<n) {
      uint4 testNum;
      memcpy(&testNum, &g_idata[16*global_tid], sizeof(bool)*16);
      s_fullMatch[tid] = testNum.x+testNum.y+testNum.z+testNum.w;
    } else {s_fullMatch[tid] = 0;}

#elif REDUCTION_VEC_SIZE == 8
    if (8*global_tid<n) {
      uint2 testNum;
      memcpy(&testNum, &g_idata[8*global_tid], sizeof(bool)*8);
      s_fullMatch[tid] = testNum.x+testNum.y;
    } else {s_fullMatch[tid] = 0;}

#elif REDUCTION_VEC_SIZE == 4
    if (4*global_tid<n) {
      unsigned int testNum;
      memcpy(&testNum, &g_idata[4*global_tid], sizeof(bool)*4);
      s_fullMatch[tid] = testNum;
    } else {s_fullMatch[tid] = 0;}

#endif // REDUCTION_VEC_SIZE must be in {4,8,16}

  __syncthreads();

  if (blockSize>=512) { if (tid<256) {s_fullMatch[tid] +=
    s_fullMatch[tid+256]; } __syncthreads(); }
  if (blockSize>=256) { if (tid<128) {s_fullMatch[tid] +=
    s_fullMatch[tid+128]; } __syncthreads(); }
  if (blockSize>=128) { if (tid<64 ) {s_fullMatch[tid] +=
    s_fullMatch[tid+64 ]; } __syncthreads(); }

  if (tid<WARP_SIZE) warpVecAnyMatch<blockSize>(s_fullMatch, tid);
  if (tid == 0) {
    g_odata[blockIdx.x] = s_fullMatch[0] != 0;
    if (writeRes) u_foundMatch[0] = s_fullMatch[0] != 0;
  }

}

//----------------------------------------------------------------------------//
//! Call reduction kernel and write result if needed
//! @param  nBlocks  how many blocks will be launched
//! @param  n  how many elements to be read from input array
//! @param  g_idata  input data: any nonzero value means we found a match
//! @param  g_odata  will alternate between reading from d_Amatch and writing
//!   to d_Bmatch, and vice versa
//! @param  sharedBytes  amount of shared memory per block
//! @param  stream  the stream in which the kernel will be launched
//! @param  u_foundMatch  if true, we found a nonzero value
//----------------------------------------------------------------------------//
__host__ void callVecAnyMatch(const unsigned int nBlocks,
    const size_t n,
    const bool * __restrict__ g_idata,
    bool * __restrict__ g_odata,
    const size_t sharedBytes,
    const cudaStream_t stream,
    bool * __restrict__ u_foundMatch)
{

  // write to unified memory on last literation
  bool writeRes = false;
  if (n<=SHMEM_SIZE*REDUCTION_VEC_SIZE) writeRes = true;
  switch (SHMEM_SIZE) {

    case 512: 
      kernVecAnyMatch<512><<< nBlocks, SHMEM_SIZE, sharedBytes, stream>>>
        (g_idata, g_odata, n, writeRes, u_foundMatch); break;
    case 256:
      kernVecAnyMatch<256><<< nBlocks, SHMEM_SIZE, sharedBytes, stream>>>
        (g_idata, g_odata, n, writeRes, u_foundMatch); break;
    case 128:
      kernVecAnyMatch<128><<< nBlocks, SHMEM_SIZE, sharedBytes, stream>>>
        (g_idata, g_odata, n, writeRes, u_foundMatch); break;
    case 64:
      kernVecAnyMatch<64 ><<< nBlocks, SHMEM_SIZE, sharedBytes, stream>>>
        (g_idata, g_odata, n, writeRes, u_foundMatch); break;
    case 32:
      kernVecAnyMatch<32 ><<< nBlocks, SHMEM_SIZE, sharedBytes, stream>>>
        (g_idata, g_odata, n, writeRes, u_foundMatch); break;
    case 16:
      kernVecAnyMatch<16 ><<< nBlocks, SHMEM_SIZE, sharedBytes, stream>>>
        (g_idata, g_odata, n, writeRes, u_foundMatch); break;

  }
}

#endif  // #ifndef _REDUCTION_KERNELS_CUH_
