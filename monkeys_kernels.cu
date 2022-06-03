/* Kernels for random number generation and target matching */
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// includes, project
#include "helper_cuda.h"
#include "config.h"
#include "structs.h"
#include "monkeys_kernels.cuh"

//----------------------------------------------------------------------------//
//! Initialize curand
//! @param  state  to be initialized by curand
//! @param  seed  change seed for curand
//! @param  arrLen  how many numbers are generated and queried
//----------------------------------------------------------------------------//
__global__ void kernInitRand(curandState * __restrict__ state,
    const unsigned int seed,
    const size_t arrLen)
{

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
  if (tid<arrLen) {
    if (seed) {
      curand_init(tid+(unsigned int)clock(), tid, 0, &state[tid]);
    }
    else {
      curand_init(seed, tid, 0, &state[tid]);
    }
  }
}

//----------------------------------------------------------------------------//
//! Fill an array with random numbers: use this to test rng distribution
//! @param  state  to be initialized by curand
//! @param  arrLen  how many numbers are generated and queried
//! @param  typewriterSize  number of characters in alphabet used
//! @param  d_testDistInts  array to contain random numbers
//----------------------------------------------------------------------------//
__global__ void kernGenOnly(curandState * __restrict__ state, 
    const size_t arrLen, 
    const size_t typewriterSize, 
    unsigned int * __restrict__ d_testDistInts)
{

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
  if (tid < arrLen) {

    // generate random uint
    float fRoll = curand_uniform(&state[tid])*typewriterSize;
    unsigned int uiRoll = (unsigned int)trunc(fRoll);
    __syncthreads();
    d_testDistInts[tid] = uiRoll;
  }
}

//----------------------------------------------------------------------------//
//! Randomly generate numbers and compare element-wise matches to target
//! @param  state  must be initialized by curand
//! @param  seed  change seed for curand
//! @param  arrLen  how many numbers are generated and queried
//! @param  typewriterSize  number of characters in alphabet used
//! @param  d_targetInt  integer representation of target string
//! @param  targetLen  length of target string
//! @param  d_charMatch  indicates character-wise matches to target
//----------------------------------------------------------------------------//
__global__ void kernGenComp(curandState * __restrict__ state, 
    const size_t arrLen, 
    const size_t typewriterSize, 
    const unsigned int*__restrict__ d_targetInt,
    const unsigned int targetLen,
    bool * __restrict__ d_charMatch)
{

  const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;

  // load target into shared memory
  volatile __shared__ unsigned int s_targetInt[SHMEM_SIZE+1];
  if ((threadIdx.x<targetLen)&&(tid<arrLen)) {
    s_targetInt[threadIdx.x] = d_targetInt[threadIdx.x];

    // duplicate target in shared mem to avoid bank conflicts later
    for(unsigned int i = threadIdx.x+BLOCK_SIZE_X; i<SHMEM_SIZE; 
        i += BLOCK_SIZE_X) {
      s_targetInt[i] = s_targetInt[threadIdx.x];
    }

  }
  __syncthreads();

  // generate random uint and compare to target
  bool match;
  if (tid<arrLen) {
    float fRoll = curand_uniform(&state[tid])*typewriterSize;
    unsigned int uiRoll = (unsigned int)trunc(fRoll);
    match = s_targetInt[threadIdx.x] == uiRoll;
    __syncthreads();
    d_charMatch[tid] = match;
  }
}

//----------------------------------------------------------------------------//
//! Vectorized memory access to check for contiguous matches: 4 char blocks
//! @param  arrLen  how many numbers are generated and queried
//! @param  targetLen  length of target string
//! @param  d_fullMatch  keeps track of when we find contiguous matches
//! @param  targetNum  consecutive "true" values represented as a number
//----------------------------------------------------------------------------//
#if TARGET_LENGTH == 4
  __global__ void kernVec4Match(const bool * __restrict__ d_charMatch,
      const size_t arrLen,
      const unsigned int targetLen,
      bool * __restrict__ d_fullMatch,
      const long targetNum)
  {

    const int tid = threadIdx.x+blockIdx.x*blockDim.x;

#if MATCH_VEC_SIZE == 4
    if (targetLen*4*tid<arrLen) {
      uint4 testNum;
      struct bool4 testBool;
      memcpy(&testNum, &d_charMatch[targetLen*4*tid], targetLen*4*sizeof(bool));
      testBool.x = testNum.x == targetNum;
      testBool.y = testNum.y == targetNum;
      testBool.z = testNum.z == targetNum;
      testBool.w = testNum.w == targetNum;
      __syncthreads();
      memcpy(&d_fullMatch[4*tid], &testBool, 4*sizeof(bool));
    }

#elif MATCH_VEC_SIZE == 2
    if (targetLen*2*tid<arrLen) {
      uint2 testNum;
      struct bool2 testBool;
      memcpy(&testNum, &d_charMatch[targetLen*2*tid], targetLen*2*sizeof(bool));
      testBool.x = testNum.x == targetNum;
      testBool.y = testNum.y == targetNum;
      __syncthreads();
      memcpy(&d_fullMatch[2*tid], &testBool, 2*sizeof(bool));
    }
#endif // #if MATCH_VEC_SIZE is 2 xor 4
  }

//----------------------------------------------------------------------------//
//! Vectorized memory access to check for contiguous matches: 8 char blocks
//! @param  arrLen  how many numbers are generated and queried
//! @param  targetLen  length of target string
//! @param  d_fullMatch  keeps track of when we find contiguous matches
//! @param  targetNum  consecutive "true" values represented as a number
//----------------------------------------------------------------------------//
#elif TARGET_LENGTH == 8
  __global__ void kernVec8Match(const bool * __restrict__ d_charMatch,
      const size_t arrLen,
      const unsigned int targetLen,
      bool * __restrict__ d_fullMatch,
      const long targetNum)
  {

    const int tid = threadIdx.x+blockIdx.x*blockDim.x;

#if MATCH_VEC_SIZE == 4
    if (targetLen*4*(tid+1)<arrLen) {
      ulong4 testNum;
      struct bool4 testBool;
      memcpy(&testNum, &d_charMatch[targetLen*4*tid], targetLen*4*sizeof(bool));
      testBool.x = testNum.x == targetNum;
      testBool.y = testNum.y == targetNum;
      testBool.z = testNum.z == targetNum;
      testBool.w = testNum.w == targetNum;
      __syncthreads();
      memcpy(&d_fullMatch[4*tid], &testBool, 4*sizeof(bool));
    }

#elif MATCH_VEC_SIZE == 2
    if (targetLen*2*(tid+1)<arrLen) {
      ulong2 testNum;
      struct bool2 testBool;
      memcpy(&testNum, &d_charMatch[targetLen*2*tid], targetLen*2*sizeof(bool));
      testBool.x = testNum.x == targetNum;
      testBool.y = testNum.y == targetNum;
      __syncthreads();
      memcpy(&d_fullMatch[2*tid], &testBool, 2*sizeof(bool));
    }
#endif // #if MATCH_VEC_SIZE is 2 xor 4
  }

#endif // #if TARGET_LENGTH is 4 xor 8
