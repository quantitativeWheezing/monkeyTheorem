/* Kernels for random number generation and target matching */
#ifndef _MONKEYS_KERNELS_CUH_
#define _MONKEYS_KERNELS_CUH_

// includes, project
#include "config.h"

//----------------------------------------------------------------------------//
//! Initialize curand
//----------------------------------------------------------------------------//
__global__ void kernInitRand(curandState * __restrict__ state,
    const unsigned int seed,
    const size_t arrLen);

//----------------------------------------------------------------------------//
//! Fill an array with random numbers: use this to test rng distribution
//----------------------------------------------------------------------------//
__global__ void kernGenOnly(curandState * __restrict__ state, 
    const size_t arrLen, 
    const size_t twSize, 
    unsigned int * __restrict__ d_testDistInts);

//----------------------------------------------------------------------------//
//! Randomly generate numbers and compare character-wise matches
//----------------------------------------------------------------------------//
__global__ void kernGenComp(curandState * __restrict__ state, 
    const size_t arrLen, 
    const size_t twSize, 
    const unsigned int*__restrict__ d_targetInt,
    const unsigned int targLen,
    bool * __restrict__ d_charMatch);

//----------------------------------------------------------------------------//
//! Vectorized memory access to check for contiguous matches
//! Respectively for matching 4 character and 8 character blocks
//----------------------------------------------------------------------------//
#if TARGET_LENGTH == 4
  __global__ void kernVec4Match(const bool * __restrict__ d_charMatch,
      const size_t arrLen,
      const unsigned int targLen,
      bool * __restrict__ d_fullMatch,
      const long targetNum,
      const int offset);

#elif TARGET_LENGTH == 8
  __global__ void kernVec8Match(const bool * __restrict__ d_charMatch,
      const size_t arrLen,
      const unsigned int targLen,
      bool * __restrict__ d_fullMatch,
      const long targetNum,
      const int offset);
#endif // TARGET_LENGTH must be 4 xor 8

#endif  // #ifndef _MONKEYS_KERNELS_CUH_
