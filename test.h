/* Verify the uniformity of rng and also use cpu to verify gpu results
 */

// includes, system
#ifndef _TEST_H_
#define _TEST_H_

#include <stddef.h>

//----------------------------------------------------------------------------//
//! GPU: verify uniformity of rng
//----------------------------------------------------------------------------//
void gpuTestDist(const size_t arrLen,
    const unsigned int typewriterSize, 
    const unsigned int nRuns,
    const unsigned int seed);

//----------------------------------------------------------------------------//
//! CPU: verify uniformity of rng
//----------------------------------------------------------------------------//
void cpuTestDist(const size_t arrLen,
    const unsigned int typewriterSize, 
    const unsigned int nRuns);

//----------------------------------------------------------------------------//
//! CPU: Generate numbers to match all words in a file
//----------------------------------------------------------------------------//
void matchFileCPU(const char *fileName,
    const size_t arrLen,
    const unsigned int targetLen,
    const unsigned int typewriterSize);

#endif  // #ifndef _TEST_H_
