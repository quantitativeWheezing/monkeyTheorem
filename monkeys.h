/* Run iterations with matchFileMultiGPU
 */

#ifndef _MONKEYS_CUH_
#define _MONKEYS_CUH_

// includes, system
#include <stddef.h>

//----------------------------------------------------------------------------//
//! GPU: Generate numbers to match all words in a file
//! @param  fileName  file to be randomly generated
//! @param  numGPUs  number of CUDA devices available
//! @param  arrLen  how many numbers are generated and queried
//! @param  targetLen  length of target string
//! @param  typewriterSize  number of characters in alphabet used
//! @param  seed  change seed for curand
//----------------------------------------------------------------------------//

void matchFileMultiGPU(const char *fileName,
    const int numGPUs,
    const size_t arrLen,
    const unsigned int targetLen,
    const unsigned int typewriterSize,
    const unsigned int seed,
    const bool cpuCheck);

#endif  // #ifndef _MONKEYS_CUH_
