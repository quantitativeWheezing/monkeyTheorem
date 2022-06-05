/* Run iterations with matchFileMultiGPU
 */

#ifndef _MONKEYS_H_
#define _MONKEYS_H_

// includes, system
#include <stddef.h>

//----------------------------------------------------------------------------//
//! GPU: Generate numbers to match all words in a file
//----------------------------------------------------------------------------//

void matchFileMultiGPU(const char *fileName,
    const char *inpDir,
    const char *outDir,
    const int numGPUs,
    const size_t arrLen,
    const unsigned int targLen,
    const unsigned int alph,
    const unsigned int seed,
    const unsigned int numThreads,
    const bool cpuCheck);

#endif  // #ifndef _MONKEYS_H_
