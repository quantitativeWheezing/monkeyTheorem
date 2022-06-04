/* Calls functions with parameters specified in config.h
 * e.g. can run tests if requested
 * can generate text for all files in ./textFiles if one uses
 * ./build/bin/monkeyTheorem "all"
 * can generate text for an individual file in ./textFiles, like
 * ./build/bin/monkeyTheorem "shakespeare.txt"
 */

// includes, system
#include <time.h>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include "helper_cuda.h"
#include "config.h"
#include "test.h"
#include "monkeys.h"

//----------------------------------------------------------------------------//
// assert requirements of input parameters
//----------------------------------------------------------------------------//
#if ALPHABET == 0
  constexpr unsigned int NUM_KEYS = 26;
#elif ALPHABET == 1
  constexpr unsigned int NUM_KEYS = 52;
#elif ALPHABET == 2
  constexpr unsigned int NUM_KEYS = 94;
#endif // ALPHABET must be in {0,1,2}

static_assert((TARGET_LENGTH == 8 || TARGET_LENGTH == 4),
      "TARGET_LENGTH must be 4 xor 8");

static_assert((REDUCTION_VEC_SIZE == 16 || REDUCTION_VEC_SIZE == 8 ||
        REDUCTION_VEC_SIZE == 4),
      "REDUCTION_VEC_SIZE must be {4,8,16}");

static_assert((MATCH_VEC_SIZE == 4 || MATCH_VEC_SIZE == 2),
      "MATCH_VEC_SIZE must be 2 xor 4");

static_assert((ALPHABET == 0 || ALPHABET == 1 || ALPHABET == 2),
      "ALPHABET must be in {0,1,2}");

static_assert((N_THREADS == 512 || N_THREADS == 256 || N_THREADS == 128 ||
      N_THREADS == 64 || N_THREADS == 32 || N_THREADS == 16),
      "N_THREADS must be in {16,32,64,128,256,512}");

static_assert(!(N_NUMS%N_THREADS), "N_THREADS must divide N_NUMS");

static_assert(!(N_NUMS%TARGET_LENGTH), "TARGET_LENGTH must divide N_NUMS");

static_assert(!(N_NUMS%(N_THREADS*MATCH_VEC_SIZE)),
      "(N_THREADS*MATCH_VEC_SIZE) must divide N_NUMS");

int main(int argc, char *argv[])
{

  // get number of GPUs
  int numGPUs; checkCudaErrors(cudaGetDeviceCount(&numGPUs));

  // initialize srand with SEED if provided
  if (SEED) {srand(SEED);}
  else {srand(time(NULL));}

#if RUN_TESTS == 1

  // verify uniformity of rng
  gpuTestDist(N_NUMS, NUM_KEYS, 128, SEED);
  cpuTestDist(N_NUMS, NUM_KEYS, 128);

  // verify that GPU is faster than CPU for
  matchFileCPU("sample1.txt", N_NUMS, TARGET_LENGTH, NUM_KEYS);
  matchFileMultiGPU("sample1.txt", "./textFiles/", "./output/",
      numGPUs, N_NUMS, TARGET_LENGTH,
      NUM_KEYS, SEED, N_THREADS, false);

  // verify that GPU results match cpu results
  matchFileMultiGPU("sample2.txt", "./textFiles/", "./output/",
      numGPUs, N_NUMS, TARGET_LENGTH,
      NUM_KEYS, SEED, N_THREADS, true);

#endif

  const unsigned int numFiles = 2;
  char fileNames[numFiles][32] =
  {
    "shakespeare.txt",
    "trump.txt",
  };

  
  if (argc > 1) {
    if (!strcmp(argv[1], "all")) {
      for(unsigned int i = 0; i < numFiles; i++) {
        matchFileMultiGPU(fileNames[i], "./textFiles/", "./output/",
            numGPUs, N_NUMS, TARGET_LENGTH,
            NUM_KEYS, SEED, N_THREADS, false);
      }
    }

    else {
      matchFileMultiGPU(argv[1], "./textFiles/", "./output/",
            numGPUs, N_NUMS, TARGET_LENGTH,
            NUM_KEYS, SEED, N_THREADS, false);
    }
  }

  return 0;
}
