/* Keep io neat 
 */

#ifndef _IO_H_
#define _IO_H_

// includes, system
#include <stdio.h>
#include <stddef.h>

struct ioPar
{

  //! file to be randomly generated
  const char *fName;

  //! file to be randomly generated
  const unsigned int twSize;

  //! number of valid characters in input file
  const size_t numChars;

  //! array for characters
  char *fChars;

  //! array for uint translation of characters
  unsigned int *fUints;

  //! file for output data
  FILE *fOut;

};

//----------------------------------------------------------------------------//
//! Initializing function: make io struct to simplify output
//----------------------------------------------------------------------------//
struct ioPar ioInit(const char *outPre,
    const char *fileName,
    const char *inpDir,
    const char *outDir,
    const unsigned int alph);

//----------------------------------------------------------------------------//
//! Print header to output file
//----------------------------------------------------------------------------//
void ioHeader(struct ioPar &io,
    const bool gpu);

//----------------------------------------------------------------------------//
//! Print device info to output file
//----------------------------------------------------------------------------//
void ioDevInfo(struct ioPar &io,
    const unsigned int devId,
    const char *devName);

//----------------------------------------------------------------------------//
//! Print titles of output columns
//----------------------------------------------------------------------------//
void ioColTitles(struct ioPar &io,
    const bool gpu);

//----------------------------------------------------------------------------//
//! If a target was matched, write info about matching it to output file
//----------------------------------------------------------------------------//
void ioWord(struct ioPar &io,
    const char *targStr,
    const unsigned int targLen,
    const unsigned int numsSoFar,
    const unsigned int devId,
    const bool gpu);

//----------------------------------------------------------------------------//
//! Print footer to output file
//----------------------------------------------------------------------------//
void ioFooter(struct ioPar &io,
    const double elapsed);

//----------------------------------------------------------------------------//
//! Free mem of io members and close its file member
//----------------------------------------------------------------------------//
void ioFree(struct ioPar &io);

#endif  // #ifndef _IO_H_
