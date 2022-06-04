/* Given a fixed alphabet, count the number of valid characters in a file, and
 * write char (uint) arrays with the valid characters (uint translation)
 */

#ifndef _IO_H_
#define _IO_H_

// includes, system
#include <stddef.h>

//----------------------------------------------------------------------------//
//! Count how many valid characters are contained in a file
//----------------------------------------------------------------------------//
size_t getFileLen(const char *fileName);

//----------------------------------------------------------------------------//
//! Write ints and chars to arrays used in the calculation
//----------------------------------------------------------------------------//
void readFile(const char *fileName,
    char *fileChars,
    unsigned int *fileUints);

//----------------------------------------------------------------------------//
//! Replaces "\n" with "\\" so that output files are neat
//----------------------------------------------------------------------------//
char *removeSlash(const char *string,
    const size_t stringLen);

void ioInit(struct ioPar &io,
    const char *outPre,
    const char *fileName,
    const char *inpDir,
    const char *outDir);

void ioHeader(struct ioPar &io,
    const char *fileName,
    const unsigned int twSize);

void ioFooter(struct ioPar &io,
    const char *fileName,
    const double elapsed);

void ioFree(struct ioPar &io);

#endif  // #ifndef _IO_H_
