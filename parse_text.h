/* Given a fixed alphabet, count the number of valid characters in a file, and
 * write char (uint) arrays with the valid characters (uint translation)
 */

#ifndef _PARSE_TEXT_H_
#define _PARSE_TEXT_H_

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

#endif  // #ifndef _PARSE_TEXT_H_
