#ifndef _READ_FROM_SEETANET_LAYER_H_
#define _READ_FROM_SEETANET_LAYER_H_

#include <stdint.h>
#include <stddef.h>
typedef enum
{
    NULL_PTR = -1,
    FILE_NOT_EXIST = -2,
    BLOB_NOT_EXIST = -3,
    LAYER_READ_NOT_FOUND = -4,
    LAYER_PROCESS_NOT_FOUND = -5,
    BUFFER_LENGTH_LESS_ZERO = -6,
    LAYER_CREATE_NOT_FOUND = -7,
} ReadCNNErrorNum;

int SeetaNetReadModelFromBuffer( const char *buffer, size_t buffer_length, void **model );
int ReadAllContentFromFile( const char *inputfilename, char **ppbuffer, int64_t &file_length );
int SeetaNetReleaseModel( void **model );

int SeetaNetModelResetInput( void *model, int width, int height );

#endif