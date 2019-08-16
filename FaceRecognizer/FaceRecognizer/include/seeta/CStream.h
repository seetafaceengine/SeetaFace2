#ifndef INC_SEETA_C_STREAM_H
#define INC_SEETA_C_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

typedef size_t SeetaStreamWrite( void *obj, const char *data, size_t length );
typedef size_t SeetaStreamRead( void *obj, char *data, size_t length );

#ifdef __cplusplus
}
#endif

#endif // INC_SEETA_C_STREAM_H
