#ifndef INC_SEETA_C_STRUCT_H
#define INC_SEETA_C_STRUCT_H

#ifdef _MSC_VER
    #ifdef SEETA_EXPORTS
        #define SEETA_API __declspec(dllexport)
    #else
        #define SEETA_API __declspec(dllimport)
    #endif
#else
    #define SEETA_API __attribute__ ((visibility("default")))
#endif

#define SEETA_C_API extern "C" SEETA_API

#define INCLUDED_SEETA_CSTRUCT

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct SeetaImageData
{
    int width;
    int height;
    int channels;
    unsigned char *data;
};

struct SeetaPoint
{
    int x;
    int y;
};

struct SeetaPointF
{
    double x;
    double y;
};

struct SeetaSize
{
    int width;
    int height;
};

struct SeetaRect
{
    int x;
    int y;
    int width;
    int height;
};

struct SeetaRegion
{
    int top;
    int bottom;
    int left;
    int right;
};

enum SeetaDevice
{
    SEETA_DEVICE_AUTO = 0,
    SEETA_DEVICE_CPU  = 1,
    SEETA_DEVICE_GPU  = 2,
};

struct SeetaModelSetting
{
    enum SeetaDevice device;
    int id; // when device is GPU, id means GPU id
    const char **model; // model string terminate with nullptr
};

struct SeetaBuffer
{
    void *buffer;
    int64_t size;
};

struct SeetaModelBuffer
{
    enum SeetaDevice device;
    int id; // when device is GPU, id means GPU id
    const SeetaBuffer *buffer; // input buffers, terminate with empty buffer(buffer=nullptr, size=0)
};

#ifdef __cplusplus
}
#endif

#endif // INC_SEETA_C_STRUCT_H
