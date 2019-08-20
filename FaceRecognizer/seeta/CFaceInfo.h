#pragma once

#include "Common/CStruct.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SeetaFaceInfo
{
    SeetaRect pos;
    float score;
};

struct SeetaFaceInfoArray
{
    struct SeetaFaceInfo *data;
    int size;
};

#ifdef __cplusplus
}
#endif
