#pragma once

#include "Struct.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SeetaTrackingFaceInfo
{
	SeetaRect pos;
    float score;

    int frame_no;
    int PID;
};

struct SeetaTrackingFaceInfoArray
{
	struct SeetaTrackingFaceInfo *data;
	int size;
};

#ifdef __cplusplus
}
#endif
