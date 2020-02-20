#ifndef INC_SEETA_C_TRACKING_FACEINFO_H
#define INC_SEETA_C_TRACKING_FACEINFO_H

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

#endif // INC_SEETA_C_TRACKING_FACEINFO_H
