#ifndef _SEETANET_STRUCT_H
#define _SEETANET_STRUCT_H

#if defined(SEETA_EXPORTS)
    #define SEETANET_EXPORTS
#endif

#if defined(_MSC_VER)
    #ifdef SEETANET_EXPORTS
        #define SEETANET_API __declspec(dllexport)
    #else
        #define SEETANET_API __declspec(dllimport)
    #endif
#else
    #define SEETANET_API
#endif

#ifdef __cplusplus
    #define SEETANET_C_API extern "C" SEETANET_API
#else
    #define SEETANET_C_API SEETANET_API
#endif

/**
* @brief The supported device.
*/
enum SeetaNet_DEVICE_TYPE
{
    SEETANET_CPU_DEVICE = 0,     /**< CPU, default */
    SEETANET_GPU_DEVICE = 1      /**< GPU, only supported in gpu version, reserved*/
};
typedef enum SeetaNet_DEVICE_TYPE SeetaNet_DEVICE_TYPE;

/**
* @brief The dummy model structure
*/
struct SeetaNet_Model;
typedef struct SeetaNet_Model SeetaNet_Model;

/**
* @brief The dummy net structure
*/
struct SeetaNet_Net;
typedef struct SeetaNet_Net SeetaNet_Net;


/**
* @brief The dummy SharedParam structure
*/
struct SeetaNet_SharedParam;
typedef struct SeetaNet_SharedParam SeetaNet_SharedParam;

//for buffer_type enum
typedef enum
{
    SEETANET_BGR_IMGE_CHAR = 0,
    SEETANET_BGR_IMGE_FLOAT = 1,
    SEETANET_NCHW_FLOAT = 2,

} SEETANET_BUFFER_STORAGE_ODER_TYPE;

/**
* @brief The base data structure
*/
struct SeetaNet_InputOutputData
{
    float *data_point_float;    /**< Used in output mode, pointing to the specific blob */
    unsigned char *data_point_char;     /**< Used in input mode, pointing to image data */
    int number;                 /**< Number of the batch size */
    int channel;                /**< Number of the channels */
    int width;                  /**< Width of the blob (or input image) */
    int height;                 /**< Height of the blob (or input image) */
    int buffer_type;            /**< Not used reserve parameter, 0 for default (means local memory data)*/
};
typedef struct SeetaNet_InputOutputData SeetaNet_InputOutputData;


/**
* @brief The global error code
*/
enum SeetaNet_ErrorCode
{
    NOERROR = 0,        /**< No error */
    UNIDENTIFIED_LAYER = 1,     /**< Got an unidentified layer */
    MISSMATCH_DEVICE_ID = 2,    /**< Missmatch  */
};
typedef enum SeetaNet_ErrorCode SeetaNet_ErrorCode;

#endif
