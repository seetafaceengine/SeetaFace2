#ifndef _SEETANET_H_
#define _SEETANET_H_

#include "SeetaNetStruct.h"

#define NetF float

enum MyEnum
{
	BLOB_NAME_NOT_EXIST = -1,
};

/**
 * \brief
 * \param model
 * \param max_batch_size
 * \param process_device_type
 * \param output_net_out
 * \param gpu_device_id, not use
 * \return
 */
int CreateNet(void *model, int max_batch_size, SeetaNet_DEVICE_TYPE process_device_type, void **output_net_out, int gpu_device_id = 0);

int CreateNetSharedParam(void *model, int max_batchsize, SeetaNet_DEVICE_TYPE process_device_type, void **output_net_out, void **output_shared_param, int gpu_device_id = 0);

int RunNetChar(void *output_net_out, int counts, SeetaNet_InputOutputData *pinput_Data);
int RunNetFloat(void *output_net_out, int counts, SeetaNet_InputOutputData *pinput_Data);
int SeetaNetGetFeatureMap(const char *buffer_name, void *pNetIn, SeetaNet_InputOutputData *outputData);

int SeetaNetGetAllFeatureMap(void *pNetIn, int *number, SeetaNet_InputOutputData **outputData);
void SeetaNetFreeAllFeatureMap(void *pNetIn, const SeetaNet_InputOutputData *outputData);

void *GetNetSharedParam(void *net);

void SeetaNetReleaseNet(void **pNetIn);

int SeetaNetReleaseSharedParam(void **shared_param);

void SeetaNetKeepBlob(struct SeetaNet_Net *net, const char *blob_name);

void SeetaNetKeepNoBlob(struct SeetaNet_Net *net);

void SeetaNetKeepAllBlob(struct SeetaNet_Net *net);

int SeetaNetHasKeptBlob(struct SeetaNet_Net *net, const char *blob_name);

#endif
