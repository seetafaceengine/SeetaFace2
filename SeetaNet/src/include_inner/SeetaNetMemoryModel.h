#ifndef _SEETANET_MEMORY_MODEL_H_
#define _SEETANET_MEMORY_MODEL_H_

#include "SeetaNetProto.h"
#include <mutex>

struct MemoryModel
{
	std::vector<seeta::SeetaNet_LayerParameter *> all_layer_params;
	std::vector<std::string> vector_blob_names;
	std::vector<std::string> vector_layer_names;

	std::mutex model_mtx;

	/* saving resized input */
	int m_new_width = -1;
	int m_new_height = -1;
};

#endif
