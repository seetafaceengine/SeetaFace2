#include "SeetaNet.h"
#include "SeetaNetMemoryModel.h"
#include <vector>
#include <algorithm>

#include "SeetaNetResource.h"
#include "SeetaNetBaseLayer.h"
#include "SeetaNetFeatureMap.h"

#include "SeetaNetCreateLayerMapCPU.h"

#include <algorithm>

#include "orz/mem/vat.h"
#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"

struct SeetaNet
{
	std::vector< SeetaNetBaseLayer<NetF>* > Layer_vector;
	SeetaNetResource<NetF> *tmp_NetResource = nullptr;
	std::vector<SeetaNetFeatureMap<NetF>*> feature_vector_cpu;
	SeetaNetFeatureMap<NetF> input_data_blob;
	std::map<std::string, float *> feature_value_map;
	std::map<std::string, size_t> feature_value_size_map;

	orz::Vat vat;
	std::vector<int> blob_bottom_refs;
	std::vector<int> blob_top_refs;
	std::vector<int> output_blob_indexs;
	std::vector<int> keep_blob_indexs;
};

int CreateNet(void *model, int max_batch_size, SeetaNet_DEVICE_TYPE process_device_type, void **output_net_out, int gpu_device_id)
{
	CreateNetSharedParam(model, max_batch_size, process_device_type, output_net_out, nullptr, gpu_device_id);
	return 0;
}

int CreateNetSharedParam(void *model, int max_batchsize, SeetaNet_DEVICE_TYPE process_device_type, void **output_net_out, void **output_shared_param, int gpu_device_id)
{
	// TODO: ����������early return ����Դ�ͷ�
	SeetaNetShareParam<NetF> *tmp_output_shared_param = nullptr;

	if (nullptr != output_shared_param && nullptr != (*output_shared_param))
	{
		tmp_output_shared_param = (SeetaNetShareParam<NetF> *)(*output_shared_param);

		if (tmp_output_shared_param->m_device != process_device_type)
		{
			*output_net_out = nullptr;
			return MISSMATCH_DEVICE_ID;
		}

		tmp_output_shared_param->m_refrence_counts = tmp_output_shared_param->m_refrence_counts + 1;
	}
	else
	{
		tmp_output_shared_param = new SeetaNetShareParam<NetF>;
		tmp_output_shared_param->m_refrence_counts = 1;
		tmp_output_shared_param->m_device = process_device_type;
	}

	SeetaNet *output_net_start = new SeetaNet();
	SeetaNet &output_net = *output_net_start;

	output_net.tmp_NetResource = new SeetaNetResource<NetF>;
	output_net.tmp_NetResource->max_batch_size = max_batchsize;
	output_net.tmp_NetResource->process_device_type = process_device_type;
	output_net.tmp_NetResource->colbuffer_memory_size = 0;
	output_net.tmp_NetResource->m_shared_param = tmp_output_shared_param;

	MemoryModel *ptmp_model = (MemoryModel *)model;
	ptmp_model->model_mtx.lock();

	output_net.tmp_NetResource->m_new_height = ptmp_model->m_new_height;
	output_net.tmp_NetResource->m_new_width = ptmp_model->m_new_width;

	SeetaNet_LayerParameter *first_data_param = ptmp_model->all_layer_params[0];
	seeta::SeetaNet_MemoryDataParameterProcess *memoryparameter = (seeta::SeetaNet_MemoryDataParameterProcess *)first_data_param->msg.get();

	auto blob_length = ptmp_model->vector_blob_names.size();
	output_net.tmp_NetResource->feature_vector_size.resize(blob_length);
	output_net.tmp_NetResource->feature_vector_size[0].data_dim.resize(4);
	output_net.tmp_NetResource->feature_vector_size[0].data_dim[0] = max_batchsize;

	//output_net.tmp_NetResource->feature_vector_size[0].data_dim[1] = first_data_param->memory_data_param().channels();
	output_net.tmp_NetResource->feature_vector_size[0].data_dim[1] = memoryparameter->channels;

	int local_input_height = output_net.tmp_NetResource->m_new_height > 0
		? output_net.tmp_NetResource->m_new_height
		: int(memoryparameter->height);
	int local_input_width = output_net.tmp_NetResource->m_new_width > 0
		? output_net.tmp_NetResource->m_new_width
		: int(memoryparameter->width);

	output_net.tmp_NetResource->feature_vector_size[0].data_dim[2] = local_input_height;
	output_net.tmp_NetResource->feature_vector_size[0].data_dim[3] = local_input_width;

	output_net.blob_bottom_refs.resize(blob_length, 0);
	output_net.blob_top_refs.resize(blob_length, 0);

	std::vector<int> input_data_shape = output_net.tmp_NetResource->feature_vector_size[0].data_dim;

	output_net.input_data_blob.data_shape.resize(4);

#ifdef FREE_DATA
	output_net.input_data_blob.m_cpu.ReshapeJustShape(input_data_shape);
#else
	output_net.input_data_blob.m_cpu.Reshape(input_data_shape);
#endif

	int return_pfun3 = 0;

	auto layer_length = ptmp_model->all_layer_params.size();
	output_net.Layer_vector.resize(layer_length, nullptr);

	for (size_t i = 0; i < ptmp_model->vector_blob_names.size(); i++)
	{
		output_net.tmp_NetResource->blob_name_map[ptmp_model->vector_blob_names[i]] = int(i);
	}

	for (size_t i = 0; i < layer_length; i++)
	{

		CreateLayerMapCPU<NetF>::CREATE_NET_PARSEFUNCTION *pfun = nullptr;
		int layer_type = ptmp_model->all_layer_params[i]->type;
		std::string layer_name = ptmp_model->all_layer_params[i]->name;

#ifdef _DEBUG
		std::cout << "LOG: Creating layer(" << layer_type << "): " << layer_name << std::endl;
#endif

		if (SEETANET_CPU_DEVICE == output_net.tmp_NetResource->process_device_type
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_SoftmaxLayer
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_MemoryDataLayer
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_ShapeIndexPatchLayer
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_CropLayer)
		{
			pfun = CreateLayerMapCPU<NetF>::FindRunFunciton(layer_type);
		}

		if (!pfun)
		{
			std::cerr << "ERROR: Unidentified layer(" << layer_type << "): " << layer_name << std::endl;
			ptmp_model->model_mtx.unlock();
			SeetaNetReleaseNet(reinterpret_cast<void **>(&output_net_start));

			return UNIDENTIFIED_LAYER;
		}

		SeetaNetBaseLayer<NetF> *tmp_layer = nullptr;
		pfun(tmp_layer, *(ptmp_model->all_layer_params[i]), output_net.tmp_NetResource);
		tmp_layer->m_layer_type = layer_type;

		std::vector<SeetaNetDataSize> tmp_vector_size;
		tmp_layer->GetTopSize(tmp_vector_size);
		std::vector<int> bottom_index;
		std::vector<int> top_index;

		for (int j = 0; j < ptmp_model->all_layer_params[i]->bottom_index.size(); j++)
		{
			bottom_index.push_back(ptmp_model->all_layer_params[i]->bottom_index[j]);
		}

		for (int j = 0; j < ptmp_model->all_layer_params[i]->top_index.size(); j++)
		{
			top_index.push_back(ptmp_model->all_layer_params[i]->top_index[j]);
		}

		for (int j = 0; j < tmp_vector_size.size(); j++)
		{
			SeetaNetDataSize current_size = tmp_vector_size[j];
			int index_value = top_index[j];
			output_net.tmp_NetResource->feature_vector_size[index_value] = current_size;
		}

		output_net.Layer_vector[i] = tmp_layer;
	}

	output_net.feature_vector_cpu.resize(ptmp_model->vector_blob_names.size(), nullptr);

	for (size_t i = 0; i < output_net.feature_vector_cpu.size(); i++)
	{
		SeetaNetFeatureMap<NetF> *tmp_feature_map = new SeetaNetFeatureMap<NetF>();
		tmp_feature_map->pNetResource = output_net.tmp_NetResource;
		output_net.feature_vector_cpu[i] = tmp_feature_map;
	}

	for (size_t i = 0; i < layer_length; i++)
	{
		if (SEETANET_CPU_DEVICE == output_net.tmp_NetResource->process_device_type
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_SoftmaxLayer
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_MemoryDataLayer
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_ShapeIndexPatchLayer
			|| ptmp_model->all_layer_params[i]->type == seeta::Enum_CropLayer)
		{
			//init blob as output
			for (int j = 0; j < output_net.Layer_vector[i]->top_index.size(); j++)
			{
				auto index_blob = output_net.Layer_vector[i]->top_index[j];
				std::vector<int> shape_vector;

				shape_vector = output_net.tmp_NetResource->feature_vector_size[index_blob].data_dim;

#ifdef FREE_DATA
				output_net.feature_vector_cpu[index_blob]->m_cpu.ReshapeJustShape(shape_vector);
#else
				output_net.feature_vector_cpu[index_blob]->m_cpu.Reshape(shape_vector);
#endif

				output_net.feature_vector_cpu[index_blob]->data_shape = shape_vector;
				output_net.blob_top_refs[index_blob]++;
			}

			//init blob as input
			for (size_t j = 0; j < output_net.Layer_vector[i]->bottom_index.size(); j++)
			{
				auto index_blob = output_net.Layer_vector[i]->bottom_index[j];
				std::vector<int> shape_vector;
				shape_vector = output_net.tmp_NetResource->feature_vector_size[index_blob].data_dim;

#ifdef FREE_DATA
				output_net.feature_vector_cpu[index_blob]->m_cpu.ReshapeJustShape(shape_vector);
#else
				output_net.feature_vector_cpu[index_blob]->m_cpu.Reshape(shape_vector);
#endif

				output_net.feature_vector_cpu[index_blob]->data_shape = shape_vector;
				output_net.blob_bottom_refs[index_blob]++;
			}
		}
	}

	// mark output blob
	output_net.output_blob_indexs.clear();

	for (size_t i = 0; i < blob_length; ++i)
	{
		if (output_net.blob_top_refs[i] > output_net.blob_bottom_refs[i])
		{
			output_net.output_blob_indexs.emplace_back(int(i));
		}
	}

	if (output_shared_param)
	{
		*output_shared_param = tmp_output_shared_param;
	}

	*output_net_out = output_net_start;
	ptmp_model->model_mtx.unlock();
	return 0;
}

template<typename Dtype, typename Dtype_input>
void OpencvDataToBlob(Dtype_input *inputMat, int height, int width, int nchannels, int num, SeetaNetBlobCpu<Dtype> &output_cube)
{
	std::vector<int> shape_vector;
	shape_vector.push_back(num);
	shape_vector.push_back(nchannels);
	shape_vector.push_back(height);
	shape_vector.push_back(width);

	// @TODO: no need reshape here
#ifdef FREE_DATA
	output_cube.ReshapeJustShape(shape_vector);
#else
	output_cube.Reshape(shape_vector);
#endif

	std::vector<int> index_vector;
	index_vector.resize(4, 0);
	int index = 0;

	for (int n = 0; n < num; n++)
	{
		index_vector[0] = n;

		for (int i = 0; i < height; i++)
		{
			index_vector[2] = i;

			for (int j = 0; j < width; j++)
			{
				index_vector[3] = j;

				for (int nc = 0; nc < nchannels; nc++)
				{
					index_vector[1] = nc;
					Dtype_input value = inputMat[index++];
					output_cube.data_at(index_vector) = value;
				}
			}
		}
	}
}

template <typename Dtype_input, typename Dtype>
void InputData2Blob(SeetaNet_InputOutputData *pinput_Data, SeetaNetBlobCpu<Dtype> &output_cube);

template <>
void InputData2Blob<char, float>(SeetaNet_InputOutputData *pinput_Data, SeetaNetBlobCpu<float> &output_cube)
{
	OpencvDataToBlob<float, unsigned char>(reinterpret_cast<unsigned char *>(pinput_Data[0].data_point_char), pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_cube);
}

template <>
void InputData2Blob<float, float>(SeetaNet_InputOutputData *pinput_Data, SeetaNetBlobCpu<float> &output_cube)
{
	OpencvDataToBlob<float, float>(pinput_Data[0].data_point_float, pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_cube);

}

template<typename Dtype, typename Dtype_input>
void OutWidthDataToBlob(Dtype_input *inputMat, int height, int width, int nchannels, int num, SeetaNetBlobCpu<Dtype> &output_blob)
{
	std::vector<int> shape_vector;
	shape_vector.push_back(num);
	shape_vector.push_back(nchannels);
	shape_vector.push_back(height);
	shape_vector.push_back(width);
	// @TODO: no need reshape here

#ifdef FREE_DATA
	output_blob.ReshapeJustShape(shape_vector);
#else
	output_blob.Reshape(shape_vector);
#endif

	std::vector<int> index_vector;
	index_vector.resize(4, 0);
	int index = 0;

	for (int n = 0; n < num; n++)
	{
		index_vector[0] = n;

		for (int nc = 0; nc < nchannels; nc++)
		{
			index_vector[1] = nc;

			for (int i = 0; i < height; i++)
			{
				index_vector[2] = i;

				for (int j = 0; j < width; j++)
				{
					index_vector[3] = j;
					output_blob.data_at(index_vector) = inputMat[index++];
				}
			}
		}
	}
}

template <typename Dtype>
int RunNetTemplate(SeetaNet *output_net, int counts, SeetaNet_InputOutputData *pinput_Data, int input_type)
{
	orz::ctx::lite::bind<orz::Vat> _bind_vat(output_net->vat);

#ifdef FREE_DATA
	// prepare input blob size, not change input blob for now
	output_net->input_data_blob.m_cpu.dispose();

	// dispose all blob, not input_blob
	for (auto &blob : output_net->feature_vector_cpu)
	{
		blob->m_cpu.dispose();
	}
#endif

	output_net->input_data_blob.data_shape[0] = pinput_Data[0].number;
	output_net->input_data_blob.data_shape[1] = pinput_Data[0].channel;
	output_net->input_data_blob.data_shape[2] = pinput_Data[0].height;
	output_net->input_data_blob.data_shape[3] = pinput_Data[0].width;

#ifdef FREE_DATA
	output_net->input_data_blob.m_cpu.ReshapeJustShape(output_net->input_data_blob.data_shape);
	auto &input_blob = output_net->input_data_blob;
	input_blob.m_cpu.set_raw_data(output_net->vat.calloc_shared<NetF>(input_blob.m_cpu.count()));
#else
	output_net->input_data_blob.m_cpu.Reshape(output_net->input_data_blob.data_shape);
#endif

	if ((pinput_Data[0].number < 0)
		|| (pinput_Data[0].number > output_net->tmp_NetResource->feature_vector_size[0].data_dim[0]))
	{
		return -1;
	}

	if (SEETANET_CPU_DEVICE == output_net->tmp_NetResource->process_device_type)
	{
		if (input_type == SEETANET_BGR_IMGE_CHAR || input_type == SEETANET_BGR_IMGE_FLOAT)
		{
			InputData2Blob<Dtype, float>(pinput_Data, output_net->input_data_blob.m_cpu);
			output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
		}
		else
			if (SEETANET_NCHW_FLOAT == input_type)
			{
				OutWidthDataToBlob<float, float>(pinput_Data[0].data_point_float, pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_net->input_data_blob.m_cpu);
				output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
			}
			else
			{
				return -1;
			}
	}

	auto local_blob_bottom_refs = output_net->blob_bottom_refs;

	for (auto &blob_index : output_net->output_blob_indexs)
	{
		local_blob_bottom_refs[blob_index]++;
	}

	for (auto &blob_index : output_net->keep_blob_indexs)
	{
		local_blob_bottom_refs[blob_index]++;
	}

	int return_result = 0;
	std::vector<SeetaNetFeatureMap<NetF>*> tmp_data_vector;
	tmp_data_vector.push_back(&(output_net->input_data_blob));

	auto run_length = output_net->Layer_vector.size();

	for (size_t i = 0; i < run_length; i++)
	{
		std::vector<int64_t> tmp_bottom_index = output_net->Layer_vector[i]->bottom_index;
		std::vector<int64_t> tmp_top_index = output_net->Layer_vector[i]->top_index;

		std::vector<SeetaNetFeatureMap<NetF>*> bottom_blob_vector;
		std::vector<SeetaNetFeatureMap<NetF>*> top_blob_vector;

		for (int j = 0; j < tmp_bottom_index.size(); j++)
		{
			bottom_blob_vector.push_back(output_net->feature_vector_cpu[tmp_bottom_index[j]]);
		}

		for (int j = 0; j < tmp_top_index.size(); j++)
		{
#ifdef FREE_DATA
			// spiecl for inplace layer
			if (tmp_bottom_index.size() > j && tmp_bottom_index[j] == tmp_top_index[j])
			{
			}
			else
			{
				auto &blob = output_net->feature_vector_cpu[tmp_top_index[j]];
				blob->m_cpu.set_raw_data(output_net->vat.calloc_shared<NetF>(blob->m_cpu.count())); // calloc top blob memory
			}
#endif

			top_blob_vector.push_back(output_net->feature_vector_cpu[tmp_top_index[j]]);
		}

		if (bottom_blob_vector.empty())
		{
			bottom_blob_vector = tmp_data_vector;
		}

		auto layer = output_net->Layer_vector[i];
		return_result = layer->Process(bottom_blob_vector, top_blob_vector);

#ifdef FREE_DATA
		// free input dummy blob
		input_blob.m_cpu.dispose();

		// free useless blob
		for (int j = 0; j < tmp_bottom_index.size(); j++)
		{
			auto blob_index = tmp_bottom_index[j];
			local_blob_bottom_refs[blob_index]--;

			if (local_blob_bottom_refs[blob_index] == 0)
			{
				output_net->feature_vector_cpu[blob_index]->m_cpu.dispose();
			}
		}
#endif

		if (0 != return_result)
		{
			std::cout << "Layer(" << i << ")\t" << "error!" << return_result << std::endl;
			break;
		}
	}

	return return_result;
}

int RunNetChar(void *output_net, int counts, SeetaNet_InputOutputData *pinput_Data)
{
	SeetaNet *tmp_output_net = (SeetaNet *)output_net;
	int return_result = RunNetTemplate<char>(tmp_output_net, counts, pinput_Data, pinput_Data[0].buffer_type);
	return return_result;
}

int RunNetFloat(void *output_net, int counts, SeetaNet_InputOutputData *pinput_Data)
{
	SeetaNet *tmp_output_net = (SeetaNet *)output_net;
	int return_result = RunNetTemplate<float>(tmp_output_net, counts, pinput_Data, pinput_Data[0].buffer_type);
	return return_result;
}

int SeetaNetGetFeatureMap(const char *buffer_name, void *pNetIn, SeetaNet_InputOutputData *outputData)
{
	int index_value(0);
	SeetaNet *pNet = (SeetaNet *)pNetIn;

	if (pNet->tmp_NetResource->blob_name_map.find(buffer_name) != pNet->tmp_NetResource->blob_name_map.end())
	{
		int index = pNet->tmp_NetResource->blob_name_map[buffer_name];

		outputData->number = pNet->feature_vector_cpu[index]->data_shape[0];
		outputData->buffer_type = SEETANET_NCHW_FLOAT;
		outputData->channel = pNet->tmp_NetResource->feature_vector_size[index].data_dim[1];
		outputData->width = pNet->tmp_NetResource->feature_vector_size[index].data_dim[3];
		outputData->height = pNet->tmp_NetResource->feature_vector_size[index].data_dim[2];

		int size_memory = outputData->number * outputData->channel * outputData->height * outputData->width;

		std::vector<int32_t> out_feature_shape = pNet->feature_vector_cpu[index]->data_shape;
		outputData->number = out_feature_shape[0];
		outputData->channel = out_feature_shape[1];
		outputData->height = out_feature_shape[2];
		outputData->width = out_feature_shape[3];

		if (pNet->feature_value_map.find(buffer_name) == pNet->feature_value_map.end())
		{

			float *out_value_innerl(nullptr);
			out_value_innerl = new float[size_memory];
			memset(out_value_innerl, 0, size_memory * sizeof(float));
			pNet->feature_value_map.insert(std::pair<std::string, float *>(buffer_name, out_value_innerl));
			pNet->feature_value_size_map.insert(std::pair<std::string, size_t>(buffer_name, size_memory));
			outputData->data_point_float = out_value_innerl;
		}
		else
		{
			if (nullptr == pNet->feature_value_map[buffer_name])
			{
				pNet->feature_value_map[buffer_name] = new float[size_memory];
				pNet->feature_value_size_map[buffer_name] = static_cast<size_t>(size_memory);
			}
			else
				if (pNet->feature_value_size_map.find(buffer_name) == pNet->feature_value_size_map.end() ||
					pNet->feature_value_size_map[buffer_name] < size_memory)
				{
					delete[] pNet->feature_value_map[buffer_name];
					pNet->feature_value_map[buffer_name] = new float[size_memory];
					pNet->feature_value_size_map[buffer_name] = static_cast<size_t>(size_memory);
				}

			outputData->data_point_float = pNet->feature_value_map[buffer_name];
		}

		index_value = 0;

		if (pNet->feature_vector_cpu[index]->dwStorageType == DATA_CPU_WIDTH)
		{

			int64_t out_puts_counts = 1;

			for (int index_shape = 0; index_shape < out_feature_shape.size(); index_shape++)
			{
				out_puts_counts *= out_feature_shape[index_shape];
			}

			auto blob_data = pNet->feature_vector_cpu[index]->m_cpu.data();

			if (blob_data)
			{
				memcpy(outputData->data_point_float, blob_data, out_puts_counts * sizeof(float));
			}
		}

		return 0;
	}
	else
	{
		(outputData->data_point_float) = nullptr;
		return BLOB_NAME_NOT_EXIST;
	}

	return 0;
}

int SeetaNetGetAllFeatureMap(void *pNetIn, int *number, SeetaNet_InputOutputData **outputData)
{
	SeetaNet *pNet = (SeetaNet *)pNetIn;
	auto all_size = pNet->tmp_NetResource->blob_name_map.size();
	SeetaNet_InputOutputData *outputDatatmp = new SeetaNet_InputOutputData[all_size];
	*number = int(all_size);

	for (auto tmp_iter = pNet->tmp_NetResource->blob_name_map.begin(); tmp_iter != pNet->tmp_NetResource->blob_name_map.end(); tmp_iter++)
	{
		int index = pNet->tmp_NetResource->blob_name_map[tmp_iter->first];

		SeetaNetGetFeatureMap(tmp_iter->first.c_str(), pNetIn, &(outputDatatmp[index]));
	}

	*outputData = outputDatatmp;

	return 0;
}

void SeetaNetFreeAllFeatureMap(void *pNetIn, const SeetaNet_InputOutputData *outputData)
{
	SeetaNet *pNet = (SeetaNet *)pNetIn;
	delete[] outputData;
}

void SeetaNetReleaseNet(void **pNetIn)
{
	if (*pNetIn)
	{
		SeetaNet *pNet = (SeetaNet *)*pNetIn;

		for (auto tmp_iteration = pNet->feature_value_map.begin(); tmp_iteration != pNet->feature_value_map.end(); tmp_iteration++)
		{
			delete[] tmp_iteration->second;
			tmp_iteration->second = nullptr;
		}

		pNet->feature_value_map.clear();
		pNet->feature_value_size_map.clear();

		for (int i = 0; i < pNet->Layer_vector.size(); i++)
		{
			pNet->Layer_vector[i]->Exit();
			delete pNet->Layer_vector[i];
		}

		pNet->Layer_vector.clear();

		for (int i = 0; i < pNet->Layer_vector.size(); i++)
		{
			delete pNet->Layer_vector[i];
		}

		pNet->Layer_vector.clear();
		pNet->tmp_NetResource->blob_name_map.clear();

		for (int i = 0; i < pNet->feature_vector_cpu.size(); i++)
		{
			delete pNet->feature_vector_cpu[i];
		}

		pNet->feature_vector_cpu.clear();
		pNet->tmp_NetResource->m_shared_param->m_refrence_counts -= 1;

		if (0 == pNet->tmp_NetResource->m_shared_param->m_refrence_counts)
		{
			delete  pNet->tmp_NetResource->m_shared_param;
			pNet->tmp_NetResource->m_shared_param = nullptr;
		}

		if (pNet->tmp_NetResource)
		{
			delete pNet->tmp_NetResource;
			pNet->tmp_NetResource = nullptr;
		}

		// free the memory, controlled by pNet->vat, avoid memory leak
#ifdef FREE_DATA
		pNet->input_data_blob.m_cpu.dispose();
#endif

		delete pNet;
		pNet = nullptr;
		*pNetIn = nullptr;
	}
}

int SeetaNetReleaseSharedParam(void **shared_param)
{
	*shared_param = nullptr;
	return 0;
}

void SeetaNetKeepBlob(struct SeetaNet_Net *net, const char *blob_name)
{
	SeetaNet *inner_net = (SeetaNet *)net;

	auto it = inner_net->tmp_NetResource->blob_name_map.find(blob_name);

	if (it == inner_net->tmp_NetResource->blob_name_map.end())
		return;

	inner_net->keep_blob_indexs.push_back(it->second);
}

void SeetaNetKeepNoBlob(struct SeetaNet_Net *net)
{
	SeetaNet *inner_net = (SeetaNet *)net;
	inner_net->keep_blob_indexs.clear();
}

void SeetaNetKeepAllBlob(struct SeetaNet_Net *net)
{
	SeetaNet *inner_net = (SeetaNet *)net;
	inner_net->keep_blob_indexs.clear();
	auto blob_length = inner_net->feature_vector_cpu.size();

	for (size_t i = 0; i < blob_length; ++i)
		inner_net->keep_blob_indexs.push_back(int(i));
}

int SeetaNetHasKeptBlob(struct SeetaNet_Net *net, const char *blob_name)
{
	SeetaNet *inner_net = (SeetaNet *)net;

	auto it = inner_net->tmp_NetResource->blob_name_map.find(blob_name);

	if (it == inner_net->tmp_NetResource->blob_name_map.end())
		return 0;

	int blob_index = it->second;

#define HAS(vec, val) (std::find((vec).begin(), (vec).end(), (val)) != (vec).end())
	return HAS(inner_net->output_blob_indexs, blob_index) || HAS(inner_net->keep_blob_indexs, blob_index);
#undef HAS
}

void *GetNetSharedParam(void *net)
{
	SeetaNet *tmp_output_net = (SeetaNet *)net;
	return tmp_output_net->tmp_NetResource->m_shared_param;
}
