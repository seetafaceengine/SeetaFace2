#ifndef _SEETANET_SCALE_CPU_H_
#define _SEETANET_SCALE_CPU_H_

#include "SeetaNetBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <class T>
class SeetaNetScaleCPU : public SeetaNetBaseLayer<T>
{
public:
	SeetaNetScaleCPU();
	~SeetaNetScaleCPU();

	int Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource);
	int Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map);

public:
	std::vector<T> m_bias_value;
	std::vector<T> m_scale_value;
};

template <class T>
SeetaNetScaleCPU<T>::SeetaNetScaleCPU()
{
}

template <class T>
SeetaNetScaleCPU<T>::~SeetaNetScaleCPU()
{
}

template <class T>
int SeetaNetScaleCPU<T>::Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource)
{

	seeta::SeetaNet_ScaleParameter *msg = (seeta::SeetaNet_ScaleParameter *)inputparam.msg.get();
	m_scale_value.clear();
	size_t length_mean = msg->scale_param.data.size();

	for (size_t i = 0; i < length_mean; i++)
	{
		auto tmp_scale_value = msg->scale_param.data[i];

		if (tmp_scale_value < FLT_EPSILON && -tmp_scale_value < FLT_EPSILON)
			tmp_scale_value = 0;

		m_scale_value.push_back(tmp_scale_value);
	}

	m_bias_value.clear();
	size_t length_covariance = msg->bias_param.data.size();

	for (size_t i = 0; i < length_covariance; i++)
	{
		auto tmp_bias_value = msg->bias_param.data[i];

		if (tmp_bias_value < FLT_EPSILON && -tmp_bias_value < FLT_EPSILON)
			tmp_bias_value = 0;

		m_bias_value.push_back(tmp_bias_value);
	}

	int index = inputparam.bottom_index[0];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = pNetResource->feature_vector_size[index];
	this->top_data_size.resize(1);
	this->top_data_size[0] = this->bottom_data_size[0];

	return 0;
}

template <class T>
int SeetaNetScaleCPU<T>::Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	if (this->bottom_index[0] != this->top_index[0])
	{
		output_data_map[0]->data_shape = input_data_map[0]->data_shape;
		memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*output_data_map[0]->count());
	}

	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

	if (gun == nullptr || gun->size() <= 1)
	{
		T *pstart = output_data_map[0]->m_cpu.dataMemoryPtr();

		for (int n = 0; n < output_data_map[0]->data_shape[0]; n++)
		{
			for (int i = 0; i < output_data_map[0]->data_shape[1]; i++)
			{
				T val2 = m_scale_value[i];
				T val3 = 0;

				if (m_bias_value.size() > 0)
				{
					val3 = m_bias_value[i];
				}

				for (int j = 0; j < output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3]; j++)
				{
					*pstart *= val2;
					*pstart += val3;
					pstart++;
				}
			}
		}
	}
	else
	{
		auto col_size = output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
		auto batch_size = output_data_map[0]->data_shape[1] * col_size;

		for (int n = 0; n < input_data_map[0]->data_shape[0]; n++)
		{
			auto local_pstart = output_data_map[0]->m_cpu.dataMemoryPtr() + n * batch_size;
			auto bins = orz::split_bins(0, output_data_map[0]->data_shape[1], int(gun->size()));

			for (auto &bin : bins)
			{
				gun->fire([&, local_pstart, bin](int)
				{
					auto pstart = local_pstart + bin.first * col_size;

					for (int i = bin.first; i < bin.second; i++)
					{
						T val2 = m_scale_value[i];
						T val3 = 0;

						if (m_bias_value.size() > 0)
						{
							val3 = m_bias_value[i];
						}

						for (int j = 0; j < col_size; j++)
						{
							*pstart *= val2;
							*pstart += val3;
							pstart++;
						}
					}
				});
			}
		}

		gun->join();
	}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
	output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
	output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

	return 0;
}

#endif //!__SCALE_H__
