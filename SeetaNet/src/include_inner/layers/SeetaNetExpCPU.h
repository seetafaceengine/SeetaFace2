#ifndef _SEETANET_EXP_CPU_H_
#define _SEETANET_EXP_CPU_H_

#include "SeetaNetBaseLayer.h"
#include <cmath>
#include <cstring>

template <class T>
class SeetaNetExpCPU : public SeetaNetBaseLayer<T>
{
public:
	SeetaNetExpCPU();
	~SeetaNetExpCPU();

	int Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource);
	int Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map);

public:
	T m_scale_in;
	T m_scale_out;

private:
	int ProcessScaleInNotOne();
	int ProcessScaleInOne();
	int ProcessScaleOutNotOne();
};

template <class T>
SeetaNetExpCPU<T>::SeetaNetExpCPU()
{
}

template <class T>
SeetaNetExpCPU<T>::~SeetaNetExpCPU()
{
}

template <class T>
int SeetaNetExpCPU<T>::Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource)
{
	seeta::SeetaNet_ExpParameter *msg = (seeta::SeetaNet_ExpParameter *)inputparam.msg.get();
	m_scale_in = msg->scale;
	m_scale_out = msg->shift;

	int index = inputparam.bottom_index[0];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = pNetResource->feature_vector_size[index];

	this->top_data_size.resize(1);
	this->top_data_size[0] = this->bottom_data_size[0];

	return 0;
}

template<typename T, typename FUNC>
static void each_do(T *arr, size_t size, FUNC func)
{
	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

	if (gun == nullptr || gun->size() <= 1)
	{
		for (size_t i = 0; i < size; ++i)
		{
			func(*arr);
			++arr;
		}
	}
	else
	{
		auto bins = orz::lsplit_bins(0, size, gun->size());

		for (auto &bin : bins)
		{
			gun->fire([&, bin](int)
			{
				auto local_arr = arr + bin.first;

				for (size_t i = bin.first; i < bin.second; ++i)
				{
					func(*local_arr);
					++local_arr;
				}
			});
		}

		gun->join();
	}
}

template <class T>
int SeetaNetExpCPU<T>::Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	if (this->bottom_index[0] != this->top_index[0])
	{
		output_data_map[0]->data_shape = input_data_map[0]->data_shape;
		memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*output_data_map[0]->count());
	}

	auto type = (m_scale_in != T(1) ? 0x01 : 0x00) | (m_scale_out != T(1) ? 0x02 : 0x00);

	switch (type)
	{
	default:
	case 0: // m_scale_in == 1, m_scale_out == 1,
		each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::exp(val);
		});
		break;
	case 1: // m_scale_in != 1, m_scale_out == 1,
		each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::exp(val * m_scale_in);
		});
		break;
	case 2: // m_scale_in == 1, m_scale_out != 1,
		each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::exp(val) * m_scale_out;
		});
		break;
	case 3: // m_scale_in != 1, m_scale_out != 1,
		each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::exp(val * m_scale_in) * m_scale_out;
		});
		break;
	}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
	output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
	output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

	return 0;
};

#endif //!__EXP_LAYER_H__
