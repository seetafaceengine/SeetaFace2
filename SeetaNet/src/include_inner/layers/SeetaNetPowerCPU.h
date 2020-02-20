#ifndef _SEETANET_POWER_CPU_H_
#define _SEETANET_POWER_CPU_H_

#include "SeetaNetBaseLayer.h"
#include <cmath>

template <class T>
class SeetaNetPowerCPU : public SeetaNetBaseLayer<T>
{
public:
	SeetaNetPowerCPU();
	~SeetaNetPowerCPU();

	int Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource);
	int Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map);

public:
	T m_scale;
	T m_shift;
	T m_power;
};

template <class T>
SeetaNetPowerCPU<T>::SeetaNetPowerCPU()
{
}

template <class T>
SeetaNetPowerCPU<T>::~SeetaNetPowerCPU()
{
}

template <class T>
int SeetaNetPowerCPU<T>::Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource)
{
	seeta::SeetaNet_PowerParameter *msg = (seeta::SeetaNet_PowerParameter *)inputparam.msg.get();
	m_scale = msg->scale;
	m_shift = msg->shift;
	m_power = msg->power;

	int index = inputparam.bottom_index[0];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = pNetResource->feature_vector_size[index];
	this->top_data_size.resize(1);
	this->top_data_size[0] = this->bottom_data_size[0];

	return 0;
}

template<typename T, typename FUNC>
static void power_each_do(T *arr, size_t size, FUNC func)
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
int SeetaNetPowerCPU<T>::Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	if (this->bottom_index[0] != this->top_index[0])
	{
		output_data_map[0]->data_shape = input_data_map[0]->data_shape;
		memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*output_data_map[0]->count());
	}

	auto type = (m_scale != T(1) ? 0x01 : 0x00) | (m_shift != T(0) ? 0x02 : 0x00) | (m_power != T(1) ? 0x04 : 0x00);

	switch (type)
	{
	default:
	case 0: // m_scale == 1, m_scale_out == 0, m_power == 1
		break;
	case 1: // m_scale != 1, m_scale_out == 0, m_power == 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val *= m_scale;
		});
		break;
	case 2: // m_scale == 1, m_scale_out != 0, m_power == 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val += m_shift;
		});
		break;
	case 3: // m_scale != 1, m_scale_out != 0, m_power == 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = val * m_scale + m_shift;
		});
		break;
	case 4: // m_scale == 1, m_scale_out == 0, m_power != 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::pow(val, m_power);
		});
		break;
	case 5: // m_scale != 1, m_scale_out == 0, m_power != 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::pow(val * m_scale, m_power);
		});
		break;
	case 6: // m_scale == 1, m_scale_out != 0, m_power != 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::pow(val + m_shift, m_power);
		});
		break;
	case 7: // m_scale != 1, m_scale_out != 0, m_power != 1
		power_each_do<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count(), [&](T & val)
		{
			val = std::pow(val * m_scale + m_shift, m_power);
		});
		break;
	}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	return 0;
}

#endif //!__POWERLAYER_H__
