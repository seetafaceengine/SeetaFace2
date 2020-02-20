#ifndef _SEETANET_RESHAPE_CPU_H_
#define _SEETANET_RESHAPE_CPU_H_

#include "SeetaNetBaseLayer.h"

template <class T>
class SeetaNetReshapeCPU : public SeetaNetBaseLayer<T>
{
public:
	SeetaNetReshapeCPU();
	~SeetaNetReshapeCPU();

	int Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource);
	int Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map);

public:
	std::vector<int> m_shape;
	std::vector<int> m_permute;
};

template <class T>
SeetaNetReshapeCPU<T>::SeetaNetReshapeCPU()
{
}

template <class T>
SeetaNetReshapeCPU<T>::~SeetaNetReshapeCPU()
{
}

template <class T>
int SeetaNetReshapeCPU<T>::Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource)
{
	int bottom_index = inputparam.bottom_index[0];
	SeetaNetDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];

	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = bottom_size;

	seeta::SeetaNet_ReshapeLayer *msg = (seeta::SeetaNet_ReshapeLayer *)inputparam.msg.get();
	m_shape.resize(msg->shape.size());

	for (size_t i = 0; i < m_shape.size(); ++i)
	{
		m_shape[i] = msg->shape[i];
	}

	assert(m_shape.size() == 4);
	assert(m_shape[0] == 1);

	m_permute.resize(msg->permute.size());

	for (size_t i = 0; i < m_permute.size(); ++i)
	{
		m_permute[i] = msg->permute[i];
	}

	assert(m_permute.empty() || m_permute.size() == 4);

	this->top_data_size.resize(1);
	this->top_data_size[0].data_dim.resize(4);
	this->top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];
	this->top_data_size[0].data_dim[1] = m_shape[1];
	this->top_data_size[0].data_dim[2] = m_shape[2];
	this->top_data_size[0].data_dim[3] = m_shape[3];

	return 0;
}

template <typename T>
static void gun_ermute_kernel(orz::Shotgun *gun, T *input_data, T *output_data,
	int input_number, int input_channels, int input_height, int input_width,
	int output_number, int output_channels, int output_height, int output_width,
	int input_size, int input_number_step, int input_channels_step, int input_height_step,
	int output_size, int output_number_step, int output_channels_step, int output_height_step,
	int dim1, int dim2, int dim3, int dim4)
{
	int transdim[4] = { dim1, dim2, dim3, dim4 };   // 0, 2, 3, 1

	auto bins = orz::split_bins(0, input_size, int(gun->size()));

	for (auto &bin : bins)
	{
		gun->fire([&, bin](int)
		{
			for (auto t = bin.first; t < bin.second; ++t)
			{
				auto index = t;

				int at_input_i = index;

				int n = index / input_number_step;
				index %= input_number_step;

				int c = index / input_channels_step;
				index %= input_channels_step;

				int h = index / input_height_step;
				index %= input_height_step;

				int w = index;

				int encode_in[4] = { n, c, h, w };
				int out_index[4] = { encode_in[transdim[0]], encode_in[transdim[1]], encode_in[transdim[2]], encode_in[transdim[3]] };

				int at_output_i = out_index[0] * output_number_step + out_index[1] * output_channels_step + out_index[2] * output_height_step + out_index[3];

				output_data[at_output_i] = input_data[at_input_i];
			}
		});
	}

	gun->join();
}

template <typename T>
void permute(const T *in_data, T *out_data,
	int shape1, int shape2, int shape3, int shape4,
	int dim1, int dim2, int dim3, int dim4)
{
	std::vector<int> m_shape = { shape1, shape2, shape3, shape4 };
	std::vector<int> dim(4), redim(4), idx(4, 0);

	dim[0] = dim1;
	redim[dim[0]] = 0;
	dim[1] = dim2;
	redim[dim[1]] = 1;
	dim[2] = dim3;
	redim[dim[2]] = 2;
	dim[3] = dim4;
	redim[dim[3]] = 3;

	std::vector<int> new_dims(4);

	for (int i = 0; i < 4; ++i)
		new_dims[i] = m_shape[dim[i]];

	T *tmp = out_data;
	const T *dat = in_data;
	int cnt = 0;

	for (idx[0] = 0; idx[0] < m_shape[dim[0]]; ++idx[0])
	{
		for (idx[1] = 0; idx[1] < m_shape[dim[1]]; ++idx[1])
		{
			for (idx[2] = 0; idx[2] < m_shape[dim[2]]; ++idx[2])
			{
				for (idx[3] = 0; idx[3] < m_shape[dim[3]]; ++idx[3])
				{
					int offset = ((idx[redim[0]] * m_shape[1] + idx[redim[1]]) * m_shape[2] + idx[redim[2]]) * m_shape[3] + idx[redim[3]];
					tmp[cnt] = dat[offset];
					cnt++;
				}
			}
		}
	}
}

template <class T>
int SeetaNetReshapeCPU<T>::Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map)
{
	input_data_map[0]->TransFormDataIn();
	int all_size = 1;

	for (int i = 0; i < 4; i++)
	{
		all_size *= input_data_map[0]->data_shape[i];
	}

	if (!m_permute.empty())
	{
		// DO PERMUTE
		auto gun = orz::ctx::lite::get<orz::Shotgun>();

		if (gun != nullptr && gun->size() > 1)
		{
			// write output
			int input_number = input_data_map[0]->data_shape[0];
			int input_channels = input_data_map[0]->data_shape[1];
			int input_height = input_data_map[0]->data_shape[2];
			int input_width = input_data_map[0]->data_shape[3];

			int output_number = input_data_map[0]->data_shape[m_permute[0]];
			int output_channels = input_data_map[0]->data_shape[m_permute[1]];
			int output_height = input_data_map[0]->data_shape[m_permute[2]];
			int output_width = input_data_map[0]->data_shape[m_permute[3]];

			int input_size = input_number * input_channels * input_height * input_width;
			int input_number_step = input_channels * input_height * input_width;
			int input_channels_step = input_height * input_width;
			int input_height_step = input_width;

			int output_size = output_number * output_channels * output_height * output_width;
			int output_number_step = output_channels * output_height * output_width;
			int output_channels_step = output_height * output_width;
			int output_height_step = output_width;

			T *input_data = input_data_map[0]->m_cpu.dataMemoryPtr();
			T *output_data = output_data_map[0]->m_cpu.dataMemoryPtr();

			gun_ermute_kernel<T>(gun,
				input_data, output_data,
				input_number, input_channels, input_height, input_width,
				output_number, output_channels, output_height, output_width,
				input_size, input_number_step, input_channels_step, input_height_step,
				output_size, output_number_step, output_channels_step, output_height_step,
				m_permute[0], m_permute[1], m_permute[2], m_permute[3]);
		}
		else
		{
			permute<T>(input_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->m_cpu.dataMemoryPtr(),
				input_data_map[0]->data_shape[0], input_data_map[0]->data_shape[1], input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3],
				m_permute[0], m_permute[1], m_permute[2], m_permute[3]);
		}
	}
	else
	{
		if (this->bottom_index[0] != this->top_index[0])
		{
			memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*all_size);
		}
	}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

	output_data_map[0]->data_shape.resize(4);
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = m_shape[1];
	output_data_map[0]->data_shape[2] = m_shape[2];
	output_data_map[0]->data_shape[3] = m_shape[3];

	return 0;
}

#endif
