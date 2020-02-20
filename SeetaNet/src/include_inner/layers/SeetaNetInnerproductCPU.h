#ifndef _SEETANET_INNERPRODUCT_CPU_H_
#define _SEETANET_INNERPRODUCT_CPU_H_

#include "SeetaNetBaseLayer.h"
#include "SeetaNetResource.h"
#include "SeetaNetCommonfuction.h"
#include <cfloat>

#include "SeetaNetMathCPU.h"

template <class T>
class SeetaNetInnerProductCPU : public SeetaNetBaseLayer<T>
{
public:
	SeetaNetInnerProductCPU();
	~SeetaNetInnerProductCPU();

	int Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource);
	int Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map);

public:
	std::vector<T> m_bias_value;

	SeetaNetBlobCpu<T> *m_p_inner_blob;
	T *m_innerproduct_width;

	int K_;
	int M_;
	int N_;

	bool transpose_;

private:
	SeetaNetResource<T> *m_p_seeta_net_resource;
};

template <class T>
SeetaNetInnerProductCPU<T>::~SeetaNetInnerProductCPU()
{
}

template <class T>
SeetaNetInnerProductCPU<T>::SeetaNetInnerProductCPU()
{
}

template <class T>
int SeetaNetInnerProductCPU<T>::Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource)
{
	this->m_layer_index = inputparam.layer_index;
	m_p_seeta_net_resource = pNetResource;

	auto bottom_length = inputparam.bottom_index.size();
	this->bottom_data_size.resize(bottom_length);

	for (size_t i = 0; i < bottom_length; i++)
	{
		int index = inputparam.bottom_index[i];
		this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
	}

	m_bias_value.clear();

	seeta::SeetaNet_InnerProductParameter *msg = (seeta::SeetaNet_InnerProductParameter *)inputparam.msg.get();

	for (int i = 0; i < msg->bias_param.data.size(); i++)
	{
		auto temp_biasvalue = msg->bias_param.data[i];

		if (temp_biasvalue < FLT_EPSILON && -temp_biasvalue < FLT_EPSILON)
			temp_biasvalue = 0;

		m_bias_value.push_back(temp_biasvalue);
	}

	int all_inner_counts = 1;
	std::vector<int> tmp_shape;
	tmp_shape.resize(msg->Inner_param.shape.dim.size());

	for (int i = 0; i < msg->Inner_param.shape.dim.size(); i++)
	{
		all_inner_counts *= msg->Inner_param.shape.dim[i];
		tmp_shape[i] = msg->Inner_param.shape.dim[i];
	}

	N_ = msg->Inner_param.shape.dim[0];
	K_ = msg->Inner_param.shape.dim[1];

	int index_key = this->m_layer_index;

	if (pNetResource->m_shared_param->param_map.find(index_key) != pNetResource->m_shared_param->param_map.end())
	{
	}
	else
	{
		SeetaNetBlobCpu<T> tmp_kernel_blob;

		pNetResource->m_shared_param->param_map.insert(std::pair<int, SeetaNetBlobCpu<T>>(index_key, tmp_kernel_blob));
		pNetResource->m_shared_param->param_map[index_key].Reshape(tmp_shape);

		T *temp_shared_kernel_value = pNetResource->m_shared_param->param_map[index_key].dataMemoryPtr();

		for (int i = 0; i < pNetResource->m_shared_param->param_map[index_key].count(); i++)
		{
			float tmp_float_value = msg->Inner_param.data[i];

			if (tmp_float_value < FLT_EPSILON && -tmp_float_value < FLT_EPSILON)
				tmp_float_value = 0;

			*temp_shared_kernel_value = tmp_float_value;
			temp_shared_kernel_value++;
		}
	}

	m_p_inner_blob = &(pNetResource->m_shared_param->param_map[index_key]);

	transpose_ = msg->transpose;

	this->top_data_size.resize(1);
	this->top_data_size[0].data_dim.resize(4);
	this->top_data_size[0].data_dim[0] = pNetResource->max_batch_size;
	this->top_data_size[0].data_dim[2] = 1;
	this->top_data_size[0].data_dim[3] = 1;
	this->top_data_size[0].data_dim[1] = msg->Inner_param.shape.dim[0];

	return 0;
}

template <class T>
int SeetaNetInnerProductCPU<T>::Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = this->top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = this->top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = this->top_data_size[0].data_dim[3];

	const T *weight = m_p_inner_blob->dataMemoryPtr();
	const T *bottom_data = input_data_map[0]->m_cpu.dataMemoryPtr();
	T *top_data = output_data_map[0]->m_cpu.dataMemoryPtr();

	M_ = input_data_map[0]->data_shape[0];

	bool fuse_bias = !m_bias_value.empty();

	//if ( fuse_bias )
	//{
	//    SetBiasBlob( output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value );
	//}

	seeta_cpu_gemm<T>(seeta::blas::NoTrans, transpose_ ? seeta::blas::NoTrans : seeta::blas::Trans,
		M_, N_, K_, (T)1.,
		bottom_data, weight, (T)0., top_data);

	if (fuse_bias)
	{
		AddBiasBlob(output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value);
	}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = this->top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = this->top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = this->top_data_size[0].data_dim[3];

	return 0;
}

#endif //!_INNERPRODUCT_H__
