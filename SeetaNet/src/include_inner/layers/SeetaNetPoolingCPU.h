#ifndef _SEETANET_POOLING_H_
#define _SEETANET_POOLING_H_

#include "SeetaNetBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <typename T>
class SeetaNetPoolingCpu : public SeetaNetBaseLayer<T>
{
public:
	SeetaNetPoolingCpu();
	~SeetaNetPoolingCpu();

	int Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource);
	int Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map);

	int MaxPooling(int number, SeetaNetBlobCpu<T> &inputdata, SeetaNetBlobCpu<T> &outputdata, int kernel_h, int kernel_w,
		int stride_h, int stride_w, int pad_h, int pad_w, std::vector<int> &shape_vector_in, std::vector<int> &shape_vector_out);

	int AveragePooling(int number, SeetaNetBlobCpu<T> &inputdata, SeetaNetBlobCpu<T> &outputdata, int kernel_h, int kernel_w,
		int stride_h, int stride_w, int pad_h, int pad_w, std::vector<int> &shape_vector_in, std::vector<int> &shape_vector_out);

private:
	void CaculatePoolSize(int input_height, int input_width, int &output_height, int &output_width);

	int m_kernel_h;
	int m_kernel_w;
	int m_stride_h;
	int m_stride_w;
	int m_pad_h;
	int m_pad_w;
	int m_dilation_h;
	int m_dilation_w;

	int m_pool_type;

	bool m_valid;

	int m_pooled_height_;
	int m_pooled_width_;

	std::string m_tf_padding;
	int m_tf_fake_padding_h = 0;
	int m_tf_fake_padding_w = 0;
};

size_t offset(std::vector<int> shape_, const int n, const int c = 0, const int h = 0,
	const int w = 0)
{
	return ((n * shape_[1] + c) * shape_[2] + h) * shape_[3] + w;
}

template <typename T>
void SeetaNetPoolingCpu<T>::CaculatePoolSize(int input_height, int input_width, int &output_height, int &output_width)
{
	if (m_tf_padding == "VALID")
	{
		output_height = int(ceil((input_height + 2 * m_pad_h - m_kernel_h + 1) / (float)m_stride_h));
		output_width = int(ceil((input_width + 2 * m_pad_w - m_kernel_w + 1) / (float)m_stride_w));
	}
	else
		if (m_tf_padding == "SAME")
		{
			output_height = int(ceil((input_height + 2 * m_pad_h) / (float)m_stride_h));
			output_width = int(ceil((input_width + 2 * m_pad_w) / (float)m_stride_w));

			// no feak padding when pooling
			m_tf_fake_padding_h = 0;
			m_tf_fake_padding_w = 0;
		}
		else
			if (m_valid)
			{
				output_height = int(floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1));
				output_width = int(floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1));
			}
			else
			{
				output_height = int(ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1));
				output_width = int(ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1));
			}
}

template <typename T>
int SeetaNetPoolingCpu<T>::Init(seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource)
{
	m_dilation_h = 1;
	m_dilation_w = 1;

	seeta::SeetaNet_PoolingParameter *msg = (seeta::SeetaNet_PoolingParameter *)inputparam.msg.get();
	m_pool_type = msg->pool;

	m_kernel_h = msg->kernel_height;
	m_kernel_w = msg->kernel_width;
	m_stride_h = msg->stride_height;
	m_stride_w = msg->stride_width;
	m_pad_h = msg->pad_height;
	m_pad_w = msg->pad_width;

	m_valid = false;

	if (msg->has_valid())
	{
		m_valid = msg->valid;
	}

	if (msg->has_tf_padding())
	{
		m_tf_padding = msg->tf_padding;
	}

	int bottom_index = inputparam.bottom_index[0];
	SeetaNetDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = bottom_size;

	if (msg->global_pooling)
	{
		m_kernel_h = this->bottom_data_size[0].data_dim[2];
		m_kernel_w = this->bottom_data_size[0].data_dim[3];
		m_pad_h = 0;
		m_pad_w = 0;
	}

	CaculatePoolSize(this->bottom_data_size[0].data_dim[2], this->bottom_data_size[0].data_dim[3], m_pooled_height_, m_pooled_width_);

	this->top_data_size.resize(1);
	this->top_data_size[0].data_dim.resize(4);
	this->top_data_size[0].data_dim[2] = m_pooled_height_;
	this->top_data_size[0].data_dim[3] = m_pooled_width_;
	this->top_data_size[0].data_dim[1] = this->bottom_data_size[0].data_dim[1];
	this->top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];

	return 0;
}

template <typename T>
int SeetaNetPoolingCpu<T>::Process(std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	CaculatePoolSize(input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3], m_pooled_height_, m_pooled_width_);

	std::vector<int> shape_vector_in;
	shape_vector_in.push_back(input_data_map[0]->data_shape[0]);
	shape_vector_in.push_back(input_data_map[0]->data_shape[1]);
	shape_vector_in.push_back(input_data_map[0]->data_shape[2]);
	shape_vector_in.push_back(input_data_map[0]->data_shape[3]);

	std::vector<int> shape_vector_out;
	shape_vector_out.push_back(input_data_map[0]->data_shape[0]);
	shape_vector_out.push_back(input_data_map[0]->data_shape[1]);
	shape_vector_out.push_back(m_pooled_height_);
	shape_vector_out.push_back(m_pooled_width_);

	if (seeta::SeetaNet_PoolingParameter::MAX == m_pool_type)
	{
		MaxPooling(input_data_map[0]->data_shape[0], input_data_map[0]->m_cpu, output_data_map[0]->m_cpu,
			m_kernel_h, m_kernel_w, m_stride_h, m_stride_w, m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w, shape_vector_in, shape_vector_out);

	}
	else
		if (seeta::SeetaNet_PoolingParameter::AVE == m_pool_type)
		{
			AveragePooling(input_data_map[0]->data_shape[0], input_data_map[0]->m_cpu, output_data_map[0]->m_cpu,
				m_kernel_h, m_kernel_w, m_stride_h, m_stride_w, m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w, shape_vector_in, shape_vector_out);

		}
		else
		{
		}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = shape_vector_out[1];
	output_data_map[0]->data_shape[2] = shape_vector_out[2];
	output_data_map[0]->data_shape[3] = shape_vector_out[3];

	return 0;
}

template <typename T>
SeetaNetPoolingCpu<T>::SeetaNetPoolingCpu()
{
};

template <typename T>
SeetaNetPoolingCpu<T>::~SeetaNetPoolingCpu()
{
};

template <typename T>
int SeetaNetPoolingCpu<T>::MaxPooling(int number, SeetaNetBlobCpu<T> &inputdata, SeetaNetBlobCpu<T> &outputdata,
	int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w
	, std::vector<int> &shape_vector_in, std::vector<int> &shape_vector_out)
{
	// 预先计算特征输出向量的维度
	const T *bottom_data = inputdata.dataMemoryPtr();
	T *top_data = outputdata.dataMemoryPtr();

	int height_ = shape_vector_in[2];
	int width_ = shape_vector_in[3];

	auto input_offset = offset(shape_vector_in, 0, 1);
	auto output_offset = offset(shape_vector_out, 0, 1);

	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

	if (gun == nullptr || gun->size() <= 1)
	{
		for (int n = 0; n < number; ++n)
		{
			for (int c = 0; c < inputdata.shape()[1]; ++c)
			{
				for (int ph = 0; ph < m_pooled_height_; ++ph)
				{
					for (int pw = 0; pw < m_pooled_width_; ++pw)
					{
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend_tmp = hstart + kernel_h;
						int wend_tmp = wstart + kernel_w;
						int hend = std::min(hend_tmp, height_);
						int wend = std::min(wend_tmp, width_);

						hstart = std::max(hstart, 0);
						wstart = std::max(wstart, 0);

						const int pool_index = ph * m_pooled_width_ + pw;
						T max_value_ = bottom_data[hstart * width_ + wstart];

						for (int h = hstart; h < hend; ++h)
						{
							for (int w = wstart; w < wend; ++w)
							{
								const int index = h * width_ + w;

								if (bottom_data[index] > max_value_)
								{
									max_value_ = bottom_data[index];
								}
							}
						}

						top_data[pool_index] = max_value_;
					}
				}

				// compute offset
				bottom_data += input_offset;
				top_data += output_offset;
			}
		}
	}
	else
	{
		auto input_batch_offset = inputdata.shape()[1] * input_offset;
		auto output_batch_offset = inputdata.shape()[1] * output_offset;

		for (int n = 0; n < number; ++n)
		{
			auto batch_bottom_data = bottom_data + n * input_batch_offset;
			auto batch_top_data = top_data + n * output_batch_offset;
			auto bins = orz::split_bins(0, inputdata.shape()[1], int(gun->size()));

			for (auto &bin : bins)
			{
				gun->fire([&, batch_bottom_data, batch_top_data, bin](int)
				{
					auto local_bottom_data = batch_bottom_data + bin.first * input_offset;
					auto local_top_data = batch_top_data + bin.first * output_offset;

					for (int c = bin.first; c < bin.second; ++c)
					{
						for (int ph = 0; ph < m_pooled_height_; ++ph)
						{
							for (int pw = 0; pw < m_pooled_width_; ++pw)
							{
								int hstart = ph * stride_h - pad_h;
								int wstart = pw * stride_w - pad_w;
								int hend_tmp = hstart + kernel_h;
								int wend_tmp = wstart + kernel_w;
								int hend = std::min(hend_tmp, height_);
								int wend = std::min(wend_tmp, width_);

								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);

								const int pool_index = ph * m_pooled_width_ + pw;
								T max_value_ = local_bottom_data[hstart * width_ + wstart];

								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int index = h * width_ + w;

										if (local_bottom_data[index] > max_value_)
										{
											max_value_ = local_bottom_data[index];
										}
									}
								}

								local_top_data[pool_index] = max_value_;
							}
						}

						// compute offset
						local_bottom_data += input_offset;
						local_top_data += output_offset;
					}
				});
			}
		}

		gun->join();
	}

	return 0;
};

template <typename T>
int SeetaNetPoolingCpu<T>::AveragePooling(int number, SeetaNetBlobCpu<T> &inputdata, SeetaNetBlobCpu<T> &outputdata,
	int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w
	, std::vector<int> &shape_vector_in, std::vector<int> &shape_vector_out)
{
	const T *bottom_data = inputdata.dataMemoryPtr();
	T *top_data = outputdata.dataMemoryPtr();

	int height_ = shape_vector_in[2];
	int width_ = shape_vector_in[3];

	auto input_offset = offset(shape_vector_in, 0, 1);
	auto output_offset = offset(shape_vector_out, 0, 1);

	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

	if (gun == nullptr || gun->size() <= 1)
	{
		for (int n = 0; n < number; ++n)
		{
			for (int c = 0; c < inputdata.shape()[1]; ++c)
			{
				for (int ph = 0; ph < m_pooled_height_; ++ph)
				{
					for (int pw = 0; pw < m_pooled_width_; ++pw)
					{
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend_tmp = hstart + kernel_h;
						int wend_tmp = wstart + kernel_w;
						int hend = std::min(hend_tmp, height_);
						int wend = std::min(wend_tmp, width_);

						hstart = std::max(hstart, 0);
						wstart = std::max(wstart, 0);

						const int pool_index = ph * m_pooled_width_ + pw;
						int current_count = 0;
						T sum_value = 0.0;

						for (int h = hstart; h < hend; ++h)
						{
							for (int w = wstart; w < wend; ++w)
							{
								const int index = h * width_ + w;
								sum_value += bottom_data[index];
								current_count += 1;
							}
						}

						top_data[pool_index] = sum_value / current_count;
					}
				}

				// compute offset
				bottom_data += input_offset;
				top_data += output_offset;
			}
		}
	}
	else
	{
		auto input_batch_offset = inputdata.shape()[1] * input_offset;
		auto output_batch_offset = inputdata.shape()[1] * output_offset;

		for (int n = 0; n < number; ++n)
		{
			auto batch_bottom_data = bottom_data + n * input_batch_offset;
			auto batch_top_data = top_data + n * output_batch_offset;
			auto bins = orz::split_bins(0, inputdata.shape()[1], int(gun->size()));

			for (auto &bin : bins)
			{
				gun->fire([&, batch_bottom_data, batch_top_data, bin](int)
				{
					auto local_bottom_data = batch_bottom_data + bin.first * input_offset;
					auto local_top_data = batch_top_data + bin.first * output_offset;

					for (int c = bin.first; c < bin.second; ++c)
					{
						for (int ph = 0; ph < m_pooled_height_; ++ph)
						{
							for (int pw = 0; pw < m_pooled_width_; ++pw)
							{
								int hstart = ph * stride_h - pad_h;
								int wstart = pw * stride_w - pad_w;
								int hend_tmp = hstart + kernel_h;
								int wend_tmp = wstart + kernel_w;
								int hend = std::min(hend_tmp, height_);
								int wend = std::min(wend_tmp, width_);

								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);

								const int pool_index = ph * m_pooled_width_ + pw;
								int current_count = 0;
								T sum_value = 0.0;

								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int index = h * width_ + w;
										sum_value += local_bottom_data[index];
										current_count += 1;
									}
								}

								local_top_data[pool_index] = sum_value / current_count;
							}
						}

						// compute offset
						local_bottom_data += input_offset;
						local_top_data += output_offset;
					}
				});
			}
		}

		gun->join();
	}

	return 0;
};

#endif
