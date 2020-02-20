#ifndef _SEETANET_BLOB_CPU_H_
#define _SEETANET_BLOB_CPU_H_

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <climits>
#include <string.h>
#include <memory>
#include <algorithm>

#define FREE_DATA

template<class Dtype>
class SeetaNetBlobCpu {
public:
	SeetaNetBlobCpu();
	SeetaNetBlobCpu(const SeetaNetBlobCpu<Dtype> &m);
	~SeetaNetBlobCpu();

	size_t offset(const int n, const int c = 0, const int h = 0, const int w = 0) const
	{
		return ((n * shape_[1] + c) * shape_[2] + h) * shape_[3] + w;
	}

	inline int num_axes() const
	{
		return int(shape_.size());
	}

	inline int CanonicalAxisIndex(int axis_index) const
	{
		if (axis_index <= -num_axes())
		{
			std::cout << "axis " << axis_index << " out of range for " << num_axes() << "-D Blob with shape " << shape_string();
		}

		if (axis_index > num_axes())
		{
			std::cout << "axis " << axis_index << " out of range for " << num_axes() << "-D Blob with shape " << shape_string();
		}

		if (axis_index < 0)
		{
			return axis_index + num_axes();
		}

		return axis_index;
	}

	inline int offset(const std::vector<int> &indices) const
	{
		int offset = 0;

		for (int i = 0; i < num_axes(); ++i)
		{
			offset *= shape_[i];

			if (indices.size() > i)
			{
				if (indices[i] < 0)
				{
					std::cout << "blob offset input error" << std::endl;
				}

				if (indices[i] > shape_[i])
				{
					std::cout << "blob offset input error" << std::endl;
				}

				offset += indices[i];
			}
		}

		return offset;
	}

	inline Dtype &data_at(const std::vector<int> &index)
	{
		return m_data.get()[offset(index)];
	}

	inline std::string shape_string() const
	{
		std::ostringstream stream;

		for (int i = 0; i < shape_.size(); ++i)
		{
			stream << shape_[i] << " ";
		}

		stream << "(" << count_ << ")";
		return stream.str();
	}

	int Reshape(const std::vector<int> &shape);
	int ReshapeJustShape(const std::vector<int> &shape);

	Dtype *data()
	{
		return m_data.get();
	}

	const Dtype *data() const
	{
		return m_data.get();
	}

	inline Dtype *dataMemoryPtr();

	template<typename functor>
	inline const SeetaNetBlobCpu<Dtype> &for_each(functor F);
	template<typename functor>
	inline const SeetaNetBlobCpu<Dtype> &for_each(functor F, size_t limit);

	inline SeetaNetBlobCpu<Dtype> operator=(SeetaNetBlobCpu<Dtype>const &m);

	// for outer memory contorl
	int count() const
	{
		return count_;
	}

#ifdef FREE_DATA
	void dispose()
	{
		m_data.reset();
	}

	void set_raw_data(const std::shared_ptr<Dtype> &data)
	{
		m_data = data;
	}
#endif

	// for outer shape get
	const std::vector<int> &shape() const
	{
		return shape_;
	}

private:
	int count_;
	int capacity;

	std::vector<int> shape_;

private:
	std::shared_ptr<Dtype> m_data;
};

template<typename Dtype>
SeetaNetBlobCpu<Dtype>::SeetaNetBlobCpu(const SeetaNetBlobCpu<Dtype> &m)
{
	count_ = m.count_;
	shape_ = m.shape_;
	capacity = 0;

	int tmp_counts = 1;

	for (int i = 0; i < shape_.size(); ++i)
	{
		if (shape_[i] <= 0)
		{
			std::cout << "blob shape error!" << std::endl;
		}

		if (tmp_counts != 0)
		{
			if (shape_[i] >= (INT_MAX / tmp_counts))
			{
				std::cout << "blob size exceeds INT_MAX";
				break;
			}
		}

		tmp_counts *= shape_[i];
	}

	if ((count_ != 0) && (tmp_counts != count_))
	{
		std::cout << "error!";
	}

	if ((nullptr != m_data) && (m_data == m.m_data))
	{
		std::cout << "error!";
	}
	else
	{
		if (0 != count_)
		{
			m_data.reset(new Dtype[count_], std::default_delete<Dtype[]>());
			memcpy(m_data.get(), m.m_data.get(), count_ * sizeof(Dtype));

			capacity = count_;
		}
	}

	if (nullptr == m.m_data)
	{
		m_data = nullptr;
	}
}

template<class Dtype>
SeetaNetBlobCpu<Dtype> SeetaNetBlobCpu<Dtype>::operator=(SeetaNetBlobCpu<Dtype>const &m)
{
	count_ = m.count_;
	shape_ = m.shape_;

	if (m_data == m.m_data)
	{

	}
	else
	{
		m_data.reset(new Dtype[count_], std::default_delete<Dtype[]>());
		memcpy(m_data.get(), m.m_data.get(), count_ * sizeof(Dtype));

		capacity = count_;
	}

	return *this;
}

template<typename Dtype>
template<typename functor>
inline const SeetaNetBlobCpu<Dtype> &SeetaNetBlobCpu<Dtype>::for_each(functor F)
{
	for (int index = 0; index < count_; index++)
	{
		F(m_data.get()[index]);
	}

	return *this;
}

template<typename Dtype>
template<typename functor>
inline const SeetaNetBlobCpu<Dtype> &SeetaNetBlobCpu<Dtype>::for_each(functor F, size_t limit)
{
	auto limit_count = std::min<int>(limit, count_);

	for (int index = 0; index < limit_count; index++)
	{
		F(m_data.get()[index]);
	}

	return *this;
}

template<class Dtype>
int SeetaNetBlobCpu<Dtype>::Reshape(const std::vector<int> &shape)
{
	int tmp_counts = 1;

	shape_.resize(shape.size());

	for (int i = 0; i < shape.size(); ++i)
	{

		if (shape[i] <= 0)
		{
			std::cout << "blob reshape error!" << "this is a test" << std::endl;
		}

		if (tmp_counts != 0)
		{
			if (shape[i] >= (INT_MAX / tmp_counts))
			{
				std::cout << "blob size exceeds INT_MAX" << " this is a test" << std::endl;
				return -1;
			}
		}

		tmp_counts *= shape[i];
		shape_[i] = shape[i];

	}

	if (capacity < tmp_counts || m_data == nullptr)
	{
		m_data.reset(new Dtype[tmp_counts], std::default_delete<Dtype[]>());
		capacity = tmp_counts;
	}

	count_ = tmp_counts;

	return 0;
}

template <class Dtype>
int SeetaNetBlobCpu<Dtype>::ReshapeJustShape(const std::vector<int> &shape)
{
	int tmp_counts = 1;

	shape_.resize(shape.size());

	for (int i = 0; i < shape.size(); ++i)
	{
		if (shape[i] <= 0)
		{

			std::cout << "blob reshape error!" << " jst test---" << std::endl;
		}

		if (tmp_counts != 0)
		{
			if (shape[i] >= (INT_MAX / tmp_counts))
			{
				std::cout << "blob size exceeds INT_MAX" << " just test----" << std::endl;;
				return -1;
			}
		}

		tmp_counts *= shape[i];
		shape_[i] = shape[i];
	}

#ifndef FREE_DATA
	if (capacity < tmp_counts || m_data == nullptr)
	{
		m_data.reset(new Dtype[tmp_counts], std::default_delete<Dtype[]>());
		capacity = tmp_counts;
	}
#endif

	count_ = tmp_counts;

	return 0;
}

template<class Dtype>
SeetaNetBlobCpu<Dtype>::SeetaNetBlobCpu()
{
	m_data.reset();

	count_ = 0;
	capacity = 0;
};


template<class Dtype>
SeetaNetBlobCpu<Dtype>::~SeetaNetBlobCpu()
{
	m_data.reset();
}

template<class Dtype>
inline int AddBiasBlob(SeetaNetBlobCpu<Dtype> &input_output_data, std::vector<int> &shape_vector, const std::vector<Dtype> &bias)
{
	Dtype *pstart = input_output_data.data();

	for (int n = 0; n < shape_vector[0]; n++)
	{
		for (int i = 0; i < shape_vector[1]; i++)
		{
			for (int j = 0; j < shape_vector[2] * shape_vector[3]; j++)
			{
				*pstart += bias[i];
				pstart++;
			}
		}
	}

	return 0;
}

template<class Dtype>
inline int AddBiasBlob(SeetaNetBlobCpu<Dtype> &input_output_data, const std::vector<Dtype> &bias)
{
	return AddBiasBlob(input_output_data, input_output_data.shape(), bias);
}

template<class Dtype>
inline int SetBiasBlob(SeetaNetBlobCpu<Dtype> &input_output_data, std::vector<int> &shape_vector, const std::vector<Dtype> &bias)
{
	Dtype *pstart = input_output_data.data();

	for (int n = 0; n < shape_vector[0]; n++)
	{
		for (int i = 0; i < shape_vector[1]; i++)
		{
			for (int j = 0; j < shape_vector[2] * shape_vector[3]; j++)
			{
				*pstart = bias[i];
				pstart++;
			}
		}
	}

	return 0;
}

template<class Dtype>
inline int SetBiasBlob(SeetaNetBlobCpu<Dtype> &input_output_data, const std::vector<Dtype> &bias)
{
	return SetBiasBlob(input_output_data, input_output_data.shape(), bias);
}

template<typename Dtype>
Dtype *SeetaNetBlobCpu<Dtype>::dataMemoryPtr()
{
	return m_data.get();
}

#endif
