#pragma once

#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>
#include <string>
#include <sstream>

namespace seeta
{
	template <typename T>
	static void CopyData(T *to, const T *from, size_t _count)
	{
#if _MSC_VER >= 1600
		memcpy_s(to, _count * sizeof(T), from, _count * sizeof(T));
#else
		memcpy(to, from, _count * sizeof(T));
#endif
	}

	template <typename T>
	struct Blob
	{
	public:
		using dtype = T;

		Blob(const std::vector<int> &dims)
		{
			reshape(dims);
		}

		Blob(int dim1, int dim2, int dim3, int dim4)
		{
			reshape(dim1, dim2, dim3, dim4);
		}

		Blob(int dim2, int dim3, int dim4)
		{
			reshape(dim2, dim3, dim4);
		}

		Blob(int dim3, int dim4)
		{
			reshape(dim3, dim4);
		}

		Blob(int dim4)
		{
			reshape(dim4);
		}

		Blob(const T *data, const std::vector<int> &dims)
		{
			reshape(dims);
			copy_from(data);
		}

		Blob(const T *data, int dim1, int dim2, int dim3, int dim4)
		{
			reshape(dim1, dim2, dim3, dim4);
			copy_from(data);
		}

		Blob(const T *data, int dim2, int dim3, int dim4)
		{
			reshape(dim2, dim3, dim4);
			copy_from(data);
		}

		Blob(const T *data, int dim3, int dim4)
		{
			reshape(dim3, dim4);
			copy_from(data);
		}

		Blob(const T *data, int dim4)
		{
			reshape(dim4);
			copy_from(data);
		}

		Blob()
		{
			reshape(1);
		}

		template <typename U>
		Blob(const Blob<U> &other)
		{
			this->operator=(other);
		}

		template <typename U>
		const Blob &operator=(const Blob<U> &other)
		{
			reshape(other.shape());

			for (int i = 0; i < count(); ++i)
				data()[i] = static_cast<T>(other.data()[i]);

			return *this;
		}

		T *data()
		{
			return m_data.get();
		}

		const T *data() const
		{
			return const_cast<Blob *>(this)->data();
		}

		const std::vector<int> &shape() const
		{
			return m_shape;
		}

		int shape(int axe) const
		{
			if (axe < 0 || axe >= int(m_shape.size()))
				return 1;

			return m_shape[axe];
		}

		void reshape(int dim1, int dim2, int dim3, int dim4)
		{
			std::vector<int> dims = { dim1, dim2, dim3, dim4 };
			reshape(dims);
		}

		void reshape(int dim2, int dim3, int dim4)
		{
			std::vector<int> dims = { dim2, dim3, dim4 };
			reshape(dims);
		}

		void reshape(int dim3, int dim4)
		{
			std::vector<int> dims = { dim3, dim4 };
			reshape(dims);
		}

		void reshape(int dim4)
		{
			std::vector<int> dims = { dim4 };
			reshape(dims);
		}

		void reshape(const std::vector<int> &dims)
		{
			std::vector<int> shape;

			if (dims.size() >= 4)
			{
				shape = std::vector<int>(dims.begin(), dims.begin() + 4);
			}
			else
			{
				shape = dims;
				while (shape.size() < 4) shape.insert(shape.begin(), 1);
			}

			int need_size = count(shape);
			int have_size = count(m_capacity);

			if (need_size > have_size)
			{
				std::shared_ptr<T> need_data(new T[need_size], std::default_delete<T[]>());
				CopyData(need_data.get(), m_data.get(), std::min(need_size, have_size));
				m_data = need_data;
				m_capacity = shape;
			}

			m_shape = shape;
		}

		Blob permute(int dim1, int dim2, int dim3, int dim4) const
		{
			std::vector<int> dim(4), redim(4), idx(4);
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

			Blob result(new_dims);

			float *tmp = result.data();
			float *dat = m_data.get();
			int cnt = 0;

			for (idx[0] = 0; idx[0] < m_shape[dim[0]]; ++idx[0])
			{
				for (idx[1] = 0; idx[1] < m_shape[dim[1]]; ++idx[1])
				{
					for (idx[2] = 0; idx[2] < m_shape[dim[2]]; ++idx[2])
					{
						for (idx[3] = 0; idx[3] < m_shape[dim[3]]; ++idx[3])
						{
							tmp[cnt] = dat[offset(idx[redim[0]], idx[redim[1]], idx[redim[2]], idx[redim[3]])];
							cnt++;
						}
					}
				}
			}

			return std::move(result);
		}

		Blob clone() const
		{
			Blob result(this->shape());
			this->copy_to(result.data());
			return std::move(result);
		}

		void copy_from(const T *data, int size = -1)
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min(copy_size, size);
			CopyData(m_data.get(), data, copy_size);
		}

		void copy_to(T *data, int size = -1) const
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min(copy_size, size);
			CopyData(data, m_data.get(), copy_size);
		}

		int count() const
		{
			return count(m_shape);
		}

		static int count(const std::vector<int> &dims)
		{
			if (dims.empty())
				return 0;

			int prod = 1;

			for (int dim : dims)
				prod *= dim;

			return prod;
		}

		T &operator[](int dim4)
		{
			return data()[offset(dim4)];
		}

		const T &operator[](int dim4) const
		{
			return const_cast<Blob *>(this)->operator[](dim4);
		}

		T &data(int dim4)
		{
			return data()[offset(dim4)];
		}

		const T &data(int dim4) const
		{
			return const_cast<Blob *>(this)->data(dim4);
		}

		T &data(int dim3, int dim4)
		{
			return data()[offset(dim3, dim4)];
		}

		const T &data(int dim3, int dim4) const
		{
			return const_cast<Blob *>(this)->data(dim3, dim4);
		}

		T &data(int dim2, int dim3, int dim4)
		{
			return data()[offset(dim2, dim3, dim4)];
		}

		const T &data(int dim2, int dim3, int dim4) const
		{
			return const_cast<Blob *>(this)->data(dim2, dim3, dim4);
		}

		T &data(int dim1, int dim2, int dim3, int dim4)
		{
			return data()[offset(dim1, dim2, dim3, dim4)];
		}

		const T &data(int dim1, int dim2, int dim3, int dim4) const
		{
			return const_cast<Blob *>(this)->data(dim1, dim2, dim3, dim4);
		}

		int offset(int dim1, int dim2, int dim3, int dim4) const
		{
			return ((dim1 * m_shape[1] + dim2) * m_shape[2] + dim3) * m_shape[3] + dim4;
		}

		int offset(int dim2, int dim3, int dim4) const
		{
			return (dim2 * m_shape[2] + dim3) * m_shape[3] + dim4;
		}

		int offset(int dim3, int dim4) const
		{
			return dim3 * m_shape[3] + dim4;
		}

		int offset(int dim4) const
		{
			return dim4;
		}

		template <typename FUNC>
		const Blob &for_each(FUNC func)
		{
			for (int i = 0; i < this->count(); ++i)
				func(this->data()[i]);

			return *this;
		}

		Blob operator-()
		{
			Blob<T> result(this->shape());

			for (int i = 0; i < this->count(); ++i)
				result.data()[i] = -this->data()[i];

			return result;
		}

		const Blob &operator*=(const T &value)
		{
			for (int i = 0; i < count(); ++i)
				data()[i] *= value;

			return *this;
		}

		friend Blob operator*(const Blob &blob, const T &value)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = blob.data()[i] * value;

			return result;
		}

		friend Blob operator*(const T &value, const Blob &blob)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = value * blob.data()[i];

			return result;
		}

		const Blob &operator/=(const T &value)
		{
			for (int i = 0; i < count(); ++i)
				data()[i] /= value;

			return *this;
		}

		friend Blob operator/(const Blob &blob, const T &value)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = blob.data()[i] / value;

			return result;
		}

		friend Blob operator/(const T &value, const Blob &blob)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = value / blob.data()[i];

			return result;
		}

		const Blob &operator+=(const T &value)
		{
			for (int i = 0; i < count(); ++i)
				data()[i] += value;

			return *this;
		}

		friend Blob operator+(const Blob &blob, const T &value)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = blob.data()[i] + value;

			return result;
		}

		friend Blob operator+(const T &value, const Blob &blob)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = value + blob.data()[i];

			return result;
		}

		const Blob &operator-=(const T &value)
		{
			for (int i = 0; i < count(); ++i)
				data()[i] -= value;

			return *this;
		}

		friend Blob operator-(const Blob &blob, const T &value)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = blob.data()[i] - value;

			return result;
		}

		friend Blob operator-(const T &value, const Blob &blob)
		{
			Blob<T> result(blob.shape());

			for (int i = 0; i < blob.count(); ++i)
				result.data()[i] = value - blob.data()[i];

			return result;
		}

		template <typename T1>
		const Blob &operator+=(const Blob<T1> &blob)
		{
			for (int i = 0; i < 4; ++i)
			{
				if (shape(i) != blob.shape(i))
					throw std::logic_error("Cant not eltwise do blob shape mismatch");
			}

			for (int i = 0; i < count(); ++i)
				data()[i] += blob.data()[i];

			return *this;
		}

		friend Blob operator+(const Blob &blob1, const Blob &blob2)
		{
			Blob result = blob1.clone();
			result += blob2;

			return std::move(result);
		}

		Blob dimshffle(int axis, const std::vector<int> dims)
		{
			if (axis < 0 || axis > 3)
				throw std::logic_error("Can not swap blob at axis " + std::to_string(axis));

			if (dims.empty())
				throw std::logic_error("Can not shuffle to empty dim");

			for (auto dim : dims)
			{
				if (dim < 0 || dim >= this->shape(axis))
					throw std::logic_error("Can not index dim(" + std::to_string(dim) + ") at axis(" + std::to_string(axis) + ")");
			}

			auto shape = this->shape();
			shape[axis] = static_cast<int>(dims.size());
			Blob<T> result(shape);

			int copy_times = 1;

			for (int i = 0; i < axis; ++i)
			{
				copy_times *= this->shape(i);
			}

			int copy_size = 1;

			for (int i = axis + 1; i < 4; ++i)
			{
				copy_size *= this->shape(i);
			}

			int src_axis_dim = this->shape(axis);
			int dst_axis_dim = result.shape(axis);

			for (int i = 0; i < copy_times; ++i)
			{
				for (int n = 0; n < dst_axis_dim; ++n)
				{
					auto dst_ptr = &result.data(i * dst_axis_dim * copy_size + n * copy_size);
					auto src_ptr = &this->data(i * src_axis_dim * copy_size + dims[n] * copy_size);
					CopyData(dst_ptr, src_ptr, copy_size);
				}
			}

			return result;
		}

	private:
		std::shared_ptr<T> m_data;
		std::vector<int> m_shape;
		std::vector<int> m_capacity;
	};

	template <typename T>
	Blob<T> stack(const std::vector<Blob<T>> &blobs, int axis)
	{
		// check axis
		if (axis < 0 || axis > 3)
			throw std::logic_error("Can not concat blobs at axis " + std::to_string(axis));

		if (blobs.empty())
			return Blob<T>();

		size_t num_in_ = blobs.size();
		const Blob<T> &input0 = blobs[0];

		// check input
		for (int i = 1; i < num_in_; ++i)
		{
			const Blob<T> &inputi = blobs[i];

			for (int j = 0; j < 4; ++j)
			{
				if (j == axis)
					continue;

				if (input0.shape(j) != inputi.shape(j))
				{
					std::ostringstream oss;
					oss << "Check failed: Input dim(" << j << ")s (input 0 vs. input " << i << ")(" <<
						input0.shape(j) << " vs. " << inputi.shape(j) << ") must be equal.";
					throw std::logic_error(oss.str());
				}
			}
		}

		// get shape
		std::vector<int> shape = input0.shape();

		for (int i = 1; i < num_in_; ++i)
		{
			const Blob<T> &inputi = blobs[i];
			shape[axis] += inputi.shape(axis);
		}

		Blob<T> stacked_blob(shape);

		// concat
		int num_concats_ = 1;

		for (int i = 0; i < axis; ++i)
		{
			num_concats_ *= input0.shape(i);
		}

		int concat_input_size_ = 1;

		for (int i = axis + 1; i < 4; ++i)
		{
			concat_input_size_ *= input0.shape(i);
		}

		T *top_data = stacked_blob.data();
		int offset_concat_axis = 0;
		const int top_concat_axis = shape[axis];

		for (int i = 0; i < num_in_; ++i)
		{
			const Blob<T> &inputi = blobs[i];
			const float *bottom_data = inputi.data();
			const int bottom_concat_axis = inputi.shape(axis);

			for (int n = 0; n < num_concats_; ++n)
			{
				auto *_dst = top_data + (n * top_concat_axis + offset_concat_axis) * concat_input_size_;
				const auto *_src = bottom_data + n * bottom_concat_axis * concat_input_size_;
				size_t _size = bottom_concat_axis * concat_input_size_;
				CopyData(_dst, _src, _size);
			}

			offset_concat_axis += bottom_concat_axis;
		}

		return stacked_blob;
	}

	template <typename T>
	Blob<T> stack(const Blob<T> &blob1, const Blob<T> &blob2, int axis)
	{
		return stack<T>({ blob1, blob2 }, axis);
	}
}
