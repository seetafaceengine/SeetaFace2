#ifndef INC_SEETA_STRUCT_H
#define INC_SEETA_STRUCT_H

#include "CStruct.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <istream>
#include <string>

#define INCLUDED_SEETA_STRUCT

namespace seeta
{
	class ImageData : public SeetaImageData
	{
	public:
		using self = ImageData;
		using supper = SeetaImageData;
		using byte = unsigned char;

		~ImageData() = default;

		ImageData(const supper &image) : ImageData(image.data, image.width, image.height, image.channels) {}

		ImageData(int width, int height, int channels) : supper({ width, height, channels, nullptr })
		{
			this->m_data.reset(new byte[this->count()], std::default_delete<byte[]>());
			this->data = this->m_data.get();
		}

		ImageData() : ImageData(0, 0, 0) {}

		ImageData(const byte *data, int width, int height, int channels) : ImageData(width, height, channels)
		{
			this->copy_from(data);
		}

		ImageData(const self &) = default;

		ImageData &operator=(const self &) = default;

		ImageData &operator=(const supper &other)
		{
			this->operator=(self(other));
			return *this;
		}

		ImageData(self &&other) : supper({ other.width, other.height, other.channels, nullptr })
		{
			this->m_data = std::move(other.m_data);
			this->data = this->m_data.get();
		}

		ImageData &operator=(self &&other)
		{
			this->width = other.width;
			this->height = other.height;
			this->channels = other.channels;
			this->m_data = std::move(other.m_data);
			this->data = this->m_data.get();
			return *this;
		}

		void copy_from(const byte *data, int size = -1)
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min<int>(copy_size, size);
			copy(this->data, data, copy_size);
		}

		void copy_to(byte *data, int size = -1) const
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min<int>(copy_size, size);
			copy(data, this->data, copy_size);
		}

		static void copy(byte *dst, const byte *src, size_t size)
		{
			std::memcpy(dst, src, size);
		}

		int count() const
		{
			return this->width * this->height * this->channels;
		}

		ImageData clone() const
		{
			return ImageData(this->data, this->width, this->height, this->channels);
		}

	private:
		std::shared_ptr<byte> m_data;
	};

	class Point : public SeetaPoint
	{
	public:
		using self = Point;
		using supper = SeetaPoint;

		Point(const supper &other) : supper(other) {}
		Point(int x, int y) : supper({ x, y }) {}

		Point() : Point(0, 0) {}
	};

	class PointF : public SeetaPointF
	{
	public:
		using self = PointF;
		using supper = SeetaPointF;

		PointF(const supper &other) : supper(other) {}
		PointF(double x, double y) : supper({ x, y }) {}

		PointF() : PointF(0, 0) {}
	};

	class Size : public SeetaSize
	{
	public:
		using self = Size;
		using supper = SeetaSize;

		Size(const supper &other) : supper(other) {}
		Size(int width, int height) : supper({ width, height }) {}

		Size() : Size(0, 0) {}
	};

	class Rect : public SeetaRect
	{
	public:
		using self = Rect;
		using supper = SeetaRect;

		Rect(const supper &other) : supper(other) {}
		Rect(int x, int y, int width, int height) : supper({ x, y, width, height }) {}

		Rect() : Rect(0, 0, 0, 0) {}

		Rect(int x, int y, const Size &size) : supper({ x, y, size.width, size.height }) {}
		Rect(const Point &top_left, int width, int height) : supper({ top_left.x, top_left.y, width, height }) {}
		Rect(const Point &top_left, const Size &size) : supper({ top_left.x, top_left.y, size.width, size.height }) {}
		Rect(const Point &top_left, const Point &bottom_right) : supper({ top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y }) {}

		operator Point() const
		{
			return{ this->x, this->y };
		}

		operator Size() const
		{
			return{ this->width, this->height };
		}
	};

	class Region : public SeetaRegion
	{
	public:
		using self = Region;
		using supper = SeetaRegion;

		Region(const supper &other) : supper(other) {}
		Region(int top, int bottom, int left, int right) : supper({ top, bottom, left, right }) {}

		Region() : Region(0, 0, 0, 0) {}

		Region(const Rect &rect) : Region(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width) {}

		operator Rect() const
		{
			return{ left, top, right - left, bottom - top };
		}
	};

	class ModelSetting : public SeetaModelSetting
	{
	public:
		using self = ModelSetting;
		using supper = SeetaModelSetting;

		enum Device
		{
			AUTO,
			CPU,
			GPU
		};

		~ModelSetting() = default;

		ModelSetting() : supper({ SEETA_DEVICE_AUTO, 0, nullptr })
		{
			this->update();
		}

		ModelSetting(const supper &other) : supper({ other.device, other.id, nullptr })
		{
			if (other.model)
			{
				int i = 0;

				while (other.model[i])
				{
					m_model_string.emplace_back(other.model[i]);
					++i;
				}
			}

			this->update();
		}

		ModelSetting(const self &other) : supper({ other.device, other.id, nullptr })
		{
			this->m_model_string = other.m_model_string;
			this->update();
		}

		ModelSetting &operator=(const supper &other)
		{
			this->operator=(self(other));
			return *this;
		}

		ModelSetting &operator=(const self &other)
		{
			this->device = other.device;
			this->id = other.id;
			this->m_model_string = other.m_model_string;
			this->update();
			return *this;
		}

		ModelSetting(self &&other) : supper({ other.device, other.id, nullptr })
		{
			this->m_model_string = std::move(other.m_model_string);
			this->update();
		}

		ModelSetting &operator=(self &&other)
		{
			this->device = other.device;
			this->id = other.id;
			this->m_model_string = std::move(other.m_model_string);
			this->update();
			return *this;
		}

		ModelSetting(const std::string &model, SeetaDevice device, int id) : supper({ device, id, nullptr })
		{
			this->append(model);
		}

		ModelSetting(const std::string &model, SeetaDevice device) : self(model, device, 0) {}

		ModelSetting(const std::string &model, Device device, int id) : self(model, SeetaDevice(device), id) {}

		ModelSetting(const std::string &model, Device device) : self(model, SeetaDevice(device)) {}

		ModelSetting(const std::string &model) : self(model, SEETA_DEVICE_AUTO) {}

		ModelSetting(const std::vector<std::string> &model, SeetaDevice device, int id) : supper({ device, id, nullptr })
		{
			this->append(model);
		}

		ModelSetting(const std::vector<std::string> &model, SeetaDevice device) : self(model, device, 0) {}

		ModelSetting(const std::vector<std::string> &model, Device device, int id) : self(model, SeetaDevice(device), id) {}

		ModelSetting(const std::vector<std::string> &model, Device device) : self(model, SeetaDevice(device)) {}

		ModelSetting(const std::vector<std::string> &model) : self(model, SEETA_DEVICE_AUTO) {}

		ModelSetting(SeetaDevice device, int id) : supper({ device, id, nullptr })
		{
			this->update();
		}

		ModelSetting(SeetaDevice device) : self(device, 0) {}

		ModelSetting(Device device, int id) : self(SeetaDevice(device), id) {}

		ModelSetting(Device device) : self(SeetaDevice(device)) {}

		Device get_device() const
		{
			return Device(this->device);
		}

		int get_id() const
		{
			return this->id;
		}

		Device set_device(Device device)
		{
			return set_device(SeetaDevice(device));
		}

		Device set_device(SeetaDevice device)
		{
			auto old = this->device;
			this->device = device;
			return Device(old);
		}

		int set_id(int id)
		{
			const auto old = this->id;
			this->id = id;
			return old;
		}

		void clear()
		{
			this->m_model_string.clear();
			this->update();
		}

		void append(const std::string &model)
		{
			this->m_model_string.push_back(model);
			this->update();
		}

		void append(const std::vector<std::string> &model)
		{
			this->m_model_string.insert(this->m_model_string.end(), model.begin(), model.end());
			this->update();
		}

		const std::vector<std::string> &get_model() const
		{
			return this->m_model_string;
		}

		const std::string &get_model(size_t i) const
		{
			return this->m_model_string[i];
		}

		size_t count() const
		{
			return this->m_model_string.size();
		}

	private:
		std::vector<const char *> m_model;
		std::vector<std::string> m_model_string;

		/**
		 * \brief build buffer::model
		 */
		void update()
		{
			m_model.clear();
			m_model.reserve(m_model_string.size() + 1);

			for (auto &model_string : m_model_string)
			{
				m_model.push_back(model_string.c_str());
			}

			m_model.push_back(nullptr);
			this->model = m_model.data();
		}
	};

	class Buffer : public SeetaBuffer
	{
	public:
		using self = Buffer;
		using supper = SeetaBuffer;
		using byte = unsigned char;

		~Buffer() = default;

		Buffer(const supper &other) : Buffer(other.buffer, other.size) {}
		Buffer(const self &) = default;

		Buffer &operator=(const self &) = default;
		Buffer &operator=(const supper &other)
		{
			this->operator=(self(other));
			return *this;
		}

		Buffer(self &&other) : supper({ other.buffer, other.size }), m_buffer(std::move(other.m_buffer)), m_size(other.m_size) {}

		Buffer &operator=(self &&other)
		{
			this->buffer = other.buffer;
			this->size = other.size;
			this->m_buffer = std::move(other.m_buffer);
			this->m_size = other.m_size;
			return *this;
		}

		/**
		 * \brief contruct cpp-style-buffer from c-style-buffer, optional borrow or copy
		 * \param other c-style-buffer
		 * \param borrow if borrow or copy buffer
		 */
		Buffer(const supper &other, bool borrow) : supper({ nullptr, 0 })
		{
			if (borrow)
				this->borrow(other.buffer, other.size);
			else
				this->rebind(other.buffer, other.size);
		}

		Buffer(const void *buffer, int64_t size) : supper({ nullptr, 0 })
		{
			this->m_size = size;

			if (this->m_size)
			{
				this->m_buffer.reset(new byte[size_t(this->m_size)], std::default_delete<byte[]>());
			}

			this->buffer = this->m_buffer.get();
			this->size = this->m_size;

			if (buffer != nullptr)
			{
				this->copy_from(buffer, size);
			}
		}

		explicit Buffer(int64_t size) : Buffer(nullptr, size) {}

		Buffer() : Buffer(nullptr, 0) {}

		Buffer(std::istream &in, int64_t size = -1) : supper({ nullptr, 0 })
		{
			if (size < 0)
			{
				const auto cur = in.tellg();
				in.seekg(0, std::ios::end);
				const auto end = in.tellg();
				size = int64_t(end - cur);
				in.seekg(-size, std::ios::end);
			}

			this->m_size = size;
			this->m_buffer.reset(new byte[size_t(this->m_size)], std::default_delete<byte[]>());
			this->buffer = this->m_buffer.get();
			this->size = this->m_size;

			in.read(reinterpret_cast<char *>(this->buffer), this->size);
		}

		void copy_from(const void *data, int64_t size = -1)
		{
			auto copy_size = this->m_size;
			copy_size = size < 0 ? copy_size : std::min<int64_t>(copy_size, size);
			copy(this->buffer, data, size_t(copy_size));
		}

		void copy_to(byte *data, int64_t size = -1) const
		{
			auto copy_size = this->m_size;
			copy_size = size < 0 ? copy_size : std::min<int64_t>(copy_size, size);
			copy(data, this->buffer, size_t(copy_size));
		}

		static void copy(void *dst, const void *src, size_t size)
		{
			if (dst == nullptr || src == nullptr)
				return;

			std::memcpy(dst, src, size);
		}

		Buffer clone() const
		{
			return self(this->buffer, this->size);
		}

		void rebind(const void *buffer, int64_t size)
		{
			if (size < 0)
				size = 0;

			if (size > this->m_size)
			{
				this->m_buffer.reset(new byte[size_t(size)], std::default_delete<byte[]>());
			}

			this->m_size = size;
			this->buffer = this->m_buffer.get();
			this->size = this->m_size;
			this->copy_from(buffer, size);
		}

		/**
		 * \brief borrow c-style buffer
		 * \param buffer pointert to buffer
		 * \param size size of buffer
		 */
		void borrow(void *buffer, int64_t size)
		{
			this->m_size = 0;
			this->m_buffer.reset();
			this->buffer = buffer;
			this->size = size;
		}

	private:
		std::shared_ptr<byte> m_buffer;
		int64_t m_size = 0;
	};

	class ModelBuffer : SeetaModelBuffer
	{
	public:
		using self = ModelBuffer;
		using supper = SeetaModelBuffer;

		enum Device
		{
			AUTO = SEETA_DEVICE_AUTO,
			CPU = SEETA_DEVICE_CPU,
			GPU = SEETA_DEVICE_GPU,
		};

		~ModelBuffer() = default;

		ModelBuffer() : supper({ SEETA_DEVICE_AUTO, 0, nullptr })
		{
			this->update();
		}

		ModelBuffer(const supper &other) : supper({ other.device, other.id, nullptr })
		{
			if (other.buffer)
			{
				int i = 0;

				while (other.buffer[i].buffer && other.buffer[i].size)
				{
					m_model_buffer.emplace_back(other.buffer[i]);
					++i;
				}
			}

			this->update();
		}

		ModelBuffer(const self &other) : supper({ other.device, other.id, nullptr })
		{
			this->m_model_buffer = other.m_model_buffer;
			this->update();
		}

		ModelBuffer &operator=(const supper &other)
		{
			this->operator=(self(other));
			return *this;
		}

		ModelBuffer &operator=(const self &other)
		{
			this->device = other.device;
			this->id = other.id;
			this->m_model_buffer = other.m_model_buffer;
			this->update();
			return *this;
		}

		ModelBuffer(self &&other) : supper({ other.device, other.id, nullptr })
		{
			this->m_model_buffer = std::move(other.m_model_buffer);
			this->update();
		}

		ModelBuffer &operator=(self &&other)
		{
			this->device = other.device;
			this->id = other.id;
			this->m_model_buffer = std::move(other.m_model_buffer);
			this->update();
			return *this;
		}

		ModelBuffer(const seeta::Buffer &buffer, SeetaDevice device, int id) : supper({ device, id, nullptr })
		{
			this->append(buffer);
		}

		ModelBuffer(const seeta::Buffer &buffer, SeetaDevice device) : self(buffer, device, 0) {}

		ModelBuffer(const seeta::Buffer &buffer, Device device, int id) : self(buffer, SeetaDevice(device), id) {}

		ModelBuffer(const seeta::Buffer &buffer, Device device) : self(buffer, SeetaDevice(device)) {}

		ModelBuffer(const seeta::Buffer &buffer) : self(buffer, SEETA_DEVICE_AUTO) {}

		ModelBuffer(const std::vector<seeta::Buffer> &buffer, SeetaDevice device, int id) : supper({ device, id, nullptr })
		{
			this->append(buffer);
		}

		ModelBuffer(const std::vector<seeta::Buffer> &buffer, SeetaDevice device) : self(buffer, device, 0) {}

		ModelBuffer(const std::vector<seeta::Buffer> &buffer, Device device, int id) : self(buffer, SeetaDevice(device), id) {}

		ModelBuffer(const std::vector<seeta::Buffer> &buffer, Device device) : self(buffer, SeetaDevice(device)) {}

		ModelBuffer(const std::vector<seeta::Buffer> &buffer) : self(buffer, SEETA_DEVICE_AUTO) {}

		ModelBuffer(SeetaDevice device, int id) : supper({ device, id, nullptr })
		{
			this->update();
		}

		ModelBuffer(SeetaDevice device) : self(device, 0) {}

		ModelBuffer(Device device, int id) : self(SeetaDevice(device), id) {}

		ModelBuffer(Device device) : self(SeetaDevice(device)) {}

		Device get_device() const
		{
			return Device(this->device);
		}

		int get_id() const
		{
			return this->id;
		}

		Device set_device(Device device)
		{
			return set_device(SeetaDevice(device));
		}

		Device set_device(SeetaDevice device)
		{
			auto old = this->device;
			this->device = device;
			return Device(old);
		}

		int set_id(int id)
		{
			const auto old = this->id;
			this->id = id;
			return old;
		}

		void clear()
		{
			this->m_model_buffer.clear();
			this->update();
		}

		void append(const seeta::Buffer &buffer)
		{
			this->m_model_buffer.push_back(buffer);
			this->update();
		}

		void append(const std::vector<seeta::Buffer> &model)
		{
			this->m_model_buffer.insert(this->m_model_buffer.end(), model.begin(), model.end());
			this->update();
		}

		const std::vector<seeta::Buffer> &get_buffer() const
		{
			return this->m_model_buffer;
		}

		const seeta::Buffer &get_buffer(size_t i) const
		{
			return this->m_model_buffer[i];
		}

		size_t count() const
		{
			return this->m_model_buffer.size();
		}

	private:
		std::vector<SeetaBuffer> m_buffer;
		std::vector<seeta::Buffer> m_model_buffer;

		/**
		 * \brief build supper::buffer
		 */
		void update()
		{
			this->m_buffer.clear();
			this->m_buffer.reserve(m_model_buffer.size() + 1);

			for (auto &model_buffer : m_model_buffer)
			{
				this->m_buffer.push_back(model_buffer);
			}

			this->m_buffer.push_back({ nullptr, 0 });  // terminate with empty buffer
			this->buffer = this->m_buffer.data();
		}
	};
}

#endif
