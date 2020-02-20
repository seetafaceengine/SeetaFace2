#include <SeetaNetParseProto.h>

#ifdef _WIN32
#include <WinSock2.h>
#else
#include <arpa/inet.h>
#endif
#include <iostream>
#include <string.h>

int read(const char *buf, int len, int32_t &value)
{
	if (len < sizeof(int32_t))
	{
		std::cout << "the buffer length is short, parse int failed" << std::endl;
		return -1;
	}

	value = 0;
	memcpy(&value, buf, sizeof(int32_t));
	value = ntohl(value);
	return sizeof(int32_t);
}

int read(const char *buf, int len, uint32_t &value)
{
	if (len < sizeof(uint32_t))
	{
		std::cout << "the buffer length is short, parse uint32_t failed" << std::endl;
		return -1;
	}

	value = 0;
	memcpy(&value, buf, sizeof(uint32_t));
	value = ntohl(value);
	return sizeof(uint32_t);
}

int read(const char *buf, int len, bool &value)
{
	if (len < 1)
	{
		std::cout << "the buffer length is short, parse bool failed" << std::endl;
		return -1;
	}

	unsigned char n = 0;
	n = buf[0];

	if (n != 0)
	{
		value = true;
	}
	else
	{
		value = false;
	}

	return 1;
}

int read(const char *buf, int len, float &value)
{
	if (len < sizeof(float))
	{
		std::cout << "the buffer length is short, parse float failed" << std::endl;
		return -1;
	}

	value = 0.0;
	memcpy(&value, buf, sizeof(float));
	return sizeof(float);
}

int read(const char *buf, int len, vector<float> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "the buffer length is short, parse array size failed" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = 0;

	offset += read(buf + offset, len - offset, size);

	if (len < offset + size * sizeof(float))
	{
		std::cout << "parse float array failed, the buf len is short!" << std::endl;
		return -1;
	}

	float tmpvalue = 0.0;

	for (int i = 0; i < size; i++)
	{
		tmpvalue = 0.0;
		memcpy(&tmpvalue, buf + offset, sizeof(float));
		value.push_back(tmpvalue);
		offset += sizeof(float);
	}

	return offset;
}

int read(const char *buf, int len, vector<uint32_t> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "the buffer length is short, parse array size failed" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = 0;
	offset += read(buf + offset, len - offset, size);

	if (len < offset + size * sizeof(uint32_t))
	{
		std::cout << "parse float array failed, the buf len is short!" << std::endl;
		return -1;
	}

	uint32_t tmpvalue = 0;

	for (int i = 0; i < size; i++)
	{
		tmpvalue = 0;
		offset += read(buf + offset, len - offset, tmpvalue);
		value.push_back(tmpvalue);
	}

	return offset;
}

int read(const char *buf, int len, vector<int32_t> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "the buffer length is short, parse array size failed" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = 0;

	offset += read(buf + offset, len - offset, size);

	if (len < offset + size * sizeof(uint32_t))
	{
		std::cout << "parse float array failed, the buf len is short!" << std::endl;
		return -1;
	}

	int32_t tmpvalue = 0;

	for (int i = 0; i < size; i++)
	{
		tmpvalue = 0;
		offset += read(buf + offset, len - offset, tmpvalue);
		value.push_back(tmpvalue);
	}

	return offset;
}

int read(const char *buf, int len, std::string &value)
{
	if (len < sizeof(uint32_t))
	{
		std::cout << "the buffer length is short, read string field failed" << std::endl;
		return -1;
	}

	int size = 0;
	int offset = 0;
	offset += read(buf + offset, len - offset, size);
	std::string str(buf + offset, size);
	value = str;

	return size + sizeof(int);;
}

int read(const char *buf, int len, vector<string> &value)
{
	if (len < sizeof(uint32_t))
	{
		std::cout << "the buffer length is short, read string field failed" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = 0;

	offset += read(buf + offset, len - offset, size);
	int nret = 0;

	for (int i = 0; i < size; i++)
	{
		string str;
		nret = read(buf + offset, len - offset, str);

		if (nret < 0)
		{
			return -1;
		}

		offset += nret;
		value.push_back(str);
	}

	return offset;;
}

int write(char *buf, int len, int32_t value)
{
	if (len < sizeof(int32_t))
	{
		std::cout << "write int failed, the buf len is short!" << std::endl;
		return -1;
	}

	value = htonl(value);
	memcpy(buf, &value, sizeof(int32_t));

	return sizeof(int32_t);
}

int write(char *buf, int len, uint32_t value)
{
	if (len < sizeof(uint32_t))
	{
		std::cout << "write uint32_t failed, the buf len is short!" << std::endl;
		return -1;
	}

	value = htonl(value);
	memcpy(buf, &value, sizeof(uint32_t));

	return sizeof(uint32_t);
}

int write(char *buf, int len, bool value)
{
	if (len < 1)
	{
		std::cout << "write uint32_t failed, the buf len is short!" << std::endl;
		return -1;
	}

	char n = 0;

	if (value)
	{
		n = 1;
	}
	else
	{
		n = 0;
	}

	buf[0] = n;
	return 1;
}

int write(char *buf, int len, float value)
{
	if (len < sizeof(float))
	{
		std::cout << "write float failed, the buf len is short!" << std::endl;
		return -1;
	}

	memcpy(buf, &value, sizeof(float));

	return sizeof(float);
}
int write(char *buf, int len, const vector<float> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "write float array failed, the buf len is short!" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = int(value.size());

	offset += write(buf + offset, len - offset, size);
	int nret = 0;

	for (size_t i = 0; i < value.size(); i++)
	{
		nret = write(buf + offset, len - offset, value[i]);

		if (nret < 0)
		{
			std::cout << "write float array failed, the buf len is short!" << std::endl;
			return -1;
		}

		offset += nret;
	}

	return offset;
}

int write(char *buf, int len, const vector<uint32_t> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "write uint32_t array failed, the buf len is short!" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = int(value.size());

	offset += write(buf + offset, len - offset, size);
	int nret = 0;

	for (size_t i = 0; i < value.size(); i++)
	{
		nret = write(buf + offset, len - offset, value[i]);

		if (nret < 0)
		{
			std::cout << "write uint32_t array failed, the buf len is short!" << std::endl;
			return -1;
		}

		offset += nret;
	}

	return offset;
}

int write(char *buf, int len, const vector<int32_t> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "write int32_t array failed, the buf len is short!" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = int(value.size());

	offset += write(buf + offset, len - offset, size);
	int nret = 0;

	for (size_t i = 0; i < value.size(); i++)
	{
		nret = write(buf + offset, len - offset, value[i]);

		if (nret < 0)
		{
			std::cout << "write int32_t array failed, the buf len is short!" << std::endl;
			return -1;
		}

		offset += nret;
	}

	return offset;
}

int write(char *buf, int len, const std::string &value)
{
	if (len < sizeof(int) + value.length())
	{
		std::cout << "write string failed, the buf len is short!" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = int(value.length());

	offset += write(buf + offset, len - offset, size);
	memcpy(buf + offset, value.c_str(), value.length());
	offset += int(value.length());

	return offset;
}

int write(char *buf, int len, const vector<string> &value)
{
	if (len < sizeof(int))
	{
		std::cout << "write string array failed, the buf len is short!" << std::endl;
		return -1;
	}

	int offset = 0;
	int size = int(value.size());

	offset += write(buf + offset, len - offset, size);
	int nret = 0;

	for (size_t i = 0; i < value.size(); i++)
	{
		nret = write(buf + offset, len - offset, value[i]);

		if (nret < 0)
		{
			std::cout << "write string array failed";
			return -1;
		}

		offset += nret;
	}

	return offset;
}

int WriteStringToFile(const std::string &input_string, std::fstream &outputfstream)
{
	int string_size = int(input_string.size());
	string_size = htonl(string_size);

	outputfstream.write(reinterpret_cast<char *>(&string_size), sizeof(int));
	outputfstream.write(input_string.data(), sizeof(char) * input_string.length());

	return int(sizeof(int) + sizeof(char) * input_string.length());
}

int WriteStringVectorToFile(const std::vector<std::string> &vec, std::fstream &outputfstream)
{
	int string_size = int(vec.size());
	string_size = htonl(string_size);

	outputfstream.write(reinterpret_cast<char *>(&string_size), sizeof(int));
	int offset = sizeof(int);

	for (size_t i = 0; i < vec.size(); i++)
	{
		offset += WriteStringToFile(vec[i], outputfstream);
	}

	return offset;
}

