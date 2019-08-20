#ifndef _SEETANET_FEATURE_MAP_H_
#define _SEETANET_FEATURE_MAP_H_

#include "SeetaNetCommon.h"
#include "SeetaNetResource.h"
#include"SeetaNetBlobCpu.h"



enum DATA_STORAGE_TYPE
{
    DATA_INVALID = 0,
    DATA_CPU_WIDTH = 1,
    DATA_CPU_SLICE = 2,
    DATA_CPU_WIDTH_CHAR = 3,
    DATA_CPU_SLICE_CHAR = 4,
    DATA_GPU = 5
};

template<typename T>
class SeetaNetFeatureMap {
public:

    SeetaNetFeatureMap() {};
    ~SeetaNetFeatureMap() {};

    int TransFormDataIn();
    std::string data_name;
    std::vector<int> data_shape;
    int dwStorageType;
    SeetaNetResource<T> *pNetResource;
    SeetaNetBlobCpu<T> m_cpu;

    std::vector<int> &shape() {
        return data_shape;
    }

    const std::vector<int> &shape() const {
        return data_shape;
    }

    int &shape( size_t axis ) {
        return data_shape[axis];
    }

    const int &shape( size_t axis ) const {
        return data_shape[axis];
    }

    int count() const {
        int mul = 1;
        for( auto dim : data_shape ) mul *= dim;
        return mul;
    }

    T *cpu_ptr() {
        return m_cpu.dataMemoryPtr();
    }
};

template<typename T>
int SeetaNetFeatureMap<T>::TransFormDataIn()
{
    switch( dwStorageType )
    {
        case DATA_CPU_WIDTH:
        {
            break;
        }
        case DATA_GPU:
        {
            break;
        }
        default:
            break;
    }
    return 0;
}



#endif
