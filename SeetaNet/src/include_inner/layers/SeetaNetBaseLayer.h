#ifndef _SEETANET_BASE_LAYER_
#define _SEETANET_BASE_LAYER_

#include <stdint.h>
#include "SeetaNetProto.h"
#include "SeetaNetCommon.h"
#include "SeetaNetFeatureMap.h"
#include "SeetaNetResource.h"


template<typename T>
class SeetaNetBaseLayer {
public:
    SeetaNetBaseLayer() {};
    virtual ~SeetaNetBaseLayer() {};

    virtual int GetTopSize( std::vector<SeetaNetDataSize> &out_data_size );
    virtual int Exit() {
        return 0;
    };
    virtual int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource ) {
        m_layer_index = inputparam.layer_index;
        return 0;
    };
    virtual int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map ) {
        return 0;
    };
public:
    std::vector<SeetaNetDataSize> bottom_data_size;
    std::vector<int64_t> bottom_index;

    std::vector<SeetaNetDataSize> top_data_size;
    std::vector<int64_t> top_index;
    int m_layer_index;
    int m_layer_type;
};

template <class T>
int SeetaNetBaseLayer<T>::GetTopSize( std::vector<SeetaNetDataSize> &out_data_size )
{
    out_data_size = this->top_data_size;
    return 0;
}

#endif

