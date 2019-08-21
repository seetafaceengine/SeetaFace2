#ifndef _SEETANET_BATCH_NORMALISE_CPU_H__
#define _SEETANET_BATCH_NORMALISE_CPU_H__


#include "SeetaNetBaseLayer.h"
#include <cfloat>

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <class T>
class SeetaNetBatchNormalizeCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetBatchNormalizeCPU();
    ~SeetaNetBatchNormalizeCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );

public:
    std::vector<T> m_meanvalue;
    std::vector<T> m_varines_value;
};



template <class T>
int SeetaNetBatchNormalizeCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    if( this->bottom_index[0] != this->top_index[0] )
    {
        output_data_map[0]->data_shape = input_data_map[0]->data_shape;
        memcpy( output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof( T )*output_data_map[0]->count() );
    }

    auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

    if( gun == nullptr || gun->size() <= 1 )
    {
        T *pstart = output_data_map[0]->m_cpu.dataMemoryPtr();
        for( int n = 0; n < input_data_map[0]->data_shape[0]; n++ )
        {
            for( int i = 0; i < output_data_map[0]->data_shape[1]; i++ )
            {
                T val2 = m_meanvalue[i];
                T val3 = m_varines_value[i];
                for( int j = 0; j < output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3]; j++ )
                {
                    *pstart = *pstart - val2;
                    *pstart = *pstart / val3;
                    pstart++;
                }
            }
        }
    }
    else
    {
        auto col_size = output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
        auto batch_size = output_data_map[0]->data_shape[1] * col_size;
        for( int n = 0; n < input_data_map[0]->data_shape[0]; n++ )
        {
            auto local_pstart = output_data_map[0]->m_cpu.dataMemoryPtr() + n * batch_size;
            auto bins = orz::split_bins( 0, output_data_map[0]->data_shape[1], int( gun->size() ) );
            for( auto &bin : bins )
            {
                gun->fire( [ &, local_pstart, bin]( int )
                {
                    auto pstart = local_pstart + bin.first * col_size;
                    for( int i = bin.first; i < bin.second; i++ )
                    {
                        T val2 = m_meanvalue[i];
                        T val3 = m_varines_value[i];
                        for( int j = 0; j < col_size; j++ )
                        {
                            *pstart = *pstart - val2;
                            *pstart = *pstart / val3;
                            pstart++;
                        }
                    }
                } );
            }
        }
        gun->join();
    }

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
    output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
    output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

    return 0;
}

template <class T>
int SeetaNetBatchNormalizeCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
    m_meanvalue.clear();
    seeta::SeetaNet_BatchNormliseParameter *msg = ( seeta::SeetaNet_BatchNormliseParameter * )inputparam.msg.get();
    auto length_mean = msg->mean_param.data.size();
    for( size_t i = 0; i < length_mean; i++ )
    {
        auto tmp_mean_value = msg->mean_param.data[i];
        if( tmp_mean_value < FLT_EPSILON && -tmp_mean_value < FLT_EPSILON ) tmp_mean_value = 0;
        m_meanvalue.push_back( tmp_mean_value );
    }
    m_varines_value.clear();
	size_t length_covariance = msg->covariance_param.data.size();
    for( size_t i = 0; i < length_covariance; i++ )
    {
        auto tmp_varines_value = msg->covariance_param.data[i];
        if( tmp_varines_value < FLT_EPSILON && -tmp_varines_value < FLT_EPSILON ) tmp_varines_value = 0;
        m_varines_value.push_back( tmp_varines_value );
    }

    int bottom_index = inputparam.bottom_index[0];
    SeetaNetDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize( 1 );
    this->bottom_data_size[0] = bottom_size;

    this->top_data_size.resize( 1 );
    this->top_data_size[0] = this->bottom_data_size[0];

    return 0;
};

template <class T>
SeetaNetBatchNormalizeCPU<T>::SeetaNetBatchNormalizeCPU()
{

};


template <class T>
SeetaNetBatchNormalizeCPU<T>::~SeetaNetBatchNormalizeCPU()
{

};


#endif //!_BATCHNORMALIZE_H__
