#ifndef _SEETANET_RELU_CPU_H_
#define _SEETANET_RELU_CPU_H_

#include "SeetaNetBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <class T>
class SeetaNetReluCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetReluCPU();
    ~SeetaNetReluCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:

    T m_negetive_slope;
    bool m_has_max = false;
    T m_max;

};


template <class T>
SeetaNetReluCPU<T>::SeetaNetReluCPU()
{

}


template <class T>
SeetaNetReluCPU<T>::~SeetaNetReluCPU()
{

}


template <class T>
int SeetaNetReluCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
    int bottom_index = inputparam.bottom_index[0];
    SeetaNetDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize( 1 );
    this->bottom_data_size[0] = bottom_size;
    seeta::SeetaNet_ReLUParameter *msg = ( seeta::SeetaNet_ReLUParameter * )inputparam.msg.get();
    m_negetive_slope = msg->negative_slope;

    m_has_max = msg->has_max();
    if( m_has_max )
    {
        m_max = msg->max;
    }

    this->top_data_size.resize( 1 );
    this->top_data_size[0] = this->bottom_data_size[0];

    return 0;
}


template <class T>
int SeetaNetReluCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    input_data_map[0]->TransFormDataIn();
    int all_size = 1;
    for( int i = 0; i < 4; i++ )
    {
        all_size *= input_data_map[0]->data_shape[i];
    }

    if( this->bottom_index[0] != this->top_index[0] )
    {
        memcpy( output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof( T )*all_size );
    }

    auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

    if( m_has_max )
    {
        if( gun == nullptr || gun->size() <= 1 )
        {
            T *start_point = output_data_map[0]->m_cpu.dataMemoryPtr();
            for( int i = 0; i < all_size; i++ )
            {
                T val = *start_point;
                T result = std::max( val, T( 0.0 ) ) + m_negetive_slope * std::min( val, T( 0.0 ) );
                *start_point = std::min<T>( result, m_max );
                start_point++;
            }
        }
        else
        {
            auto bins = orz::split_bins( 0, all_size, int( gun->size() ) );
            for( auto &bin : bins )
            {
                gun->fire( [ &, bin]( int )
                {
                    auto start_point = output_data_map[0]->m_cpu.dataMemoryPtr() + bin.first;
                    for( int i = bin.first; i < bin.second; ++i )
                    {
                        T val = *start_point;
                        T result = std::max( val, T( 0.0 ) ) + m_negetive_slope * std::min( val, T( 0.0 ) );
                        *start_point = std::min<T>( result, m_max );
                        start_point++;
                    }
                } );
            }
            gun->join();
        }
    }
    else
    {
        if( gun == nullptr || gun->size() <= 1 )
        {
            T *start_point = output_data_map[0]->m_cpu.dataMemoryPtr();
            for( int i = 0; i < all_size; i++ )
            {
                T val = *start_point;
                T result = std::max( val, T( 0.0 ) ) + m_negetive_slope * std::min( val, T( 0.0 ) );
                *start_point = result;
                start_point++;
            }
        }
        else
        {
            auto bins = orz::split_bins( 0, all_size, int( gun->size() ) );
            for( auto &bin : bins )
            {
                gun->fire( [ &, bin]( int )
                {
                    auto start_point = output_data_map[0]->m_cpu.dataMemoryPtr() + bin.first;
                    for( int i = bin.first; i < bin.second; ++i )
                    {
                        T val = *start_point;
                        T result = std::max( val, T( 0.0 ) ) + m_negetive_slope * std::min( val, T( 0.0 ) );
                        *start_point = result;
                        start_point++;
                    }
                } );
            }
            gun->join();
        }
    }

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
    output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
    output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

    return 0;
}

#endif
