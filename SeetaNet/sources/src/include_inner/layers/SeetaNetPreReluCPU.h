#ifndef _SEETANET_PRERELU_CPU_H_
#define _SEETANET_PRERELU_CPU_H_
#include "SeetaNetBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <class T>
class SeetaNetPreReluCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetPreReluCPU();
    ~SeetaNetPreReluCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource );
    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:
    std::vector<T> m_slope_value;
};

template <class T>
SeetaNetPreReluCPU<T>::SeetaNetPreReluCPU()
{

}


template <class T>
SeetaNetPreReluCPU<T>::~SeetaNetPreReluCPU()
{

}


template <class T>
int SeetaNetPreReluCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
    seeta::SeetaNet_PreluParameter *msg = ( seeta::SeetaNet_PreluParameter * )inputparam.msg.get();
    m_slope_value.clear();
    int length_slope = msg->param.data.size();
    for( int i = 0; i < length_slope; i++ )
    {
        m_slope_value.push_back( msg->param.data[i] );
    }

    int bottom_length = inputparam.bottom_index.size();
    this->bottom_data_size.resize( bottom_length );
    for( size_t i = 0; i < bottom_length; i++ )
    {
        int index = inputparam.bottom_index[i];
        this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
    }

    this->top_data_size.resize( 1 );
    this->top_data_size[0] = this->bottom_data_size[0];

    return 0;
}


template <class T>
int SeetaNetPreReluCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    input_data_map[0]->TransFormDataIn();
    if( this->bottom_index[0] != this->top_index[0] )
    {
        output_data_map[0]->data_shape = input_data_map[0]->data_shape;
        memcpy( output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof( T )*output_data_map[0]->count() );
    }

    auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

    if( gun == nullptr || gun->size() <= 1 )
    {
        T *pstart = output_data_map[0]->m_cpu.dataMemoryPtr();
        for( int n = 0; n < output_data_map[0]->data_shape[0]; n++ )
        {
            for( int i = 0; i < output_data_map[0]->data_shape[1]; i++ )
            {
                T val2 = m_slope_value[i];
                for( int j = 0; j < output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3]; j++ )
                {
                    *pstart = ( std::max )( *pstart, T( 0 ) ) + val2 * ( std::min )( *pstart, T( 0 ) );
                    pstart++;
                }
            }
        }
    }
    else
    {
        auto col_size = output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
        auto batch_size = output_data_map[0]->data_shape[1] * col_size;
        for( int n = 0; n < output_data_map[0]->data_shape[0]; n++ )
        {
            T *local_pstart = output_data_map[0]->m_cpu.dataMemoryPtr() + n * batch_size;
            auto bins = orz::split_bins( 0, output_data_map[0]->data_shape[1], int( gun->size() ) );
            for( auto &bin : bins )
            {
                gun->fire( [ &, local_pstart, bin]( int )
                {
                    auto pstart = local_pstart + bin.first * col_size;
                    for( int i = bin.first; i < bin.second; ++i )
                    {
                        T val2 = m_slope_value[i];
                        for( int j = 0; j < output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3]; j++ )
                        {
                            *pstart = ( std::max )( *pstart, T( 0 ) ) + val2 * ( std::min )( *pstart, T( 0 ) );
                            ++pstart;
                        }
                    }
                } );
            }
        }
        gun->join();
    }

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];


    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
    output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
    output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

    return 0;
}




#endif //!__PRELU_H__
