#ifndef _SEETANET_SPACE_TO_BATCH_ND_H_
#define _SEETANET_SPACE_TO_BATCH_ND_H_

#include "SeetaNetBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

#include <assert.h>

template <class T>
class SeetaNetSpaceToBatchNDCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetSpaceToBatchNDCPU();
    ~SeetaNetSpaceToBatchNDCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );

    void CaculateOutputSize( int input_number, int input_height, int input_width, int input_channels,
                             int &output_number, int &output_height, int &output_width, int &output_channels );

    void CaculateOutputSize( const std::vector<int> &input_shape, std::vector<int> &output_shape ) {
        output_shape.resize( 4 );
        CaculateOutputSize( input_shape[0], input_shape[2], input_shape[3], input_shape[1],
                            output_shape[0], output_shape[2], output_shape[3], output_shape[1] );
    }
public:
    std::vector<int> m_block_shape;
    std::vector<int> m_paddings;
};


template <class T>
SeetaNetSpaceToBatchNDCPU<T>::SeetaNetSpaceToBatchNDCPU()
{

}


template <class T>
SeetaNetSpaceToBatchNDCPU<T>::~SeetaNetSpaceToBatchNDCPU()
{

}

template <class T>
void SeetaNetSpaceToBatchNDCPU<T>::CaculateOutputSize( int input_number, int input_height, int input_width, int input_channels,
        int &output_number, int &output_height, int &output_width, int &output_channels )
{
    output_number = input_number * m_block_shape[0] * m_block_shape[1];
    output_height = ( input_height + m_paddings[0] + m_paddings[1] ) / m_block_shape[0];
    output_width = ( input_width + m_paddings[2] + m_paddings[3] ) / m_block_shape[1];
    output_channels = input_channels;
}


template <class T>
int SeetaNetSpaceToBatchNDCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
    // set bottom size
    int bottom_index = inputparam.bottom_index[0];
    SeetaNetDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize( 1 );
    this->bottom_data_size[0] = bottom_size;

    // read param from inputparam.spacetobatchnd_param();
    seeta::SeetaNet_SpaceToBatchNDLayer *msg = ( seeta::SeetaNet_SpaceToBatchNDLayer * )inputparam.msg.get();
    for( int i = 0; i < msg->block_shape.size(); i++ )
    {
        m_block_shape.push_back( msg->block_shape[i] );
    }
    for( int i = 0; i < msg->paddings.size(); i++ )
    {
        m_paddings.push_back( msg->paddings[i] );
    }

    assert( m_block_shape.size() == 2 && m_paddings.size() == 4 );

    // set top size
    this->top_data_size.resize( 1 );
    this->top_data_size[0].data_dim.resize( 4 );
    CaculateOutputSize( this->bottom_data_size[0].data_dim, this->top_data_size[0].data_dim );

    return 0;
}


template <class T>
int SeetaNetSpaceToBatchNDCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    // trans param to cpu
    input_data_map[0]->TransFormDataIn();

    // set output data type and shape
    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    // set output data shape
    CaculateOutputSize( input_data_map[0]->data_shape, output_data_map[0]->data_shape );

    // write output
    int input_number = input_data_map[0]->data_shape[0];
    int input_channels = input_data_map[0]->data_shape[1];
    int input_height = input_data_map[0]->data_shape[2];
    int input_width = input_data_map[0]->data_shape[3];

    int output_number = output_data_map[0]->data_shape[0];
    int output_channels = output_data_map[0]->data_shape[1];
    int output_height = output_data_map[0]->data_shape[2];
    int output_width = output_data_map[0]->data_shape[3];

    int input_number_step = input_channels * input_height * input_width;
    int input_channels_step = input_height * input_width;
    int input_height_step = input_width;

    int output_size = output_number * output_channels * output_height * output_width;
    int output_number_step = output_channels * output_height * output_width;
    int output_channels_step = output_height * output_width;
    int output_height_step = output_width;

    const T *input_data = input_data_map[0]->m_cpu.dataMemoryPtr();
    T *output_data = output_data_map[0]->m_cpu.dataMemoryPtr();

    std::memset( output_data, 0, output_size * sizeof( T ) );

    const std::vector<int> &B = m_block_shape;
    const std::vector<int> &P = m_paddings;
    auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

    if( gun == nullptr || gun->size() <= 1 )
    {
        for( int n = 0; n < input_number; ++n )
        {
            for( int c = 0; c < input_channels; ++c )
            {
                for( int h = 0; h < input_height; ++h )
                {
                    for( int w = 0; w < input_width; ++w )
                    {
                        int on = ( ( h + P[0] ) % B[0] * B[1] + ( w + P[2] ) % B[1] ) * input_number + n;
                        int oc = c;
                        int oh = ( h + P[0] ) / B[0];
                        int ow = ( w + P[2] ) / B[1];

                        int at_input_i = n * input_number_step
                                         + c * input_channels_step
                                         + h * input_height_step
                                         + w;

                        int at_output_i = on * output_number_step
                                          + oc * output_channels_step
                                          + oh * output_height_step
                                          + ow;

                        output_data[at_output_i] = input_data[at_input_i];
                    }
                }
            }
        }
    }
    else
    {
        for( int n = 0; n < input_number; ++n )
        {
            auto bins = orz::split_bins( 0, input_channels, int( gun->size() ) );
            for( auto &bin : bins )
            {
                gun->fire( [ &, n, bin]( int )
                {
                    for( int c = bin.first; c < bin.second; ++c )
                    {
                        for( int h = 0; h < input_height; ++h )
                        {
                            for( int w = 0; w < input_width; ++w )
                            {
                                int on = ( ( h + P[0] ) % B[0] * B[1] + ( w + P[2] ) % B[1] ) * input_number + n;
                                int oc = c;
                                int oh = ( h + P[0] ) / B[0];
                                int ow = ( w + P[2] ) / B[1];

                                int at_input_i = n * input_number_step
                                                 + c * input_channels_step
                                                 + h * input_height_step
                                                 + w;

                                int at_output_i = on * output_number_step
                                                  + oc * output_channels_step
                                                  + oh * output_height_step
                                                  + ow;

                                output_data[at_output_i] = input_data[at_input_i];
                            }
                        }
                    }
                } );
            }
        }
        gun->join();
    }

    return 0;
}

#endif
