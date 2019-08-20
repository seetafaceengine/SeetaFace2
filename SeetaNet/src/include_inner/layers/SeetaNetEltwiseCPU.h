#ifndef _SEETANET_ELTWISE_CPU_H_
#define _SEETANET_ELTWISE_CPU_H_

#include "SeetaNetBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <class T>
class SeetaNetEltwiseCPU : public SeetaNetBaseLayer<T> {
public:
    enum EltwiseType
    {
        EltwiseParameter_EltwiseOp_PROD = 0,
        EltwiseParameter_EltwiseOp_SUM = 1,
        EltwiseParameter_EltwiseOp_MAX = 2
    };
    SeetaNetEltwiseCPU();
    ~SeetaNetEltwiseCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource );
    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );

public:
    int m_elt_type;

    std::vector<T> m_eltwise_coeff;
};

template <class T>
SeetaNetEltwiseCPU<T>::SeetaNetEltwiseCPU()
{

}

template <class T>
SeetaNetEltwiseCPU<T>::~SeetaNetEltwiseCPU()
{

}

template <class T>
int SeetaNetEltwiseCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
    int bottom_length = inputparam.bottom_index.size();
    this->bottom_data_size.resize( bottom_length );
    for( size_t i = 0; i < bottom_length; i++ )
    {
        int index = inputparam.bottom_index[i];
        this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
    }

    seeta::SeetaNet_EltwiseParameter *msg = ( seeta::SeetaNet_EltwiseParameter * )inputparam.msg.get();
    m_elt_type = msg->operation;
    int coeff_length = msg->coeff.size();
    m_eltwise_coeff.clear();
    for( int i = 0; i < coeff_length; i++ )
    {
        m_eltwise_coeff.push_back( msg->coeff[i] );
    }

    if( m_eltwise_coeff.empty() )
    {
        m_eltwise_coeff = std::vector<T>( this->bottom_data_size.size(), 1 );
    }


    this->top_data_size.resize( 1 );
    this->top_data_size[0] = this->bottom_data_size[0];

    return 0;
}

template<typename T>
void eltwise_prob( T *output, const std::vector<T *> &input, size_t size )
{
    auto local_input = input;
    for( size_t i = 0; i < size; i++ )
    {
        T inner_value = 1;
        for( size_t j = 0; j < local_input.size(); j++ )
        {
            inner_value *= *local_input[j];
            local_input[j] = local_input[j] + 1;
        }
        output[i] = inner_value;
    }
}

template<typename T>
void eltwise_sum( const std::vector<T> &coeff, T *output, const std::vector<T *> &input, size_t size )
{
    auto local_input = input;
    for( size_t i = 0; i < size; i++ )
    {
        T sum_value = 0.0;

        for( size_t j = 0; j < local_input.size(); j++ )
        {
            sum_value += coeff[j] * *local_input[j];
            local_input[j] = local_input[j] + 1;
        }
        output[i] = sum_value;
    }
}

template<typename T>
void eltwise_max( T *output, const std::vector<T *> &input, size_t size )
{
    if( input.empty() ) return;
    auto local_input = input;
    for( int i = 0; i < size; i++ )
    {
        T max_value = *local_input[0];
        for( int j = 1; j < input.size(); j++ )
        {
            max_value = std::max( max_value, *local_input[j] );
            local_input[j] = local_input[j] + 1;
        }
        output[i] = max_value;
    }

}

template<typename T, typename FUNC>
static void split_do( T *output, const std::vector<T *> &input, size_t size, FUNC func )
{
    auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
    if( gun == nullptr || gun->size() <= 1 )
    {
        func( output, input, size );
    }
    else
    {
        auto bins = orz::lsplit_bins( 0, size, gun->size() );
        for( auto &bin : bins )
        {
            gun->fire( [ &, bin]( int )
            {
                auto local_output = output + bin.first;
                auto local_input = input;
                for( auto &ptr : local_input ) ptr += bin.first;
                func( local_output, local_input, bin.second - bin.first );
            } );
        }
        gun->join();
    }
}

template <class T>
int SeetaNetEltwiseCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    int process_length = input_data_map.size();
    std::vector<T * > input_data_point;
    input_data_point.resize( process_length );
    T *ouput_point = output_data_map[0]->m_cpu.dataMemoryPtr();
    for( int index_bottom = 0; index_bottom < process_length; index_bottom++ )
    {
        input_data_point[index_bottom] = input_data_map[index_bottom]->m_cpu.dataMemoryPtr();
    }

    int needProcess_counts = 1;
    for( int i = 0; i < input_data_map[0]->data_shape.size(); i++ )
    {
        needProcess_counts *= input_data_map[0]->data_shape[i];
    }

    if( EltwiseParameter_EltwiseOp_PROD == m_elt_type )
    {
        split_do<T>( ouput_point, input_data_point, needProcess_counts, eltwise_prob<T> );
    }
    else
        if( EltwiseParameter_EltwiseOp_SUM == m_elt_type )
        {
            split_do<T>( ouput_point, input_data_point, needProcess_counts, std::bind( eltwise_sum<T>, m_eltwise_coeff, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3 ) );
        }
        else
            if( EltwiseParameter_EltwiseOp_MAX == m_elt_type )
            {
                split_do<T>( ouput_point, input_data_point, needProcess_counts, eltwise_max<T> );
            }

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
    output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
    output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

    return 0;
}

#endif//!__WLTWISE_H__
