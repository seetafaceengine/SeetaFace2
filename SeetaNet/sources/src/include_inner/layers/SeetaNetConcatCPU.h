#ifndef _SEETANET_CONCAT_CPU_H_
#define _SEETANET_CONCAT_CPU_H_


#include "SeetaNetBaseLayer.h"


template <class T>
class SeetaNetConcatCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetConcatCPU();
    ~SeetaNetConcatCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:

    T m_scale_in;
    T m_scale_out;


private:
    int64_t concat_axis_;
    int64_t concat_axis_dims;
    std::vector<int64_t> concat_input_dim_vector;

    int64_t fixed_num_concats_;
    int64_t fixed_concat_input_size_;
};

template <class T>
SeetaNetConcatCPU<T>::SeetaNetConcatCPU()
{

}


template <class T>
SeetaNetConcatCPU<T>::~SeetaNetConcatCPU()
{

}


template <class T>
int SeetaNetConcatCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
    concat_input_dim_vector.clear();
    int return_result = 0;

    int bottom_length = inputparam.bottom_index.size();
    this->bottom_data_size.resize( bottom_length );
    for( size_t i = 0; i < bottom_length; i++ )
    {
        int index = inputparam.bottom_index[i];
        this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
    }

    seeta::SeetaNet_ConcatParameter *msg = ( seeta::SeetaNet_ConcatParameter * )inputparam.msg.get();
    concat_axis_ = msg->axis;

    int output_number = this->bottom_data_size[0].data_dim[0];
    int output_channel = this->bottom_data_size[0].data_dim[1];
    int output_height = this->bottom_data_size[0].data_dim[2];
    int output_width = this->bottom_data_size[0].data_dim[3];


    fixed_concat_input_size_ = 1;
    if( 1 == concat_axis_ )
    {
        concat_input_dim_vector.push_back( this->bottom_data_size[0].data_dim[1] );
        for( int i = 1; i < this->bottom_data_size.size(); i++ )
        {
            output_channel += this->bottom_data_size[i].data_dim[1];
            concat_input_dim_vector.push_back( this->bottom_data_size[i].data_dim[1] );
        }
        concat_axis_dims = output_channel;
        fixed_num_concats_ = 1;
        fixed_concat_input_size_ *= this->bottom_data_size[0].data_dim[2] * this->bottom_data_size[0].data_dim[3];
    }
    if( 2 == concat_axis_ )
    {
        concat_input_dim_vector.push_back( this->bottom_data_size[0].data_dim[2] );
        for( int i = 1; i < this->bottom_data_size.size(); i++ )
        {
            output_height += this->bottom_data_size[i].data_dim[2];
            concat_input_dim_vector.push_back( this->bottom_data_size[i].data_dim[2] );
        }
        concat_axis_dims = output_height;
        fixed_num_concats_ = this->bottom_data_size[0].data_dim[1];
        fixed_concat_input_size_ = this->bottom_data_size[0].data_dim[3];
    }
    if( 3 == concat_axis_ )
    {
        for( int i = 1; i < this->bottom_data_size.size(); i++ )
        {
            output_width += this->bottom_data_size[i].data_dim[3];
            concat_input_dim_vector.push_back( this->bottom_data_size[i].data_dim[3] );
        }
        concat_axis_dims = output_width;
        fixed_num_concats_ = this->bottom_data_size[0].data_dim[1] * this->bottom_data_size[0].data_dim[2];
        fixed_concat_input_size_ = 1;
    }

    this->top_data_size.resize( 1 );
    this->top_data_size[0].data_dim.resize( 4 );

    this->top_data_size[0].data_dim[0] = output_number;
    this->top_data_size[0].data_dim[1] = output_channel;
    this->top_data_size[0].data_dim[2] = output_height;
    this->top_data_size[0].data_dim[3] = output_width;

    return return_result;
}


template <class T>
int SeetaNetConcatCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    for( int i = 0; i < this->bottom_data_size.size(); i++ )
    {
        input_data_map[i]->TransFormDataIn();
    }

    std::vector<int> output_vector;
    output_vector.resize( 4 );

    output_vector[0] = input_data_map[0]->data_shape[0];
    output_vector[1] = input_data_map[0]->data_shape[1];
    output_vector[2] = input_data_map[0]->data_shape[2];
    output_vector[3] = input_data_map[0]->data_shape[3];

    int number_concat_all = 1;
    for( int i = 2; i < output_vector.size(); i++ )
    {
        number_concat_all *= output_vector[i];
    }

    for( int i = 1; i < input_data_map.size(); i++ )
    {
        output_vector[concat_axis_] += input_data_map[i]->data_shape[concat_axis_];
    }

    concat_axis_dims = output_vector[concat_axis_];

    fixed_num_concats_ = 1;
    for( int i = 1; i < concat_axis_; i++ )
    {
        fixed_num_concats_ *= output_vector[i];
    }

    fixed_concat_input_size_ = number_concat_all / ( fixed_num_concats_ * input_data_map[0]->data_shape[1] );

    T *top_data = output_data_map[0]->m_cpu.dataMemoryPtr();

    const int top_concat_axis = concat_axis_dims;

    int64_t num_concats_ = 1;
    for( int i = 0; i < concat_axis_; i++ )
    {
        num_concats_ *= input_data_map[0]->data_shape[i];
    }

    int64_t concat_input_size_ = 1;
    for( int i = concat_axis_ + 1; i < input_data_map[0]->data_shape.size(); i++ )
    {
        concat_input_size_ *= input_data_map[0]->data_shape[i];
    }

    int offset_concat_axis = 0;
    for( int i = 0; i < input_data_map.size(); ++i )
    {
        const T *bottom_data = input_data_map[i]->m_cpu.dataMemoryPtr();
        const int bottom_concat_axis = input_data_map[i]->data_shape[concat_axis_];
        for( int n = 0; n < num_concats_; ++n )
        {
            seeta_copy( bottom_concat_axis * concat_input_size_,
                        bottom_data + n * bottom_concat_axis * concat_input_size_,
                        top_data + ( n * top_concat_axis + offset_concat_axis )
                        * concat_input_size_ );
        }
        offset_concat_axis += bottom_concat_axis;
    }

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
    output_data_map[0]->data_shape = input_data_map[0]->data_shape;

    for( int i = 1; i < input_data_map.size(); ++i )
    {
        output_data_map[0]->data_shape[concat_axis_] += input_data_map[i]->data_shape[concat_axis_];
    }

    return 0;
};


#endif
