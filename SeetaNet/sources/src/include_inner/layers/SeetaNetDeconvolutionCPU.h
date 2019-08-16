#ifndef _SEETANET_DECONVOLUTION_CPU_H_
#define _SEETANET_DECONVOLUTION_CPU_H_


#include "SeetaNetBaseLayer.h"
#include <cfloat>

#include "SeetaNetMathCPU.h"

template <class T>
class SeetaNetDeconvolutionCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetDeconvolutionCPU();
    ~SeetaNetDeconvolutionCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *p_seeta_net_resource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:


    int count_out_dim( int start_axis, int end_axis, std::vector<int> input_shape );
    int Caculate( const int input, const int kernel, const int pad, const int stride, const int dilation, int &output );

    SeetaNetBlobCpu<T> weight_blobs_;
    SeetaNetBlobCpu<T> *m_p_kernel_blob;

    SeetaNetBlobCpu<T> bias;
    int m_stride_h;
    int m_stride_w;
    int m_pad_h;
    int m_pad_w;
    int m_dilation_h;
    int m_dilation_w;
    int m_kenerl_channels;
    int m_kernel_h;
    int m_kernel_w;
    int m_kenerl_channels_complement;
    int m_group_;

    int num_spatial_axes_;
    int channel_axis_;

    std::vector<int> dilation_data;
    std::vector<int> stride_data;
    std::vector<int>  pad_data;
    std::vector<int> kernel_shape_data;

    std::vector<int> output_shape_;
    std::vector<int> input_shape_;

    SeetaNetBlobCpu<T> col_buffer_;
    std::vector<int> col_buffer_shape_;
    std::vector<int> conv_input_shape_;
    int conv_out_channels_;
    int conv_in_channels_;

    int kernel_dim_;

    int bottom_dim_;
    int top_dim_;

    void conv_col2im_cpu( const T *col_buff, T *data );
    void BaseMulti( const T *output, const T *weights, T *input );
    int m_kenerl_number;


    int conv_out_spatial_dim_;

    int output_offset_;
    int col_offset_;
    int weight_offset_;
    std::vector<T> m_bias_value;




private:
    SeetaNetResource<T> *m_p_seeta_net_resource;

};

template <class T>
SeetaNetDeconvolutionCPU<T>::SeetaNetDeconvolutionCPU()
{

}


template <class T>
SeetaNetDeconvolutionCPU<T>::~SeetaNetDeconvolutionCPU()
{

}

template <class T>
int SeetaNetDeconvolutionCPU<T>::count_out_dim( int start_axis, int end_axis, std::vector<int> input_shape )
{
    int out_count = 1;
    for( int i = start_axis; i < end_axis; i++ )
    {
        out_count *= input_shape[i];
    }
    return out_count;
}

template<class T>
void SeetaNetDeconvolutionCPU<T>::BaseMulti( const T *output,
        const T *weights, T *input )
{
    T *col_buff = m_p_seeta_net_resource->col_buffer_.data();


    for( int g = 0; g < m_group_; ++g )
    {
        seeta_cpu_gemm<T>( seeta::blas::Trans, seeta::blas::NoTrans, kernel_dim_,
                           conv_out_spatial_dim_, conv_out_channels_ / m_group_,
                           ( T )1., weights + weight_offset_ * g, output + output_offset_ * g,
                           ( T )0., col_buff + col_offset_ * g );
    }

    conv_col2im_cpu( col_buff, input );
}

template<class T>
inline void SeetaNetDeconvolutionCPU<T>::conv_col2im_cpu( const T *col_buff, T *data )
{
    col2im_cpu<T>( col_buff, conv_in_channels_,
                   conv_input_shape_[1], conv_input_shape_[2],
                   m_kernel_h, m_kernel_w,
                   m_pad_h, m_pad_w,
                   m_stride_h, m_stride_w,
                   m_dilation_h, m_dilation_w, data );

}

template<class Dtype>
int SeetaNetDeconvolutionCPU<Dtype>::Caculate( const int input, const int kernel,
        const int pad, const int stride, const int dilation,
        int &output )
{
    output = ( input - 1 ) * stride - 2 * pad + dilation * ( kernel - 1 ) + 1;

    return 0;
}

template<class T>
int SeetaNetDeconvolutionCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    const T *weight = m_p_kernel_blob->dataMemoryPtr();
    for( int i = 0; i < input_data_map.size(); ++i )
    {
        std::vector<int> bottom_shape = input_data_map[i]->data_shape;
        bottom_dim_ = count_out_dim( channel_axis_, bottom_shape.size(), bottom_shape );

        int output_h;
        int output_w;
        Caculate( bottom_shape[2], m_kernel_h, m_pad_h, m_stride_h, m_dilation_h, output_h );
        Caculate( bottom_shape[3], m_kernel_w, m_pad_w, m_stride_w, m_dilation_w, output_w );

        conv_input_shape_[1] = output_h;
        conv_input_shape_[2] = output_w;

        int output_channel = m_kenerl_channels;

        output_data_map[i]->data_shape[0] = input_data_map[i]->data_shape[0];
        output_data_map[i]->data_shape[1] = output_channel;
        output_data_map[i]->data_shape[2] = output_h;
        output_data_map[i]->data_shape[3] = output_w;

        top_dim_ = count_out_dim( channel_axis_, output_data_map[i]->data_shape.size(), output_data_map[i]->data_shape );

        const T *bottom_data = input_data_map[i]->m_cpu.dataMemoryPtr();
        T *top_data = output_data_map[i]->m_cpu.dataMemoryPtr();

        conv_out_spatial_dim_ = input_data_map[i]->data_shape[2] * input_data_map[i]->data_shape[3];
        conv_out_channels_ = input_data_map[i]->data_shape[1];

        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / m_group_;
        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;

        for( int n = 0; n < input_data_map[i]->data_shape[0]; ++n )
        {
            BaseMulti( bottom_data + n * bottom_dim_, weight,
                       top_data + n * top_dim_ );

            if( !m_bias_value.empty() )
            {
                AddBiasBlob( output_data_map[i]->m_cpu, output_data_map[0]->data_shape, m_bias_value );
            }


        }
        output_data_map[i]->dwStorageType = DATA_CPU_WIDTH;
    }

    return 0;
}

template<class T>
int SeetaNetDeconvolutionCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *p_seeta_net_resource )
{
    this->m_layer_index = inputparam.layer_index;
    m_p_seeta_net_resource = p_seeta_net_resource;
    int bottom_index = inputparam.bottom_index[0];
    SeetaNetDataSize bottom_size = p_seeta_net_resource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize( 1 );
    this->bottom_data_size[0] = bottom_size;

    seeta::SeetaNet_ConvolutionParameter *msg = ( seeta::SeetaNet_ConvolutionParameter * )inputparam.msg.get();
    std::vector<int> shape;
    const seeta::SeetaNet_BlobShape &tmp_shape = msg->kernel_param.shape;

    for( int i = 0; i < tmp_shape.dim.size(); i++ )
    {
        shape.push_back( tmp_shape.dim[i] );
    }

    int index_key = this->m_layer_index;
    if( p_seeta_net_resource->m_shared_param->param_map.find( index_key ) != p_seeta_net_resource->m_shared_param->param_map.end() )
    {

    }
    else
    {
        SeetaNetBlobCpu<T> tmp_kernel_blob;

        p_seeta_net_resource->m_shared_param->param_map.insert( std::pair<int, SeetaNetBlobCpu<T>>( index_key, tmp_kernel_blob ) );
        p_seeta_net_resource->m_shared_param->param_map[index_key].Reshape( shape );

        T *temp_shared_kernel_value = p_seeta_net_resource->m_shared_param->param_map[index_key].dataMemoryPtr();
        for( int i = 0; i < p_seeta_net_resource->m_shared_param->param_map[index_key].count(); i++ )
        {
            float tmp_float_value = msg->kernel_param.data[i];
            if( tmp_float_value < FLT_EPSILON && -tmp_float_value < FLT_EPSILON ) tmp_float_value = 0;
            *temp_shared_kernel_value = tmp_float_value;
            temp_shared_kernel_value++;
        }
    }
    m_p_kernel_blob = &( p_seeta_net_resource->m_shared_param->param_map[index_key] );


    m_kenerl_number = msg->kernel_param.shape.dim[0];
    m_kenerl_channels = msg->kernel_param.shape.dim[1];

    if( 0 != this->bottom_data_size[0].data_dim[1] % m_kenerl_channels )
    {
        return -1;
    }

    m_kernel_h = msg->kernel_param.shape.dim[2];
    m_kernel_w = msg->kernel_param.shape.dim[3];

    m_group_ = msg->group;
    m_stride_h = msg->stride_height;
    m_stride_w = msg->stride_width;
    m_pad_h = msg->pad_height;
    m_pad_w = msg->pad_width;
    m_dilation_h = msg->dilation_height;
    m_dilation_w = msg->dilation_width;

    if( msg->bias_param.data.size() > 0 )
    {
        int temp_biasnum = msg->bias_param.data.size();

        for( int i = 0; i < temp_biasnum; i++ )
        {
            float temp_biasvalue = msg->bias_param.data[i];
            if( temp_biasvalue < FLT_EPSILON && -temp_biasvalue < FLT_EPSILON ) temp_biasvalue = 0;
            m_bias_value.push_back( temp_biasvalue );
        }
    }

    dilation_data.push_back( m_dilation_h );
    dilation_data.push_back( m_dilation_w );

    pad_data.push_back( m_pad_h );
    pad_data.push_back( m_pad_w );

    stride_data.push_back( m_stride_h );
    stride_data.push_back( m_stride_w );

    kernel_shape_data.push_back( m_kernel_h );
    kernel_shape_data.push_back( m_kernel_w );

    channel_axis_ = 1;
    const int first_spatial_axis = channel_axis_ + 1;
    num_spatial_axes_ = 4 - first_spatial_axis;


    input_shape_.push_back( this->bottom_data_size[0].data_dim[2] );
    input_shape_.push_back( this->bottom_data_size[0].data_dim[3] );

    for( int i = 0; i < num_spatial_axes_; ++i )
    {
        // i + 1 to skip channel axis
        const int input_dim = input_shape_[i];
        const int kernel_extent = dilation_data[i] * ( kernel_shape_data[i] - 1 ) + 1;
        const int output_dim = stride_data[i] * ( input_dim - 1 )
                               + kernel_extent - 2 * pad_data[i];
        output_shape_.push_back( output_dim );
    }
    std::vector<int> kernel_shape = m_p_kernel_blob->shape();
    kernel_dim_ = count_out_dim( 1, kernel_shape.size(), kernel_shape );

    col_buffer_shape_.push_back( kernel_dim_ * m_group_ );
    for( int i = 0; i < num_spatial_axes_; ++i )
    {
        col_buffer_shape_.push_back( input_shape_[i] );
    }

    m_p_seeta_net_resource->UpdateNetResourceMemory( col_buffer_shape_ );


    conv_input_shape_.resize( 3 );
    conv_input_shape_[0] = this->bottom_data_size[0].data_dim[1];
    conv_input_shape_[1] = this->bottom_data_size[0].data_dim[2];
    conv_input_shape_[2] = this->bottom_data_size[0].data_dim[3];

    conv_out_spatial_dim_ = this->bottom_data_size[0].data_dim[1] * this->bottom_data_size[0].data_dim[2] * this->bottom_data_size[0].data_dim[3];

    conv_out_channels_ = m_kenerl_number;
    weight_offset_ = conv_out_channels_ * kernel_dim_ / m_group_;

    conv_in_channels_ = m_kenerl_channels;

    int output_h;
    int output_w;
    Caculate( this->bottom_data_size[0].data_dim[2], m_kernel_h, m_pad_h, m_stride_h, m_dilation_h, output_h );
    Caculate( this->bottom_data_size[0].data_dim[3], m_kernel_w, m_pad_w, m_stride_w, m_dilation_w, output_w );

    this->top_data_size.resize( 1 );
    this->top_data_size[0].data_dim.resize( 4 );
    this->top_data_size[0].data_dim[2] = output_h;
    this->top_data_size[0].data_dim[3] = output_w;
    this->top_data_size[0].data_dim[1] = m_kenerl_channels;
    this->top_data_size[0].data_dim[0] = p_seeta_net_resource->max_batch_size;

    return 0;
}


#endif;
