#ifndef _SEETANET_CONVOLUTIONCPU_H_
#define _SEETANET_CONVOLUTIONCPU_H_

#include <vector>
#include<iomanip>

#include "SeetaNetBaseLayer.h"
#include "SeetaNetCommon.h"
#include "SeetaNetFeatureMap.h"
#include "SeetaNetResource.h"
#include "SeetaNetCommonfuction.h"
#include "SeetaNetIm2Col.h"
#include <fstream>
#include <cfloat>

#include "SeetaNetMathCPU.h"

template<class T>
class SeetaNetConvolutionCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetConvolutionCPU();
    SeetaNetConvolutionCPU( const SeetaNetConvolutionCPU &m );
    ~SeetaNetConvolutionCPU();
    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *p_seeta_net_resource );
    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );


    int m_stride_h;
    int m_stride_w;
    int m_pad_h;
    int m_pad_w;
    int m_dilation_h;
    int m_dilation_w;
    int m_kenerl_channels;
    int m_kernel_h;
    int m_kernel_w;
    int m_group_;
    int m_kenerl_number;
    int kernel_dims_;
    std::vector<T> m_bias_value;

    std::vector<int> col_buffer_shape_;
    SeetaNetBlobCpu<T> *m_p_kernel_blob;
    int weight_offset_;
    int conv_out_spatial_dim_;
    int col_offset_;
    int output_offset_;

    std::string m_tf_padding;
    int m_tf_fake_padding_h = 0;
    int m_tf_fake_padding_w = 0;
    int m_tf_conv_shift_h = 0;
    int m_tf_conv_shift_w = 0;
private:
    SeetaNetResource<T> *m_p_seeta_net_resource;

private:
    inline void conv_im2col_cpu( const T *data, T *col_buff ) {
        shift_im2col_cpu( data, this->bottom_data_size[0].data_dim[1],
                          this->bottom_data_size[0].data_dim[2], this->bottom_data_size[0].data_dim[3],
                          m_kernel_h, m_kernel_w,
                          m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w,
                          m_tf_conv_shift_h, m_tf_conv_shift_w,
                          m_stride_h, m_stride_w,
                          m_dilation_h, m_dilation_w, col_buff );

    }
    int Caculate( const int height, const int width,
                  const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, int &output_h, int &output_w );

    inline void conv_im2col_cpu( SeetaNetFeatureMap<T> *input_map, T *data, T *col_buff ) {
        shift_im2col_cpu( data, input_map->data_shape[1],
                          input_map->data_shape[2], input_map->data_shape[3],
                          m_kernel_h, m_kernel_w,
                          m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w,
                          m_tf_conv_shift_h, m_tf_conv_shift_w,
                          m_stride_h, m_stride_w,
                          m_dilation_h, m_dilation_w, col_buff );

    }



};

template<class T>
int SeetaNetConvolutionCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *p_seeta_net_resource )
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

        p_seeta_net_resource->m_shared_param->param_map.insert( std::pair<int, SeetaNetBlobCpu<T>> ( index_key, tmp_kernel_blob ) );
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

    m_group_ = msg->group;
    m_stride_h = msg->stride_height;
    m_stride_w = msg->stride_width;
    m_pad_h = msg->pad_height;
    m_pad_w = msg->pad_width;
    m_dilation_h = msg->dilation_height;
    m_dilation_w = msg->dilation_width;
    if( msg->bias_param.data.size() > 0 )
    {
        auto temp_biasnum = msg->bias_param.data.size();

        for( size_t i = 0; i < temp_biasnum; i++ )
        {
            float temp_biasvalue = msg->bias_param.data[i];
            if( temp_biasvalue < FLT_EPSILON && -temp_biasvalue < FLT_EPSILON ) temp_biasvalue = 0;
            m_bias_value.push_back( temp_biasvalue );
        }
    }
    m_kernel_h = msg->kernel_height;
    m_kernel_w = msg->kernel_height;

    bool is_1x1_conv = m_kernel_h == 1 && m_kernel_w == 1 && m_pad_h == 0 && m_pad_w == 0 && m_stride_h == 1 && m_stride_w == 1;

    if( msg->has_tf_padding() )
    {
        m_tf_padding = msg->tf_padding;
    }

    int output_h;
    int output_w;
    Caculate( this->bottom_data_size[0].data_dim[2], this->bottom_data_size[0].data_dim[3], m_kernel_h, m_kernel_w, m_pad_h, m_pad_w, m_stride_h, m_stride_w, m_dilation_h, m_dilation_w, output_h, output_w );


    this->top_data_size.resize( 1 );
    this->top_data_size[0].data_dim.resize( 4 );
    this->top_data_size[0].data_dim[2] = output_h;
    this->top_data_size[0].data_dim[3] = output_w;
    this->top_data_size[0].data_dim[1] = m_kenerl_number;
    this->top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];

    kernel_dims_ = m_kernel_h * m_kernel_w * m_kenerl_channels;
    col_buffer_shape_.push_back( kernel_dims_ * m_group_ );
    col_buffer_shape_.push_back( output_h );
    col_buffer_shape_.push_back( output_w );


    if( !is_1x1_conv )
    {
        m_p_seeta_net_resource->UpdateNetResourceMemory( col_buffer_shape_ );
    }
    conv_out_spatial_dim_ = output_h * output_w;
    col_offset_ = kernel_dims_ * conv_out_spatial_dim_;
    weight_offset_ = m_kenerl_number * kernel_dims_ / m_group_;
    output_offset_ = this->top_data_size[0].data_dim[1] * conv_out_spatial_dim_ / m_group_;

    return 0;
}

template<class T>
int SeetaNetConvolutionCPU<T>::Caculate( const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
        int &output_h, int &output_w )
{
    if( m_tf_padding == "VALID" )
    {
        output_h = int(ceil( ( height + 2 * pad_h -
                           ( dilation_h * ( kernel_h - 1 ) ) ) / float( stride_h ) ));
        output_w = int(ceil( ( width + 2 * pad_w -
                           ( dilation_w * ( kernel_w - 1 ) ) ) / float( stride_w ) ));
    }
    else
        if( m_tf_padding == "SAME" )
        {
            output_h = int(ceil( ( height + 2 * pad_h ) / float( stride_h ) ));
            output_w = int(ceil( ( width + 2 * pad_w ) / float( stride_w ) ));

            int original_view_h = height + 2 * pad_h;
            int original_view_w = width + 2 * pad_w;

            int need_view_h = output_h * stride_h + kernel_h - 1;
            int need_view_w = output_w * stride_w + kernel_w - 1;

            m_tf_fake_padding_h = ( need_view_h - original_view_h ) / 2;
            m_tf_fake_padding_w = ( need_view_w - original_view_w ) / 2;

            int tf_need_view_h = ( output_h - 1 ) * stride_h + kernel_h;
            int tf_need_view_w = ( output_w - 1 ) * stride_w + kernel_w;

            m_tf_conv_shift_h = -m_tf_fake_padding_h + ( tf_need_view_h - original_view_h ) / 2;
            m_tf_conv_shift_w = -m_tf_fake_padding_w + ( tf_need_view_w - original_view_w ) / 2;
        }
        else
        {
            output_h = ( height + 2 * pad_h -
                         ( dilation_h * ( kernel_h - 1 ) + 1 ) ) / stride_h + 1;
            output_w = ( width + 2 * pad_w -
                         ( dilation_w * ( kernel_w - 1 ) + 1 ) ) / stride_w + 1;
        }

    return 0;
}

template<class T>
int SeetaNetConvolutionCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    T *output = output_data_map[0]->m_cpu.data();
    T  *input = input_data_map[0]->m_cpu.data();
    int num = input_data_map[0]->data_shape[0];
    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

    Caculate( input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3], m_kernel_h, m_kernel_w, m_pad_h, m_pad_w,
              m_stride_h, m_stride_w, m_dilation_h, m_dilation_w, output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3] );

    output_data_map[0]->data_shape[1] = m_kenerl_number;

    conv_out_spatial_dim_ = output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
    col_offset_ = kernel_dims_ * conv_out_spatial_dim_;

    int output_number_offset = output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];;
    int input_number_offset = input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];

    T *weights = m_p_kernel_blob->dataMemoryPtr();

    bool fuse_bias = !m_bias_value.empty();
    //if( fuse_bias )
    //{
    //    SetBiasBlob( output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value );
    //}

    bool is_1x1_conv = m_kernel_h == 1 && m_kernel_w == 1 && m_pad_h == 0 && m_pad_w == 0 && m_stride_h == 1 && m_stride_w == 1;

    int multi_number = m_kenerl_number / m_group_;
    for( int n = 0; n < num; ++n )
    {
        T *col_buff = nullptr;

        if( is_1x1_conv )
        {
            col_buff = input;
        }
        else
        {
            col_buff = m_p_seeta_net_resource->col_buffer_.data();
            conv_im2col_cpu( input_data_map[0], input, col_buff );
        }

        for( int g = 0; g < m_group_; g++ )
        {
            seeta_cpu_gemm<T>( seeta::blas::NoTrans, seeta::blas::NoTrans, multi_number, conv_out_spatial_dim_, kernel_dims_,
                               ( T )1., weights + weight_offset_ * g, col_buff + col_offset_ * g, ( T )0., output + output_offset_ * g );


        }
        output += output_number_offset;
        input += input_number_offset;

    }

    if( fuse_bias )
    {
        AddBiasBlob( output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value );
    }

    return 0;
}

template<class T>
SeetaNetConvolutionCPU<T>::~SeetaNetConvolutionCPU()
{
    m_p_seeta_net_resource = nullptr;
}
template<class T>
SeetaNetConvolutionCPU<T>::SeetaNetConvolutionCPU()
{
    m_p_seeta_net_resource = nullptr;
}



#endif
