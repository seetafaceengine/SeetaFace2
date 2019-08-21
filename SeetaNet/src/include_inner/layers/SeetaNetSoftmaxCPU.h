#ifndef _SEETANET_SOFTMAX_CPU_H_
#define _SEETANET_SOFTMAX_CPU_H_

#include "SeetaNetBaseLayer.h"
#include "SeetaNetResource.h"
#include "SeetaNetCommonfuction.h"
#include "SeetaNetMathCPU.h"


template <class T>
class SeetaNetSoftMaxCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetSoftMaxCPU() {};
    ~SeetaNetSoftMaxCPU() {};

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource );
    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:
    int SoftMaxOperation_s( int number, SeetaNetBlobCpu<T> &inputData, SeetaNetBlobCpu<T> &outputData );
    int SoftMaxOperation_Axis2_s( int number, SeetaNetBlobCpu<T> &inputData, SeetaNetBlobCpu<T> &outputData );
private:
    int64_t axis;
    int32_t engine;
public:
    typedef int ( SeetaNetSoftMaxCPU<T>::*pDetailProcessFun )( int number, SeetaNetBlobCpu<T> &inputData, SeetaNetBlobCpu<T> &outputData );
    pDetailProcessFun m_pdetailProcessFun;

    int count_out_permute_dim( int start_axis, int end_axis, std::vector<int> input_shape );

    SeetaNetBlobCpu<T> max_value_blob;
    SeetaNetBlobCpu<T> sum_value_blob;

    int outer_num_;
    int inner_num_;
    int softmax_axis_;
    /// sum_multiplier is used to carry out sum using BLAS
    SeetaNetBlobCpu<T> sum_multiplier_;
    /// scale is an intermediate Blob to hold temporary results.
    SeetaNetBlobCpu<T> scale_;
};

template <class T>
int SeetaNetSoftMaxCPU<T>::count_out_permute_dim( int start_axis, int end_axis, std::vector<int> input_shape )
{
    int out_count = 1;
    for( int i = start_axis; i < end_axis; i++ )
    {
        out_count *= input_shape[i];
    }
    return out_count;
}

template <class T>
int SeetaNetSoftMaxCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
	size_t bottom_length = inputparam.bottom_index.size();
    this->bottom_data_size.resize( bottom_length );
    for( size_t i = 0; i < bottom_length; i++ )
    {
        int index = inputparam.bottom_index[i];
        this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
    }


    this->top_data_size.resize( 1 );
    this->top_data_size[0] = this->bottom_data_size[0];

    seeta::SeetaNet_SoftmaxParameter    *msg = ( seeta::SeetaNet_SoftmaxParameter * )inputparam.msg.get();
    axis = msg->axis;

    std::vector<int> bottom_shape;
    bottom_shape.push_back( pNetResource->max_batch_size );
    bottom_shape.push_back( this->bottom_data_size[0].data_dim[1] );
    bottom_shape.push_back( this->bottom_data_size[0].data_dim[2] );
    bottom_shape.push_back( this->bottom_data_size[0].data_dim[3] );

    std::vector<int> scale_dims = bottom_shape;
    scale_dims[axis] = 1;

    max_value_blob.Reshape( scale_dims );
    sum_value_blob.Reshape( scale_dims );

    if( 1 == axis )
    {
        m_pdetailProcessFun = &SeetaNetSoftMaxCPU<T>::SoftMaxOperation_s;
    }
    else
        if( 2 == axis )
        {
            m_pdetailProcessFun = &SeetaNetSoftMaxCPU<T>::SoftMaxOperation_Axis2_s;
        }
    return 0;
}

template <class T>
int SeetaNetSoftMaxCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    input_data_map[0]->TransFormDataIn();
    int softmax_axis_ = int(axis);

    int outer_num_ = count_out_permute_dim( 0, softmax_axis_, input_data_map[0]->data_shape );
    int inner_num_ = count_out_permute_dim( softmax_axis_ + 1, int(input_data_map[0]->data_shape.size()), input_data_map[0]->data_shape );

    std::vector<int> scale_dims = input_data_map[0]->data_shape;
    scale_dims[softmax_axis_] = 1;

    scale_.Reshape( scale_dims );

    std::vector<int> mult_dims( 1, input_data_map[0]->data_shape[softmax_axis_] );
    sum_multiplier_.Reshape( mult_dims );
    T *multiplier_data = sum_multiplier_.dataMemoryPtr();
    seeta_set( sum_multiplier_.count(), T( 1 ), multiplier_data );

    T *top_data = output_data_map[0]->m_cpu.dataMemoryPtr();
    T *scale_data = scale_.dataMemoryPtr();
    T *bottom_data = input_data_map[0]->m_cpu.dataMemoryPtr();

    int channels = input_data_map[0]->data_shape[softmax_axis_];
    int bottom_counts = count_out_permute_dim( 0, int(input_data_map[0]->data_shape.size()), input_data_map[0]->data_shape );

    int dim = bottom_counts / outer_num_;
    seeta_copy( bottom_counts, bottom_data, top_data );

    T *buffer = new T[inner_num_];

    for( int i = 0; i < outer_num_; ++i )
    {
        // initialize scale_data to the first plane
        seeta_copy( inner_num_, bottom_data + i * dim, scale_data );
        for( int j = 0; j < channels; j++ )
        {
            for( int k = 0; k < inner_num_; k++ )
            {
                scale_data[k] = std::max( scale_data[k],
                                          bottom_data[i * dim + j * inner_num_ + k] );
            }
        }
        // subtraction
        memset( buffer, 0, sizeof( T ) * inner_num_ );
        for( int j = 0; j < channels; j++ )
        {
            for( int k = 0; k < inner_num_; k++ )
            {
                int step = i * channels * inner_num_ + j * inner_num_ + k;
                top_data[step] = exp( top_data[step] - scale_data[k] );
                buffer[k] += top_data[step];
            }
        }

        // division
        for( int j = 0; j < channels; j++ )
        {
            seeta_div( inner_num_, top_data, buffer, top_data );
            top_data += inner_num_;
        }
    }

    delete [] buffer;

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
    output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
    output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];
    return 0;
}

template<typename T>
int SeetaNetSoftMaxCPU<T>::SoftMaxOperation_s( int number, SeetaNetBlobCpu<T> &inputData, SeetaNetBlobCpu<T> &outputData )
{
    SeetaNetBlobCpu<T> tmp_cube = inputData;
    std::vector<int> position_index;
    position_index.resize( 4 );
    std::vector<int> position_index1;
    position_index1.resize( 4 );
    position_index[1] = 0;
    position_index1[1] = 0;
    for( int in_number = 0; in_number < number; in_number++ )
    {
        position_index1[0] = in_number;
        position_index[0] = in_number;
        for( int r = 0; r < inputData.shape()[2]; r++ )
        {
            position_index[2] = r;
            for( int c = 0; c < inputData.shape()[3]; c++ )
            {
                position_index[3] = c;
                max_value_blob.data_at( position_index ) = inputData.data_at( position_index );
                sum_value_blob.data_at( position_index ) = 0;
            }
        }


        for( int s = 0; s < inputData.shape()[1]; s++ )
        {
            position_index1[1] = s;
            for( int r = 0; r < inputData.shape()[2]; r++ )
            {
                position_index[2] = r;
                position_index1[2] = r;
                for( int c = 0; c < inputData.shape()[3]; c++ )
                {
                    position_index[3] = c;
                    position_index1[3] = c;
                    max_value_blob.data_at( position_index ) = std::max( max_value_blob.data_at( position_index ), inputData.data_at( position_index1 ) );
                }
            }
        }
        for( int s = 0; s < inputData.shape()[1]; s++ )
        {
            position_index1[1] = s;
            for( int r = 0; r < inputData.shape()[2]; r++ )
            {
                position_index1[2] = r;
                position_index[2] = r;
                for( int c = 0; c < inputData.shape()[3]; c++ )
                {
                    position_index1[3] = c;
                    position_index[3] = c;
                    T value_tmp = inputData.data_at( position_index1 ) - max_value_blob.data_at( position_index );
                    outputData.data_at( position_index1 ) = exp( value_tmp );
                    sum_value_blob.data_at( position_index ) += outputData.data_at( position_index1 );
                }
            }
        }

        for( int s = 0; s < inputData.shape()[1]; s++ )
        {
            position_index1[1] = s;
            for( int r = 0; r < inputData.shape()[2]; r++ )
            {
                position_index1[2] = r;
                position_index[2] = r;
                for( int c = 0; c < inputData.shape()[3]; c++ )
                {
                    position_index1[3] = c;
                    position_index[3] = c;
                    outputData.data_at( position_index1 ) /= sum_value_blob.data_at( position_index );
                }
            }
        }
    }

    return 0;
};

template <class T>
int SeetaNetSoftMaxCPU<T>::SoftMaxOperation_Axis2_s( int number, SeetaNetBlobCpu<T> &inputData, SeetaNetBlobCpu<T> &outputData )
{
    std::vector<int> position_index;
    position_index.resize( 4 );
    std::vector<int> position_index1;
    position_index1.resize( 4 );
    for( int i = 0; i < 4; i++ )
    {
        position_index[1] = 0;
        position_index1[1] = 0;
    }

    for( int in_number = 0; in_number < number; in_number++ )
    {
        position_index[0] = in_number;
        position_index1[0] = in_number;
        for( int s = 0; s < inputData.shape()[1]; s++ )
        {
            position_index[1] = s;
            for( int c = 0; c < inputData.shape()[3]; c++ )
            {
                position_index[3] = c;
                max_value_blob.data_at( position_index ) = inputData.data_at( position_index );
                sum_value_blob.data_at( position_index ) = 0;
            }
        }

        position_index[2] = 0;

        for( int s = 0; s < inputData.shape()[1]; s++ )
        {
            position_index[1] = s;
            position_index1[1] = s;
            for( int r = 0; r < inputData.shape()[2]; r++ )
            {
                position_index1[2] = r;
                for( int c = 0; c < inputData.shape()[3]; c++ )
                {
                    position_index[3] = c;
                    position_index1[3] = c;
                    max_value_blob.data_at( position_index ) = std::max( max_value_blob.data_at( position_index ), inputData.data_at( position_index1 ) );
                }
            }
        }
        position_index[2] = 0;
        for( int s = 0; s < outputData.shape()[1]; s++ )
        {
            position_index[1] = s;
            position_index1[1] = s;
            for( int r = 0; r < outputData.shape()[2]; r++ )
            {
                position_index1[2] = r;
                for( int c = 0; c < outputData.shape()[3]; c++ )
                {
                    position_index[3] = c;
                    position_index1[3] = c;
                    T value_tmp = inputData.data_at( position_index1 ) - max_value_blob.data_at( position_index );
                    outputData.data_at( position_index1 ) = exp( value_tmp );
                    sum_value_blob.data_at( position_index ) += outputData.data_at( position_index1 );
                }
            }
        }
        position_index[2] = 0;
        for( int s = 0; s < outputData.shape()[1]; s++ )
        {
            position_index[1] = s;
            position_index1[1] = s;
            for( int r = 0; r < outputData.shape()[2]; r++ )
            {
                position_index1[2] = r;
                for( int c = 0; c < outputData.shape()[3]; c++ )
                {
                    position_index[3] = c;
                    position_index1[3] = c;
                    outputData.data_at( position_index1 ) /= sum_value_blob.data_at( position_index );
                }
            }
        }

    }


    return 0;
}


#endif //!_SOFTMAX_H__
