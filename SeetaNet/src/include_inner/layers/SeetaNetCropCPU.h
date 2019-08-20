#ifndef _SEETANET_CROP_CPU_H_
#define _SEETANET_CROP_CPU_H_


#include "SeetaNetBaseLayer.h"

template <class T>
class SeetaNetCropCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetCropCPU();
    ~SeetaNetCropCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *p_seeta_net_resource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:

    std::vector<int> offsets_;

    int start_axis_;

    void crop_copy( const std::vector<SeetaNetFeatureMap<T>*> &bottom,
                    const std::vector<SeetaNetFeatureMap<T>*> &top,
                    const std::vector<int> &offsets,
                    std::vector<int> indices,
                    int cur_dim,
                    const T *src_data,
                    T *dest_data,
                    bool is_forward );
};

int crop_offset( const std::vector<int> &indices, std::vector<int> shape_ )
{
    int offset = 0;
    for( int i = 0; i < shape_.size(); ++i )
    {
        offset *= shape_[i];
        if( indices.size() > i )
        {
            if( indices[i] < 0 )
            {
                std::cout << "blob offset input error" << std::endl;
            }
            if( indices[i] > shape_[i] )
            {
                std::cout << "blob offset input error" << std::endl;
            }

            offset += indices[i];
        }
    }
    return offset;
}

template <class T>
SeetaNetCropCPU<T>::SeetaNetCropCPU()
{

}


template <class T>
SeetaNetCropCPU<T>::~SeetaNetCropCPU()
{

}


template<class T>
int SeetaNetCropCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    int start_axis = start_axis_;


    std::vector<int> new_shape( input_data_map[0]->data_shape );

    for( int i = 0; i < input_data_map[0]->data_shape.size(); ++i )
    {
        int new_size = input_data_map[0]->data_shape[i];
        if( i >= start_axis )
        {
            new_size = input_data_map[1]->data_shape[i];
        }
        new_shape[i] = new_size;
    }

    output_data_map[0]->data_shape = new_shape;

    std::vector<int> indices( output_data_map[0]->data_shape.size(), 0 );
    const T *bottom_data = input_data_map[0]->m_cpu.dataMemoryPtr();
    T *top_data = output_data_map[0]->m_cpu.dataMemoryPtr();

    crop_copy( input_data_map, output_data_map, offsets_, indices, 0, bottom_data, top_data, true );

    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    return 0;
}

template<class T>
int SeetaNetCropCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *p_seeta_net_resource )
{
    int bottom_index_0 = inputparam.bottom_index[0];
    int bottom_index_1 = inputparam.bottom_index[1];
    SeetaNetDataSize bottom_size_0 = p_seeta_net_resource->feature_vector_size[bottom_index_0];
    SeetaNetDataSize bottom_size_1 = p_seeta_net_resource->feature_vector_size[bottom_index_1];

    seeta::SeetaNet_CropParameter *msg = ( seeta::SeetaNet_CropParameter * )inputparam.msg.get();
    int start_axis = msg->axis;
    if( start_axis < 0 )
    {
        start_axis += int(bottom_size_0.data_dim.size());
    }
    start_axis_ = start_axis;

    offsets_ = std::vector<int>( bottom_size_0.data_dim.size(), 0 );
    std::vector<int> new_shape( bottom_size_0.data_dim );

    for( int i = 0; i < bottom_size_0.data_dim.size(); ++i )
    {
        int crop_offset = 0;
        int new_size = bottom_size_0.data_dim[i];
        if( i >= start_axis_ )
        {
            new_size = bottom_size_1.data_dim[i];
            if( msg->offset.size() == 1 )
            {
                // If only one offset is given, all crops have the same offset.
                crop_offset = msg->offset[0];
            }
            else
                if( msg->offset.size() > 1 )
                {
                    // For several offsets, the number of offsets must be equal to the
                    // number of dimensions to crop, that is dimensions after the axis.
                    crop_offset = msg->offset[i - start_axis_];
                }
            // Check that the crop and offset are within the dimension's bounds.
            if( bottom_size_0.data_dim[i] - crop_offset < bottom_size_1.data_dim[i] )
            {
                std::cout << "the crop for dimension " << i << " is out-of-bounds with "
                          << "size " << bottom_size_1.data_dim[i] << " and offset " << crop_offset;
            }

        }
        new_shape[i] = new_size;
        offsets_[i] = crop_offset;
    }
    this->top_data_size.resize( 1 );
    this->top_data_size[0].data_dim = new_shape;

    return 0;
}

template <typename Dtype>
void SeetaNetCropCPU<Dtype>::crop_copy( const std::vector<SeetaNetFeatureMap<Dtype>*> &bottom,
                                        const std::vector<SeetaNetFeatureMap<Dtype>*> &top,
                                        const std::vector<int> &offsets,
                                        std::vector<int> indices,
                                        int cur_dim,
                                        const Dtype *src_data,
                                        Dtype *dest_data,
                                        bool is_forward )
{
    if( cur_dim + 1 < top[0]->data_shape.size() )
    {
        // We are not yet at the final dimension, call copy recursively
        for( int i = 0; i < top[0]->data_shape[cur_dim]; ++i )
        {
            indices[cur_dim] = i;
            crop_copy( bottom, top, offsets, indices, cur_dim + 1,
                       src_data, dest_data, is_forward );
        }
    }
    else
    {
        // We are at the last dimensions, which is stored continously in memory
        for( int i = 0; i < top[0]->data_shape[cur_dim]; ++i )
        {
            // prepare index vector reduced(red) and with offsets(off)
            std::vector<int> ind_red( cur_dim, 0 );
            std::vector<int> ind_off( cur_dim + 1, 0 );
            for( int j = 0; j < cur_dim; ++j )
            {
                ind_red[j] = indices[j];
                ind_off[j] = indices[j] + offsets[j];
            }
            ind_off[cur_dim] = offsets[cur_dim];
            // do the copy
            if( is_forward )
            {
                seeta_copy( top[0]->data_shape[cur_dim],
                            src_data + crop_offset( ind_off, bottom[0]->data_shape ),
                            dest_data + crop_offset( ind_red, top[0]->data_shape ) );
            }
            else
            {
                // in the backwards pass the src_data is top_diff
                // and the dest_data is bottom_diff
                seeta_copy( top[0]->data_shape[cur_dim],
                            src_data + crop_offset( ind_red, top[0]->data_shape ),
                            dest_data + crop_offset( ind_off, bottom[0]->data_shape ) );
            }
        }
    }
}

#endif;
