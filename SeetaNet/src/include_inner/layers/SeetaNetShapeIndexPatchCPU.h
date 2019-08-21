#ifndef _SEETANET_SHAPEINDEXPATCH_CPU_H_
#define _SEETANET_SHAPEINDEXPATCH_CPU_H_

#include "SeetaNetBaseLayer.h"
#include "SeetaNetHypeShape.h"

template <class T>
class SeetaNetShapeIndexPatchCPU : public SeetaNetBaseLayer<T> {
public:
    SeetaNetShapeIndexPatchCPU();
    ~SeetaNetShapeIndexPatchCPU();

    int Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pdjNetResource );

    int Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map );
public:

    std::vector<int> m_origin_patch;
    std::vector<int> m_origin;

};


template <class T>
SeetaNetShapeIndexPatchCPU<T>::SeetaNetShapeIndexPatchCPU()
{

}


template <class T>
SeetaNetShapeIndexPatchCPU<T>::~SeetaNetShapeIndexPatchCPU()
{

}


template <class T>
int SeetaNetShapeIndexPatchCPU<T>::Init( seeta::SeetaNet_LayerParameter &inputparam, SeetaNetResource<T> *pNetResource )
{
	size_t bottom_length = inputparam.bottom_index.size();

    assert( bottom_length == 2 );
    this->bottom_data_size.resize( bottom_length );
    for( size_t i = 0; i < bottom_length; i++ )
    {
        int index = inputparam.bottom_index[i];
        this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
    }



    seeta::SeetaNet_ShapeIndexPatchLayer *msg = ( seeta::SeetaNet_ShapeIndexPatchLayer * )inputparam.msg.get();
    m_origin_patch.resize( msg->origin_patch.size() );
    for( size_t i = 0; i < m_origin_patch.size(); ++i )
    {
        m_origin_patch[i] = msg->origin_patch[i];
    }
    assert( m_origin_patch.size() == 2 );

    m_origin.resize( msg->origin.size() );
    for( size_t i = 0; i < m_origin.size(); ++i )
    {
        m_origin[i] = msg->origin[i];
    }
    assert( m_origin.size() == 2 );

    int landmarkx2 = this->bottom_data_size[1].data_dim[1];
    assert( ( landmarkx2 % 2 ) == 0 );

    int x_patch_h = int( m_origin_patch[0] * this->bottom_data_size[0].data_dim[2] / float( m_origin[0] ) + 0.5f );
    int x_patch_w = int( m_origin_patch[1] * this->bottom_data_size[0].data_dim[3] / float( m_origin[1] ) + 0.5f );

    this->top_data_size.resize( 1 );
    this->top_data_size[0].data_dim.resize( 4 );
    this->top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];
    this->top_data_size[0].data_dim[1] = this->bottom_data_size[0].data_dim[1];
    this->top_data_size[0].data_dim[2] = x_patch_h;
    this->top_data_size[0].data_dim[3] = int(landmarkx2 * 0.5 * x_patch_w);

    return 0;
}



template <class T>
int SeetaNetShapeIndexPatchCPU<T>::Process( std::vector<SeetaNetFeatureMap<T>*> input_data_map, std::vector<SeetaNetFeatureMap<T>*> &output_data_map )
{
    int feat_h = input_data_map[0]->data_shape[2];
    int feat_w = input_data_map[0]->data_shape[3];

    int landmarkx2 = this->bottom_data_size[1].data_dim[1];
    int x_patch_h = int( m_origin_patch[0] * this->bottom_data_size[0].data_dim[2] / float( m_origin[0] ) + 0.5f );
    int x_patch_w = int( m_origin_patch[1] * this->bottom_data_size[0].data_dim[3] / float( m_origin[1] ) + 0.5f );

    int feat_patch_h = x_patch_h;
    int feat_patch_w = x_patch_w;

    int num = input_data_map[0]->data_shape[0];
    int channels = input_data_map[0]->data_shape[1];

    const float r_h = ( feat_patch_h - 1 ) / 2.0f;
    const float r_w = ( feat_patch_w - 1 ) / 2.0f;
    const int landmark_num = int(landmarkx2 * 0.5);

    HypeShape pos_offset( {input_data_map[1]->data_shape[0], input_data_map[1]->data_shape[1]} );
    HypeShape feat_offset( {input_data_map[0]->data_shape[0], input_data_map[0]->data_shape[1],
                            input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3]
                           } );

    int nmarks = int( landmarkx2 * 0.5 );
    HypeShape out_offset( {input_data_map[0]->data_shape[0], input_data_map[0]->data_shape[1],
                           x_patch_h, nmarks, x_patch_w
                          } );

    T *buff = output_data_map[0]->m_cpu.dataMemoryPtr();
    const T *feat_data = input_data_map[0]->m_cpu.dataMemoryPtr();
    const T *pos_data  = input_data_map[1]->m_cpu.dataMemoryPtr();
    T zero = 0;
    for( int i = 0; i < landmark_num; i++ )
    {
        for( int n = 0; n < num; n++ )  // x1, y1, ..., xn, yn
        {
            // coordinate of the first patch pixel, scale to the feature map coordinate
            const int y = int( pos_data[pos_offset.to_index( {n, 2 * i + 1} )] * ( feat_h - 1 ) - r_h + 0.5f );
            const int x = int( pos_data[pos_offset.to_index( {n, 2 * i} )] * ( feat_w - 1 ) - r_w + 0.5f );

            for( int c = 0; c < channels; c++ )
            {
                for( int ph = 0; ph < feat_patch_h; ph++ )
                {
                    for( int pw = 0; pw < feat_patch_w; pw++ )
                    {
                        const int y_p = y + ph;
                        const int x_p = x + pw;
                        // set zero if exceed the img bound
                        if( y_p < 0 || y_p >= feat_h || x_p < 0 || x_p >= feat_w )
                            buff[out_offset.to_index( {n, c, ph, i, pw} )] = zero;
                        else
                            buff[out_offset.to_index( {n, c, ph, i, pw} )] =
                                feat_data[feat_offset.to_index( {n, c, y_p, x_p} )];
                    }
                }
            }
        }
    }

    //////////////////////////////////////
    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
    output_data_map[0]->data_shape.resize( 4 );
    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
    output_data_map[0]->data_shape[2] = x_patch_h;
    output_data_map[0]->data_shape[3] = int(landmarkx2 * 0.5 * x_patch_w);

    return 0;
}

#endif
