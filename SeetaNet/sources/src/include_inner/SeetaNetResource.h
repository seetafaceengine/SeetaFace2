#ifndef _SEETANET_RESOURCE_H_
#define _SEETANET_RESOURCE_H_

#include <vector>
#include <map>

#include "SeetaNetMacro.h"
#include "SeetaNetCommon.h"
#include "SeetaNetBlobCpu.h"

template<class T>
struct SeetaNetShareParam
{
    std::map<int, SeetaNetBlobCpu<T> > param_map;
    int m_refrence_counts = 0;
    int m_device; // type of SeetaNet_DEVICE_TYPE

    SeetaNetShareParam() {
    }

    ~SeetaNetShareParam() {

    }
};

template<class T>
struct SeetaNetResource
{
    int max_batch_size;

    SeetaNetShareParam<T> *m_shared_param;

    std::map<std::string, int> blob_name_map;
    std::vector<int> layer_type_vector;

    std::vector<SeetaNetDataSize> feature_vector_size;

    /* saving resized input */
    int m_new_width = -1;
    int m_new_height = -1;

    SeetaNetBlobCpu<T> col_buffer_;
    std::vector<int> col_buffer_shape_;

    int process_device_type;
    int process_max_batch_size;

    int current_process_size;

    int colbuffer_memory_size;

    int CaculateMemorySize( std::vector<int> shape_vector ) {
        int counts = 0;
        if( !shape_vector.empty() ) {
            counts = 1;
            for( int i = 0; i < shape_vector.size(); i++ ) {
                counts *= shape_vector[i];
            }
        }

        return counts;
    };

    int UpdateNetResourceMemory( std::vector<int> shape_vector ) {
        int new_memory_size = CaculateMemorySize( shape_vector );
        if( new_memory_size > colbuffer_memory_size ) {
            col_buffer_shape_ = shape_vector;
            colbuffer_memory_size = new_memory_size;

            col_buffer_.Reshape( shape_vector );
        }


        return 0;
    };

};





#endif
