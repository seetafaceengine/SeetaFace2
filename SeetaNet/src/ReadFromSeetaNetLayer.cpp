#include "ReadFromSeetaNetLayer.h"
#include "SeetaNetMemoryModel.h"

#include <string>
#include <vector>
#include <fstream>
#include <exception>
#include <iostream>
#include "SeetaNetParseProto.h"



int ReadAllContentFromFile( const char *inputfilename, char **ppbuffer, int64_t &file_length )
{
    std::ifstream fin( inputfilename, std::ios::binary | std::ios::in );
    if( !fin.is_open() )
    {
        return -1;
    }
    fin.seekg( 0, std::ios::end );
    file_length = fin.tellg();

    *ppbuffer = new char[file_length];
    fin.seekg( 0, std::ios::beg );
    fin.read( *ppbuffer, file_length);
    fin.close();

    return 0;
}


int SeetaNetReadModelFromBuffer( const char *buffer, size_t buffer_length, void **model )
{
    MemoryModel **tmp_model = ( MemoryModel ** )model;
    *tmp_model = new MemoryModel;
    if( buffer == nullptr )
    {
        return NULL_PTR;
    }

	auto ibuffer_length = int(buffer_length);

    int offset = read( buffer, ibuffer_length, ( *tmp_model )->vector_blob_names );
    offset += read( buffer + offset, ibuffer_length - offset, ( *tmp_model )->vector_layer_names );

    int32_t nlayers = 0;
    offset += read( buffer + offset, ibuffer_length - offset, nlayers );

    int index_layer = 0;
    int return_result = -1;

    for( int i = 0; i < nlayers; i++ )
    {
        return_result = -1;
        seeta::SeetaNet_LayerParameter   *output_param = new seeta::SeetaNet_LayerParameter;
        return_result = output_param->read( buffer + offset, ibuffer_length - offset );

        output_param->set_layer_index( index_layer );
        index_layer++;

        if( return_result >= 0 )
        {
            ( *tmp_model )->all_layer_params.push_back( output_param );
        }
        else
        {
            std::cout << "SeetaNetReadModelFromBuffer failed" << std::endl;
            delete( *tmp_model );
            throw std::logic_error("SeetanetReadModelFromBuffer failed!");
        }
        offset += return_result;
    }

    return 0;
}


int SeetaNetReleaseModel( void **model )
{
    MemoryModel **tmp_model = ( MemoryModel ** )model;
    for( int i = 0; i < ( *tmp_model )->all_layer_params.size(); i++ )
    {
        delete( *tmp_model )->all_layer_params[i];
    }
    ( *tmp_model )->all_layer_params.clear();
    ( *tmp_model )->vector_blob_names.clear();
    ( *tmp_model )->vector_layer_names.clear();

    delete *tmp_model;

    *tmp_model = nullptr;

    return 0;
}

int SeetaNetModelResetInput( void *model, int width, int height )
{
    MemoryModel *tmp_model = ( MemoryModel * )model;
    tmp_model->m_new_width = width;
    tmp_model->m_new_height = height;
    return 0;
}


