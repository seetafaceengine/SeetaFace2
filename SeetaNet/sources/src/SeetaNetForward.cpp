#include "SeetaNetForward.h"
#include "ReadFromSeetaNetLayer.h"
#include "SeetaNet.h"
#include "include_inner/SeetaNetMacro.h"
#include "SeetaNetProto.h"


#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"


struct SeetaNet_Model
{

};

struct SeetaNet_Net
{

};

struct SeetaNet_SharedParam
{

};

int SeetaReadModelFromBuffer( const char *buffer, size_t buffer_length, struct SeetaNet_Model **pmodel )
{
    return SeetaNetReadModelFromBuffer( buffer, buffer_length, ( void ** )pmodel );
}
int SeetaReadAllContentFromFile( const char *file_name, char **pbuffer, int64_t *file_length )
{
    return ReadAllContentFromFile( file_name, pbuffer, *file_length );
}

void SeetaFreeBuffer( char *buffer )
{
    delete[] buffer;
}


int SeetaCreateNet( struct SeetaNet_Model *model, int max_batch_size, enum SeetaNet_DEVICE_TYPE process_device_type, struct SeetaNet_Net **pnet )
{
    return CreateNet( model, max_batch_size, process_device_type, ( void ** )pnet );
}

int SeetaCreateNetSharedParam( struct SeetaNet_Model *model, int max_batch_size, enum SeetaNet_DEVICE_TYPE process_device_type, struct SeetaNet_Net **pnet, struct SeetaNet_SharedParam **pparam )
{
    return CreateNetSharedParam( model, max_batch_size, process_device_type, ( void ** )pnet, ( void ** )pparam );
}


void SeetaKeepBlob( struct SeetaNet_Net *net, const char *blob_name )
{
    SeetaNetKeepBlob( net, blob_name );
}

void SeetaKeepNoBlob( struct SeetaNet_Net *net )
{
    SeetaNetKeepNoBlob( net );
}

void SeetaKeepAllBlob( struct SeetaNet_Net *net )
{
    SeetaNetKeepAllBlob( net );
}

int SeetaHasKeptBlob( struct SeetaNet_Net *net, const char *blob_name )
{
    return SeetaNetHasKeptBlob( net, blob_name );
}

struct SeetaNet_SharedParam *SeetaGetSharedParam( struct SeetaNet_Net *net )
{
    return reinterpret_cast<struct SeetaNet_SharedParam *>( GetNetSharedParam( net ) );
}

int SeetaRunNetChar( struct SeetaNet_Net *net, int counts, struct SeetaNet_InputOutputData *pinput_data )
{
    return RunNetChar( net, counts, pinput_data );
}

int SeetaRunNetFloat( struct SeetaNet_Net *net, int counts, struct SeetaNet_InputOutputData *pinput_data )
{
    return RunNetFloat( net, counts, pinput_data );
}

int SeetaGetFeatureMap( struct SeetaNet_Net *net, const char *blob_name, struct SeetaNet_InputOutputData *poutput_data )
{
    return SeetaNetGetFeatureMap( blob_name, net, poutput_data );
}

int SeetaGetAllFeatureMap( struct SeetaNet_Net *net, int *number, struct SeetaNet_InputOutputData **poutput_data )
{
    return SeetaNetGetAllFeatureMap( net, number, poutput_data );
}

void SeetaFreeAllFeatureMap( struct SeetaNet_Net *net, const struct SeetaNet_InputOutputData *poutput_data )
{
    SeetaNetFreeAllFeatureMap( net, poutput_data );
}

void SeetaFinalizeLibrary()
{
}

void SeetaReleaseNet( struct SeetaNet_Net *net )
{
    SeetaNetReleaseNet( ( void ** )&net );
}

void SeetaReleaseModel( struct SeetaNet_Model *model )
{
    SeetaNetReleaseModel( ( void ** )&model );
}

enum SeetaNet_DEVICE_TYPE SeetaDefaultDevice()
{
    return SEETANET_CPU_DEVICE;
}

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define SEETANET_VERSION GENERATE_VER((SEETANET_MAJOR_VERSION) (SEETANET_MINOR_VERSION) (SEETANET_SINOR_VERSION))

const char *SeetaLibraryVersionString()
{
    return SEETANET_VERSION;
}



int SeetaModelResetInput( struct SeetaNet_Model *model, int width, int height )
{
    return SeetaNetModelResetInput( model, width, height );
}

