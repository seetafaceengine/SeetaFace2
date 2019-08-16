#include "SeetaNetProto.h"
#include "SeetaNetParseProto.h"
#include "SeetaNetLayerType.h"

#include<iostream>

#include <exception>

#define READ_FIELD(tag, id, nret, buf, len, offset, field, fieldname)  \
    if(tag & id) { \
        nret = ::read(buf + offset, len - offset, field); \
        if(nret < 0) { \
            std::cout << "parse " << fieldname << " failed!" << std::endl; \
            throw std::logic_error("read field failed!"); \
        } \
        offset += nret; \
    }

#define READ_BLOBPROTO(tag, id, nret, buf, len, offset, field, fieldname)  \
    if(tag & id) { \
        nret = field.read(buf + offset, len - offset); \
        if(nret < 0) { \
            std::cout << "parse " << fieldname << " failed!" << std::endl; \
            throw std::logic_error("read blob field failed!"); \
        } \
        offset += nret; \
    }

#define WRITE_FIELD(tag, id, nret, buf, len, offset, field, fieldname)  \
    if( (tag & id) > 0 ) {  \
        nret = ::write(buf + offset, len - offset, field); \
        if(nret < 0) { \
            std::cout << "write " << fieldname << " failed" << std::endl; \
            throw std::logic_error("write field failed!"); \
        }  \
        offset += nret; \
    }

#define WRITE_ARRAY_FIELD(tag, id, nret, buf, len, offset, field, fieldname)  \
    if(field.size() > 0) { \
        tag |= id; \
        nret = ::write(buf + offset, len - offset, field); \
        if(nret < 0) { \
            std::cout << "write " << fieldname << " failed" << std::endl; \
            throw std::logic_error("write array field failed!"); \
        }  \
        offset += nret; \
    }

#define WRITE_STRING_FIELD(tag, id, nret, buf, len, offset, field, fieldname)  \
    if((tag & id) > 0) { \
        nret = ::write(buf + offset, len - offset, field); \
        if(nret < 0) { \
            std::cout << "write " << fieldname << " failed" << std::endl; \
            throw std::logic_error("write string field failed!"); \
        }  \
        offset += nret; \
    }

#define WRITE_BLOBPROTO(tag, id, nret, buf, len, offset, field, fieldname)  \
    if(field.data.size() > 0 || field.shape.dim.size() > 0) { \
        tag |= id; \
        nret = field.write(buf + offset, len - offset); \
        if(nret < 0) { \
            std::cout << "write " << fieldname << " failed" << std::endl; \
            throw std::logic_error("write blob field failed!");; \
        }  \
        offset += nret; \
    }


namespace seeta
{

    SeetaNet_BaseMsg::SeetaNet_BaseMsg()
    {
        tag = 0;
    }

    SeetaNet_BaseMsg::~SeetaNet_BaseMsg()
    {
    }

    int SeetaNet_BaseMsg::read_tag( const char *buf, int len )
    {
        int offset = ::read( buf, len, tag );
        if( offset < 0 )
        {
            std::cout << "read tag failed"  << std::endl;
            throw std::logic_error("read tag field failed!");;
        }

        if( ( tag & 0x80000000 ) > 0 )
        {
            std::cout << "tag is invalid!"  << std::endl;
            throw std::logic_error("tag is invalid!");
        }
        return offset;
    }

    int SeetaNet_BaseMsg::write_tag( char *buf, int len )
    {
        int offset = ::write( buf, len, tag );

        if( offset < 0 )
        {
            std::cout << "write tag failed"  << std::endl;
            throw std::logic_error("write tag failed!");
        }
        return offset;
    }

    /////////////////////////////////////

    SeetaNet_BlobShape::SeetaNet_BlobShape()
    {
    }

    SeetaNet_BlobShape::~SeetaNet_BlobShape()
    {
    }

    int SeetaNet_BlobShape::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );

        int nret = 0;

        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, dim, "SeetaNet_BlobShape dim" );
        return offset;
    }


    int SeetaNet_BlobShape::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_BlobShape failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_BlobShape failed!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_ARRAY_FIELD( tag, 0x00000001, nret, buf, len, offset, dim, "SeetaNet_BlobShape dim" );

        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    //////////////////////////////////////////////////////
    SeetaNet_BlobProto::SeetaNet_BlobProto()
    {
    }

    SeetaNet_BlobProto::~SeetaNet_BlobProto()
    {
    }

    int SeetaNet_BlobProto::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;

        if( tag & 0x00000001 )
        {
            nret = shape.read( buf + offset, len - offset );
            if( nret < 0 )
            {
                std::cout << "parse SeetaNet_BlobProto shape field failed!" << std::endl;
                throw std::logic_error("parse SeetaNet_BlobProto shape field failed!");;
            }
            offset += nret;
        }

        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, data, "SeetaNet_BlobProto data" );
        return offset;
    }


    int SeetaNet_BlobProto::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_BlobProto failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_BlobProto failed, the buf len is short!");;
        }

        int nret = 0;
        int offset = sizeof( tag );
        if( shape.dim.size() > 0 )
        {
            tag |= 0x00000001;
            nret = shape.write( buf + offset, len - offset );
            if( nret < 0 )
            {
                std::cout << "write SeetaNet_BlobProto shape field failed" << std::endl;
                throw std::logic_error("write SeetaNet_BlobProto shape field failed!"); 
            }
            offset += nret;
        }

        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, offset, data, "SeetaNet_BlobProto data" );

        write_tag( buf, sizeof( tag ) );
        return offset;
    }
    ///////////////////////////////////////////////////////////
    SeetaNet_PreluParameter::SeetaNet_PreluParameter()
    {
    }

    SeetaNet_PreluParameter::~SeetaNet_PreluParameter()
    {
    }


    int SeetaNet_PreluParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );

        int nret = 0;

        READ_BLOBPROTO( tag, 0x00000001, nret, buf, len, offset, param, "SeetaNet_PreluParameter param" );
        return offset;
    }


    int SeetaNet_PreluParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_PreluParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_PreluParameter failed, the buf len is short!"); 
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_BLOBPROTO( tag, 0x00000001, nret, buf, len, noffset, param, "SeetaNet_PreluParameter param" );

        write_tag( buf, sizeof( tag ) );
        return noffset;
    }


    ///////////////////////////////////////////////////////////
    SeetaNet_CropParameter::SeetaNet_CropParameter()
    {
        axis = 2;
    }


    SeetaNet_CropParameter::~SeetaNet_CropParameter()
    {
    }

    int SeetaNet_CropParameter::read( const char *buf, int len )
    {
        int noffset = read_tag( buf, len );
        int nret = 0;

        READ_FIELD( tag, 0x00000001, nret, buf, len, noffset, axis, "SeetaNet_CropParameter axis" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, noffset, offset, "SeetaNet_CropParameter offset" );
        return noffset;
    }


    int SeetaNet_CropParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_CropParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_CropParameter failed, the !");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, noffset, axis, "SeetaNet_CropParameter axis" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, noffset, offset, "SeetaNet_CropParameter offset" );

        write_tag( buf, sizeof( tag ) );
        return noffset;
    }
    ///////////////////////////////////////////////////////////

    SeetaNet_ConvolutionParameter::SeetaNet_ConvolutionParameter()
    {
        //dilation_height = 1;
        //dilation_width = 1;
        //num_output = 1;

        pad_height = 0;
        pad_width = 0;
        //set_pad_height(0);
        //set_pad_width(0);
        //kernel_height = 1;
        //kernel_width = 1;
        //stride_height = 1;
        //stride_width = 1;
        group = 1;
        axis = 1;
        force_nd_im2col = false;
        //set_group(1);
        //set_axis(1);
        //set_force_nd_im2col(false);
    }

    SeetaNet_ConvolutionParameter::~SeetaNet_ConvolutionParameter()
    {

    }


    int SeetaNet_ConvolutionParameter::read( const char *buf, int len )
    {
        int noffset = read_tag( buf, len );
        int nret = 0;

        READ_BLOBPROTO( tag, 0x00000001, nret, buf, len, noffset, bias_param, "SeetaNet_ConvolutionParameter bias_param" );
        READ_BLOBPROTO( tag, 0x00000002, nret, buf, len, noffset, kernel_param, "SeetaNet_ConvolutionParameter kernel_param" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, noffset, dilation_height, "SeetaNet_ConvolutionParameter dilation_height" );
        READ_FIELD( tag, 0x00000008, nret, buf, len, noffset, dilation_width, "SeetaNet_ConvolutionParameter dilation_width" );
        READ_FIELD( tag, 0x00000010, nret, buf, len, noffset, num_output, "SeetaNet_ConvolutionParameter num_output" );
        READ_FIELD( tag, 0x00000020, nret, buf, len, noffset, pad_height, "SeetaNet_ConvolutionParameter pad_height" );
        READ_FIELD( tag, 0x00000040, nret, buf, len, noffset, pad_width, "SeetaNet_ConvolutionParameter pad_width" );
        READ_FIELD( tag, 0x00000080, nret, buf, len, noffset, kernel_height, "SeetaNet_ConvolutionParameter kernel_height" );
        READ_FIELD( tag, 0x00000100, nret, buf, len, noffset, kernel_width, "SeetaNet_ConvolutionParameter kernel_width" );
        READ_FIELD( tag, 0x00000200, nret, buf, len, noffset, stride_height, "SeetaNet_ConvolutionParameter stride_height" );
        READ_FIELD( tag, 0x00000400, nret, buf, len, noffset, stride_width, "SeetaNet_ConvolutionParameter stride_width" );
        READ_FIELD( tag, 0x00000800, nret, buf, len, noffset, group, "SeetaNet_ConvolutionParameter group" );
        READ_FIELD( tag, 0x00001000, nret, buf, len, noffset, axis, "SeetaNet_ConvolutionParameter axis" );
        READ_FIELD( tag, 0x00002000, nret, buf, len, noffset, force_nd_im2col, "SeetaNet_ConvolutionParameter force_nd_im2col" );
        READ_FIELD( tag, 0x00004000, nret, buf, len, noffset, tf_padding, "SeetaNet_ConvolutionParameter tf_padding" );

        return noffset;
    }


    int SeetaNet_ConvolutionParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ConvolutionParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error(" failed!");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_BLOBPROTO( tag, 0x00000001, nret, buf, len, noffset, bias_param, "SeetaNet_ConvolutionParameter bias_param" );
        WRITE_BLOBPROTO( tag, 0x00000002, nret, buf, len, noffset, kernel_param, "SeetaNet_ConvolutionParameter kernel_param" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, noffset, dilation_height, "SeetaNet_ConvolutionParameter dilation_height" );
        WRITE_FIELD( tag, 0x00000008, nret, buf, len, noffset, dilation_width, "SeetaNet_ConvolutionParameter dilation_width" );
        WRITE_FIELD( tag, 0x00000010, nret, buf, len, noffset, num_output, "SeetaNet_ConvolutionParameter num_output" );
        WRITE_FIELD( tag, 0x00000020, nret, buf, len, noffset, pad_height, "SeetaNet_ConvolutionParameter pad_height" );
        WRITE_FIELD( tag, 0x00000040, nret, buf, len, noffset, pad_width, "SeetaNet_ConvolutionParameter pad_width" );
        WRITE_FIELD( tag, 0x00000080, nret, buf, len, noffset, kernel_height, "SeetaNet_ConvolutionParameter kernel_height" );
        WRITE_FIELD( tag, 0x00000100, nret, buf, len, noffset, kernel_width, "SeetaNet_ConvolutionParameter kernel_width" );
        WRITE_FIELD( tag, 0x00000200, nret, buf, len, noffset, stride_height, "SeetaNet_ConvolutionParameter stride_height" );
        WRITE_FIELD( tag, 0x00000400, nret, buf, len, noffset, stride_width, "SeetaNet_ConvolutionParameter stride_width" );
        WRITE_FIELD( tag, 0x00000800, nret, buf, len, noffset, group, "SeetaNet_ConvolutionParameter group" );
        WRITE_FIELD( tag, 0x00001000, nret, buf, len, noffset, axis, "SeetaNet_ConvolutionParameter axis" );
        WRITE_FIELD( tag, 0x00002000, nret, buf, len, noffset, force_nd_im2col, "SeetaNet_ConvolutionParameter force_nd_im2col" );
        WRITE_STRING_FIELD( tag, 0x00004000, nret, buf, len, noffset, tf_padding, "SeetaNet_ConvolutionParameter tf_padding" );

        write_tag( buf, sizeof( tag ) );
        return noffset;
    }
    //////////////////////////////////////////////////////


    SeetaNet_BatchNormliseParameter::SeetaNet_BatchNormliseParameter()
    {
    }

    SeetaNet_BatchNormliseParameter::~SeetaNet_BatchNormliseParameter()
    {
    }

    int SeetaNet_BatchNormliseParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;

        READ_BLOBPROTO( tag, 0x00000001, nret, buf, len, offset, mean_param, "SeetaNet_BatchNormliseParameter mean_param" );
        READ_BLOBPROTO( tag, 0x00000002, nret, buf, len, offset, covariance_param, "SeetaNet_BatchNormliseParameter covariance_param" );
        return offset;
    }


    int SeetaNet_BatchNormliseParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_BatchNormliseParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_BatchNormliseParameter failed, the buf len is short!");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_BLOBPROTO( tag, 0x00000001, nret, buf, len, noffset, mean_param, "SeetaNet_BatchNormliseParameter mean_param" );
        WRITE_BLOBPROTO( tag, 0x00000002, nret, buf, len, noffset, covariance_param, "SeetaNet_BatchNormliseParameter covariance_param" );
        write_tag( buf, sizeof( tag ) );
        return noffset;
    }


    ///////////////////////////////////////////////////////////

    SeetaNet_ScaleParameter::SeetaNet_ScaleParameter()
    {
    }

    SeetaNet_ScaleParameter::~SeetaNet_ScaleParameter()
    {
    }

    int SeetaNet_ScaleParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;

        READ_BLOBPROTO( tag, 0x00000001, nret, buf, len, offset, scale_param, "SeetaNet_ScaleParameter scale_param" );
        READ_BLOBPROTO( tag, 0x00000002, nret, buf, len, offset, bias_param, "SeetaNet_ScaleParameter bias_param" );
        return offset;
    }


    int SeetaNet_ScaleParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ScaleParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_ScaleParameter failed, the buf len is short!");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_BLOBPROTO( tag, 0x00000001, nret, buf, len, noffset, scale_param, "SeetaNet_ScaleParameter scale_param" );
        WRITE_BLOBPROTO( tag, 0x00000002, nret, buf, len, noffset, bias_param, "SeetaNet_ScaleParameter bias_param" );
        write_tag( buf, sizeof( tag ) );
        return noffset;
    }


    ///////////////////////////////////////////////////////
    SeetaNet_ConcatParameter::SeetaNet_ConcatParameter()
    {
        //set_concat_dim(1);
        //set_axis(1);
        concat_dim = 1;
        axis = 1;
    }

    SeetaNet_ConcatParameter::~SeetaNet_ConcatParameter()
    {
    }

    int SeetaNet_ConcatParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;

        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, concat_dim, "SeetaNet_ConcatParameter concat_dim" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, axis, "SeetaNet_ConcatParameter axis" );
        return offset;
    }


    int SeetaNet_ConcatParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ConcatParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_ConcatParameter failed, the buf len is short!");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, noffset, concat_dim, "SeetaNet_ConcatParameter concat_dim" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, noffset, axis, "SeetaNet_ConcatParameter axis" );
        write_tag( buf, sizeof( tag ) );
        return noffset;
    }


    ///////////////////////////////////////////////////////
    SeetaNet_EltwiseParameter::SeetaNet_EltwiseParameter()
    {
        //set_operation(SUM);
        //set_stable_prod_grad (true);
        operation = SUM;
        stable_prod_grad = true;
    }

    SeetaNet_EltwiseParameter::~SeetaNet_EltwiseParameter()
    {
    }

    int SeetaNet_EltwiseParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );

        int nret = 0;
        int32_t value = 0;

        if( tag & 0x00000001 )
        {
            nret = ::read( buf + offset, len - offset, value );
            if( nret < 0 )
            {
                std::cout << "parse SeetaNet_EltwiseParameter operation field failed!" << std::endl;
                throw std::logic_error("parse SeetaNeet_EltwiseParameter operation field failed!");
            }

            operation = ( EltwiseOp )value;
            offset += nret;
        }

        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, coeff, "SeetaNet_EltwiseParameter coeff" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, stable_prod_grad, "SeetaNet_EltwiseParameter stable_prod_grad" );
        return offset;
    }


    int SeetaNet_EltwiseParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_EltwiseParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_EltwiseParameter failed, the buf len is short!");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        int32_t value = ( int32_t )operation;

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, noffset, value, "SeetaNet_EltwiseParameter operation" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, noffset, coeff, "SeetaNet_EltwiseParameter coeff" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, noffset, stable_prod_grad, "SeetaNet_EltwiseParameter stable_prod_grad" );
        write_tag( buf, sizeof( tag ) );
        return noffset;
    }


    ///////////////////////////////////////////////////////
    SeetaNet_ExpParameter::SeetaNet_ExpParameter()
    {
        //set_base(0.0);
        //set_scale(1.0);
        //set_shift(0.0);
        base = 0.0;
        scale = 1.0;
        shift = 0.0;
    }

    SeetaNet_ExpParameter::~SeetaNet_ExpParameter()
    {
    }

    int SeetaNet_ExpParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, base, "SeetaNet_ExpParameter base" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, scale, "SeetaNet_ExpParameter scale" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, shift, "SeetaNet_ExpParameter shift" );

        return offset;
    }


    int SeetaNet_ExpParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ExpParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_ExpParameter failed, the buf len is short!");
        }

        int nret = 0;
        int noffset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, noffset, base, "SeetaNet_ExpParameter base" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, noffset, scale, "SeetaNet_ExpParameter scale" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, noffset, shift, "SeetaNet_ExpParameter shift" );
        write_tag( buf, sizeof( tag ) );
        return noffset;
    }



    ////////////////////////////////////////////////////////

    SeetaNet_MemoryDataParameterProcess::SeetaNet_MemoryDataParameterProcess()
    {
        //batch_size = 0;
        //channels = 0;
        //height = 0;
        //width  = 0;
        new_height = 0;
        new_width = 0;
        scale = 1;
        crop_size_height = 0;
        crop_size_width = 0;
        prewhiten = false;

        //set_new_height(0);
        //set_new_width(0);
        //set_scale(1);
        //set_crop_size_height(0);
        //set_crop_size_width(0);
        //channel_swaps = 0;
        //set_prewhiten(false);
    }

    SeetaNet_MemoryDataParameterProcess::~SeetaNet_MemoryDataParameterProcess()
    {
    }

    int SeetaNet_MemoryDataParameterProcess::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, batch_size, "SeetaNet_MemoryDataParameterProcess batch_size" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, channels, "SeetaNet_MemoryDataParameterProcess channels" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, height, "SeetaNet_MemoryDataParameterProcess height" );
        READ_FIELD( tag, 0x00000008, nret, buf, len, offset, width, "SeetaNet_MemoryDataParameterProcess width" );
        READ_FIELD( tag, 0x00000010, nret, buf, len, offset, new_height, "SeetaNet_MemoryDataParameterProcess new_height" );
        READ_FIELD( tag, 0x00000020, nret, buf, len, offset, new_width, "SeetaNet_MemoryDataParameterProcess new_width" );
        READ_FIELD( tag, 0x00000040, nret, buf, len, offset, scale, "SeetaNet_MemoryDataParameterProcess scale" );
        READ_BLOBPROTO( tag, 0x00000080, nret, buf, len, offset, mean_file, "SeetaNet_MemoryDataParameterProcess mean_file" );
        READ_FIELD( tag, 0x00000100, nret, buf, len, offset, mean_value, "SeetaNet_MemoryDataParameterProcess mean_value" );
        READ_FIELD( tag, 0x00000200, nret, buf, len, offset, crop_size_height, "SeetaNet_MemoryDataParameterProcess crop_size_height" );
        READ_FIELD( tag, 0x00000400, nret, buf, len, offset, crop_size_width, "SeetaNet_MemoryDataParameterProcess crop_sie_width" );
        READ_FIELD( tag, 0x00000800, nret, buf, len, offset, channel_swaps, "SeetaNet_MemoryDataParameterProcess channel_swaps" );
        READ_FIELD( tag, 0x00001000, nret, buf, len, offset, prewhiten, "SeetaNet_MemoryDataParameterProcess prewhiten" );
        return offset;
    }


    int SeetaNet_MemoryDataParameterProcess::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_MemoryDataParameterProcess failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_MemoryDataParameterProcess failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, batch_size, "SeetaNet_MemoryDataParameterProcess batch_size" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, channels, "SeetaNet_MemoryDataParameterProcess channels" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, height, "SeetaNet_MemoryDataParameterProcess height" );
        WRITE_FIELD( tag, 0x00000008, nret, buf, len, offset, width, "SeetaNet_MemoryDataParameterProcess width" );
        WRITE_FIELD( tag, 0x00000010, nret, buf, len, offset, new_height, "SeetaNet_MemoryDataParameterProcess new_height" );
        WRITE_FIELD( tag, 0x00000020, nret, buf, len, offset, new_width, "SeetaNet_MemoryDataParameterProcess new_width" );
        WRITE_FIELD( tag, 0x00000040, nret, buf, len, offset, scale, "SeetaNet_MemoryDataParameterProcess scale" );
        WRITE_BLOBPROTO( tag, 0x00000080, nret, buf, len, offset, mean_file, "SeetaNet_MemoryDataParameterProcess mean_file" );
        WRITE_ARRAY_FIELD( tag, 0x00000100, nret, buf, len, offset, mean_value, "SeetaNet_MemoryDataParameterProcess mean_value" );
        WRITE_FIELD( tag, 0x00000200, nret, buf, len, offset, crop_size_height, "SeetaNet_MemoryDataParameterProcess crop_size_height" );
        WRITE_FIELD( tag, 0x00000400, nret, buf, len, offset, crop_size_width, "SeetaNet_MemoryDataParameterProcess crop_sie_width" );
        WRITE_ARRAY_FIELD( tag, 0x00000800, nret, buf, len, offset, channel_swaps, "SeetaNet_MemoryDataParameterProcess channel_swaps" );
        WRITE_FIELD( tag, 0x00001000, nret, buf, len, offset, prewhiten, "SeetaNet_MemoryDataParameterProcess prewhiten" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }


    ////////////////////////////////////////////////////////
    SeetaNet_TransformationParameter::SeetaNet_TransformationParameter()
    {
        scale = 1;
        mirror = false;
        crop_height = 0;
        crop_width = 0;
        force_color = false;
        force_gray = false;
        //set_scale (1);
        //set_mirror(false);
        //set_crop_height (0);
        //set_crop_width(0);
        //set_force_color(false);
        //set_force_gray(false);
        //mean_value = 0.0;
    }

    SeetaNet_TransformationParameter::~SeetaNet_TransformationParameter()
    {

    }


    int SeetaNet_TransformationParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, scale, "SeetaNet_TransformationParameter scale" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, mirror, "SeetaNet_TransformationParameter mirror" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, crop_height, "SeetaNet_TransformationParameter crop_height" );
        READ_FIELD( tag, 0x00000008, nret, buf, len, offset, crop_width, "SeetaNet_TransformationParameter crop_width" );

        READ_FIELD( tag, 0x00000010, nret, buf, len, offset, mean_file, "SeetaNet_TransformationParameter mean_file" );
        READ_FIELD( tag, 0x00000020, nret, buf, len, offset, mean_value, "SeetaNet_TransformationParameter mean_value" );
        READ_FIELD( tag, 0x00000040, nret, buf, len, offset, force_color, "SeetaNet_TransformationParameter force_color" );
        READ_FIELD( tag, 0x00000080, nret, buf, len, offset, force_gray, "SeetaNet_TransformationParameter force_gray" );
        return offset;
    }


    int SeetaNet_TransformationParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_TransformationParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_TransformationParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, scale, "SeetaNet_TransformationParameter scale" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, mirror, "SeetaNet_TransformationParameter mirror" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, crop_height, "SeetaNet_TransformationParameter crop_height" );
        WRITE_FIELD( tag, 0x00000008, nret, buf, len, offset, crop_width, "SeetaNet_TransformationParameter crop_width" );

        WRITE_FIELD( tag, 0x00000010, nret, buf, len, offset, mean_file, "SeetaNet_TransformationParameter mean_file" );
        WRITE_FIELD( tag, 0x00000020, nret, buf, len, offset, mean_value, "SeetaNet_TransformationParameter mean_value" );
        WRITE_FIELD( tag, 0x00000040, nret, buf, len, offset, force_color, "SeetaNet_TransformationParameter force_color" );
        WRITE_FIELD( tag, 0x00000080, nret, buf, len, offset, force_gray, "SeetaNet_TransformationParameter force_gray" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }
    ////////////////////////////////////////////////////////////////////////


    SeetaNet_InnerProductParameter::SeetaNet_InnerProductParameter()
    {
        //num_output = 1;
        axis = 1;
        transpose = false;
        //set_axis(1);
        //set_transpose(false);
    }

    SeetaNet_InnerProductParameter::~SeetaNet_InnerProductParameter()
    {

    }


    int SeetaNet_InnerProductParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );

        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, num_output, "SeetaNet_InnerProductParameter num_output" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, axis, "SeetaNet_InnerProductParameter axis" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, transpose, "SeetaNet_InnerProductParameter transpose" );
        READ_BLOBPROTO( tag, 0x00000008, nret, buf, len, offset, bias_param, "SeetaNet_InnerProductParameter bias_param" );
        READ_BLOBPROTO( tag, 0x00000010, nret, buf, len, offset, Inner_param, "SeetaNet_InnerProductParameter Inner_param" );

        return offset;
    }

    int SeetaNet_InnerProductParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_InnerProductParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_InnerProductParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, num_output, "SeetaNet_InnerProductParameter num_output" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, axis, "SeetaNet_InnerProductParameter axis" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, transpose, "SeetaNet_InnerProductParameter transpose" );
        WRITE_BLOBPROTO( tag, 0x00000008, nret, buf, len, offset, bias_param, "SeetaNet_InnerProductParameter bias_param" );
        WRITE_BLOBPROTO( tag, 0x00000010, nret, buf, len, offset, Inner_param, "SeetaNet_InnerProductParameter Inner_param" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    ////////////////////////////////////////////////////////
    SeetaNet_LRNParameter::SeetaNet_LRNParameter()
    {
        //set_local_size(5);
        //set_alpha(1.0);
        //set_beta(0.75);
        //set_norm_region(ACROSS_CHANNELS);
        //set_k (1.0);
        local_size = 5;
        alpha = 1.0;
        beta = 0.75;
        norm_region = ACROSS_CHANNELS;
        k = 1.0;
    }

    SeetaNet_LRNParameter::~SeetaNet_LRNParameter()
    {

    }


    int SeetaNet_LRNParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        int32_t value = 0;//(int32_t)norm_region;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, local_size, "SeetaNet_LRNParameter local_size" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, alpha, "SeetaNet_LRNParameter alpha" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, beta, "SeetaNet_LRNParameter beta" );
        READ_FIELD( tag, 0x00000008, nret, buf, len, offset, value, "SeetaNet_LRNParameter norm_region" );
        READ_FIELD( tag, 0x00000010, nret, buf, len, offset, k, "SeetaNet_LRNParameter k" );
        norm_region = ( NormRegion )value;
        return offset;
    }

    int SeetaNet_LRNParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_LRNParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_LRNParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );
        int32_t value = ( int32_t )norm_region;
        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, local_size, "SeetaNet_LRNParameter local_size" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, alpha, "SeetaNet_LRNParameter alpha" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, beta, "SeetaNet_LRNParameter beta" );
        WRITE_FIELD( tag, 0x00000008, nret, buf, len, offset, value, "SeetaNet_LRNParameter norm_region" );
        WRITE_FIELD( tag, 0x00000010, nret, buf, len, offset, k, "SeetaNet_LRNParameter k" );

        write_tag( buf, sizeof( tag ) );
        return offset;
    }


    SeetaNet_PoolingParameter::SeetaNet_PoolingParameter()
    {
        //set_pool(MAX);
        //set_pad_height( 0);
        //set_pad_width (0);
        //kernel_height = 1;
        //kernel_width = 1;
        //stride_height = 1;
        //stride_width = 1;
        //set_global_pooling(false);
        //set_valid (false);
        pool = MAX;
        pad_height = 0;
        pad_width = 0;
        global_pooling = false;
        valid = false;

    }


    SeetaNet_PoolingParameter::~SeetaNet_PoolingParameter()
    {

    }


    int SeetaNet_PoolingParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        int32_t value = 0;//(int32_t)norm_region;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, value, "SeetaNet_PoolingParameter pool" );
        pool = ( PoolMethod )value;

        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, pad_height, "SeetaNet_PoolingParameter pad_height" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, pad_width, "SeetaNet_PoolingParameter pad_width" );
        READ_FIELD( tag, 0x00000008, nret, buf, len, offset, kernel_height, "SeetaNet_PoolingParameter kernel_height" );
        READ_FIELD( tag, 0x00000010, nret, buf, len, offset, kernel_width, "SeetaNet_PoolingParameter kernel_width" );
        READ_FIELD( tag, 0x00000020, nret, buf, len, offset, stride_height, "SeetaNet_PoolingParameter stride_height" );
        READ_FIELD( tag, 0x00000040, nret, buf, len, offset, stride_width, "SeetaNet_PoolingParameter stride_width" );
        READ_FIELD( tag, 0x00000080, nret, buf, len, offset, global_pooling, "SeetaNet_PoolingParameter global_pooling" );
        READ_FIELD( tag, 0x00000100, nret, buf, len, offset, valid, "SeetaNet_PoolingParameter valid" );
        READ_FIELD( tag, 0x00000200, nret, buf, len, offset, tf_padding, "SeetaNet_PoolingParameter tf_padding" );
        return offset;
    }

    int SeetaNet_PoolingParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_PoolingParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_PoolingParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );
        int32_t value = ( int32_t )pool;

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, value, "SeetaNet_PoolingParameter pool" );

        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, pad_height, "SeetaNet_PoolingParameter pad_height" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, pad_width, "SeetaNet_PoolingParameter pad_width" );
        WRITE_FIELD( tag, 0x00000008, nret, buf, len, offset, kernel_height, "SeetaNet_PoolingParameter kernel_height" );
        WRITE_FIELD( tag, 0x00000010, nret, buf, len, offset, kernel_width, "SeetaNet_PoolingParameter kernel_width" );
        WRITE_FIELD( tag, 0x00000020, nret, buf, len, offset, stride_height, "SeetaNet_PoolingParameter stride_height" );
        WRITE_FIELD( tag, 0x00000040, nret, buf, len, offset, stride_width, "SeetaNet_PoolingParameter stride_width" );
        WRITE_FIELD( tag, 0x00000080, nret, buf, len, offset, global_pooling, "SeetaNet_PoolingParameter global_pooling" );
        WRITE_FIELD( tag, 0x00000100, nret, buf, len, offset, valid, "SeetaNet_PoolingParameter valid" );
        WRITE_STRING_FIELD( tag, 0x00000200, nret, buf, len, offset, tf_padding, "SeetaNet_PoolingParameter tf_padding" );

        write_tag( buf, sizeof( tag ) );
        return offset;
    }


    /////////////////////////////////////////////////////////
    SeetaNet_PowerParameter::SeetaNet_PowerParameter()
    {
        //set_power(1.0);
        //set_scale(1.0);
        //set_shift(0.0);
        power = 1.0;
        scale = 1.0;
        shift = 0.0;
    }

    SeetaNet_PowerParameter::~SeetaNet_PowerParameter()
    {

    }

    int SeetaNet_PowerParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, power, "SeetaNet_PowerParameter power" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, scale, "SeetaNet_PowerParameter scale" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, shift, "SeetaNet_PowerParameter shift" );
        return offset;
    }

    int SeetaNet_PowerParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_PowerParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_PowerParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, power, "SeetaNet_PowerParameter power" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, scale, "SeetaNet_PowerParameter scale" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, shift, "SeetaNet_PowerParameter shift" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    //////////////////////////////////////////////////////
    SeetaNet_ReLUParameter::SeetaNet_ReLUParameter()
    {
        //set_negative_slope(0);
        negative_slope = 0;
    }

    SeetaNet_ReLUParameter::~SeetaNet_ReLUParameter()
    {
    }


    int SeetaNet_ReLUParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, negative_slope, "SeetaNet_ReLUParameter negative_slope" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, max, "SeetaNet_ReLUParameter max" );
        return offset;
    }

    int SeetaNet_ReLUParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ReLUParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_ReLUParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, negative_slope, "SeetaNet_ReLUParameter negative_slope" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, max, "SeetaNet_ReLUParameter max" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    //////////////////////////////////////////////////
    SeetaNet_SoftmaxParameter::SeetaNet_SoftmaxParameter()
    {
        //set_axis(1);
        axis = 1;
    }

    SeetaNet_SoftmaxParameter::~SeetaNet_SoftmaxParameter()
    {

    }


    int SeetaNet_SoftmaxParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, axis, "SeetaNet_SoftmaxParameter axis" );
        return offset;
    }

    int SeetaNet_SoftmaxParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_SoftmaxParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_SoftmaxParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, axis, "SeetaNet_SoftmaxParameter axis" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }



    ////////////////////////////////////////////
    SeetaNet_SliceParameter::SeetaNet_SliceParameter()
    {
        //set_axis(1);
        //set_slice_dim(1);
        axis = 1;
        slice_dim = 1;
    }

    SeetaNet_SliceParameter::~SeetaNet_SliceParameter()
    {

    }

    int SeetaNet_SliceParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );

        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, axis, "SeetaNet_SliceParameter axis" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, slice_point, "SeetaNet_SliceParameter slice_point" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, slice_dim, "SeetaNet_SliceParameter slice_dim" );
        return offset;
    }

    int SeetaNet_SliceParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_SliceParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_SliceParameter failed, the buf len is short");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_FIELD( tag, 0x00000001, nret, buf, len, offset, axis, "SeetaNet_SliceParameter axis" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, offset, slice_point, "SeetaNet_SliceParameter slice_point" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, slice_dim, "SeetaNet_SliceParameter slice_dim" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    ////////////////////////////////////////////
    SeetaNet_SigmoidParameter::SeetaNet_SigmoidParameter()
    {
    }

    SeetaNet_SigmoidParameter::~SeetaNet_SigmoidParameter()
    {
    }

    int SeetaNet_SigmoidParameter::read( const char *buf, int len )
    {
        return 0;
    }

    int SeetaNet_SigmoidParameter::write( char *buf, int len )
    {
        return 0;
    }

    //////////////////////////////////////////////////////////
    SeetaNet_SpaceToBatchNDLayer::SeetaNet_SpaceToBatchNDLayer()
    {

    }


    SeetaNet_SpaceToBatchNDLayer::~SeetaNet_SpaceToBatchNDLayer()
    {

    }


    int SeetaNet_SpaceToBatchNDLayer::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, block_shape, "SeetaNet_SpaceToBatchNDLayer block_shape" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, paddings, "SeetaNet_SpaceToBatchNDLayer paddings" );
        return offset;
    }

    int SeetaNet_SpaceToBatchNDLayer::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_SpaceToBatchNDLayer failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_SpaceToBatchNDLayer failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_ARRAY_FIELD( tag, 0x00000001, nret, buf, len, offset, block_shape, "SeetaNet_SpaceToBatchNDLayer block_shape" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, offset, paddings, "SeetaNet_SpaceToBatchNDLayer paddings" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    //////////////////////////////////////////////////////////
    SeetaNet_BatchToSpaceNDLayer::SeetaNet_BatchToSpaceNDLayer()
    {

    }


    SeetaNet_BatchToSpaceNDLayer::~SeetaNet_BatchToSpaceNDLayer()
    {

    }


    int SeetaNet_BatchToSpaceNDLayer::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, block_shape, "SeetaNet_BatchToSpaceNDLayer block_shape" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, crops, "SeetaNet_BatchToSpaceNDLayer crops" );
        return offset;
    }

    int SeetaNet_BatchToSpaceNDLayer::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_BatchToSpaceNDLayer failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_BatchToSpaceNDLayer failed, the buf len is short!");            
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_ARRAY_FIELD( tag, 0x00000001, nret, buf, len, offset, block_shape, "SeetaNet_BatchToSpaceNDLayer block_shape" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, offset, crops, "SeetaNet_BatchToSpaceNDLayer crops" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    ///////////////////////////////////////////////////////
    SeetaNet_ReshapeLayer::SeetaNet_ReshapeLayer()
    {

    }


    SeetaNet_ReshapeLayer::~SeetaNet_ReshapeLayer()
    {

    }


    int SeetaNet_ReshapeLayer::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, shape, "SeetaNet_ReshapeLayer shape" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, permute, "SeetaNet_ReshapeLayer permute" );
        return offset;
    }

    int SeetaNet_ReshapeLayer::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ReshapeLayer failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_ReshapeLayer failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_ARRAY_FIELD( tag, 0x00000001, nret, buf, len, offset, shape, "SeetaNet_ReshapeLayer shape" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, offset, permute, "SeetaNet_ReshapeLayer permute" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }


    ///////////////////////////////////////////////////////
    SeetaNet_RealMulLayer::SeetaNet_RealMulLayer()
    {

    }


    SeetaNet_RealMulLayer::~SeetaNet_RealMulLayer()
    {

    }


    int SeetaNet_RealMulLayer::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_BLOBPROTO( tag, 0x00000001, nret, buf, len, offset, y, "SeetaNet_RealMulLayer y" );
        return offset;
    }

    int SeetaNet_RealMulLayer::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_RealMulLayer failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_RealMulLayer failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_BLOBPROTO( tag, 0x00000001, nret, buf, len, offset, y, "SeetaNet_RealMulLayer y" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }

    ///////////////////////////////////////
    SeetaNet_ShapeIndexPatchLayer::SeetaNet_ShapeIndexPatchLayer()
    {

    }


    SeetaNet_ShapeIndexPatchLayer::~SeetaNet_ShapeIndexPatchLayer()
    {

    }


    int SeetaNet_ShapeIndexPatchLayer::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;
        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, origin_patch, "SeetaNet_ShapeIndexPatchLayer origin_patch" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, origin, "SeetaNet_ShapeIndexPatchLayer origin" );
        return offset;
    }

    int SeetaNet_ShapeIndexPatchLayer::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_ShapeIndexPatchLayer failed, the buf len is short!" << std::endl;
            throw std::logic_error("write SeetaNet_ShapeIndexPatchLayer failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_ARRAY_FIELD( tag, 0x00000001, nret, buf, len, offset, origin_patch, "SeetaNet_ShapeIndexPatchLayer origin_patch" );
        WRITE_ARRAY_FIELD( tag, 0x00000002, nret, buf, len, offset, origin, "SeetaNet_ShapeIndexPatchLayer origin" );
        write_tag( buf, sizeof( tag ) );
        return offset;
    }


    ///////////////////////////////////////
    SeetaNet_LayerParameter::SeetaNet_LayerParameter()
    {
    }

    SeetaNet_LayerParameter::~SeetaNet_LayerParameter()
    {
    }


    int SeetaNet_LayerParameter::read( const char *buf, int len )
    {
        int offset = read_tag( buf, len );
        int nret = 0;

        READ_FIELD( tag, 0x00000001, nret, buf, len, offset, name, "SeetaNet_LayerParameter name" );
        READ_FIELD( tag, 0x00000002, nret, buf, len, offset, type, "SeetaNet_LayerParameter type" );
        READ_FIELD( tag, 0x00000004, nret, buf, len, offset, layer_index, "SeetaNet_LayerParameter layer_index" );
        READ_FIELD( tag, 0x00000008, nret, buf, len, offset, bottom, "SeetaNet_LayerParameter bottom" );
        READ_FIELD( tag, 0x00000010, nret, buf, len, offset, top, "SeetaNet_LayerParameter top" );
        READ_FIELD( tag, 0x00000020, nret, buf, len, offset, top_index, "SeetaNet_LayerParameter top_index" );
        READ_FIELD( tag, 0x00000040, nret, buf, len, offset, bottom_index, "SeetaNet_LayerParameter bottom_index" );

        switch( ( seeta::Enum_SeetaNetLayerType )type )
        {
            case seeta::Enum_ConvolutionLayer:
                msg.reset( new SeetaNet_ConvolutionParameter() );
                break;
            case seeta::Enum_EltwiseLayer:
                msg.reset( new SeetaNet_EltwiseParameter() );
                break;
            case seeta::Enum_ConcatLayer:
                msg.reset( new SeetaNet_ConcatParameter() );
                break;
            case seeta::Enum_ExpLayer:
                msg.reset( new SeetaNet_ExpParameter() );
                break;
            case seeta::Enum_InnerProductLayer:
                msg.reset( new SeetaNet_InnerProductParameter() );
                break;
            case seeta::Enum_LRNLayer:
                msg.reset( new SeetaNet_LRNParameter() );
                break;
            case seeta::Enum_MemoryDataLayer:
                msg.reset( new SeetaNet_MemoryDataParameterProcess() );
                break;
            case seeta::Enum_PoolingLayer:
                msg.reset( new SeetaNet_PoolingParameter() );
                break;
            case seeta::Enum_PowerLayer:
                msg.reset( new SeetaNet_PowerParameter() );
                break;
            case seeta::Enum_ReLULayer:
                msg.reset( new SeetaNet_ReLUParameter() );
                break;
            case seeta::Enum_SoftmaxLayer:
                msg.reset( new SeetaNet_SoftmaxParameter() );
                break;
            case seeta::Enum_SliceLayer:
                msg.reset( new SeetaNet_SliceParameter() );
                break;
            //case seeta::Enum_TransformationLayer:
            //    msg = new SeetaNet_TransformationParameter();
            //    break;
            case seeta::Enum_BatchNormliseLayer:
                msg.reset( new SeetaNet_BatchNormliseParameter() );
                break;
            case seeta::Enum_ScaleLayer:
                msg.reset( new SeetaNet_ScaleParameter() );
                break;
            //case seeta::Enum_SplitLayer:

            case seeta::Enum_PreReLULayer:
                msg.reset( new SeetaNet_PreluParameter() );
                break;
            case seeta::Enum_DeconvolutionLayer:
                msg.reset( new SeetaNet_ConvolutionParameter() );
                break;
            case seeta::Enum_CropLayer:
                msg.reset( new SeetaNet_CropParameter() );
                break;
            case seeta::Enum_SigmoidLayer:
                msg.reset( new SeetaNet_SigmoidParameter() );
                break;
            // tf convert operator
            case seeta::Enum_SpaceToBatchNDLayer:
                msg.reset( new SeetaNet_SpaceToBatchNDLayer() );
                break;
            case seeta::Enum_BatchToSpaceNDLayer:
                msg.reset( new SeetaNet_BatchToSpaceNDLayer() );
                break;
            // tf reshape
            case seeta::Enum_ReshapeLayer:
                msg.reset( new SeetaNet_ReshapeLayer() );
                break;
            case seeta::Enum_RealMulLayer:
                msg.reset( new SeetaNet_RealMulLayer() );
                break;
            case seeta::Enum_ShapeIndexPatchLayer:
                msg.reset( new SeetaNet_ShapeIndexPatchLayer() );
                break;
            default:
                std::cout << "new layer type:" << type << std::endl;
                //exit(-1);
                break;
        }

        if( msg.get() != nullptr )
        {
            if( tag & 0x00000080 )
            {
                nret = msg->read( buf + offset, len - offset );
                offset += nret;
            }
        }
        return offset;
    }

    int SeetaNet_LayerParameter::write( char *buf, int len )
    {
        if( len < sizeof( tag ) )
        {
            std::cout << "write SeetaNet_LayerParameter failed, the buf len is short!" << std::endl;
            throw std::logic_error("write Seetanet_LayerParameter failed, the buf len is short!");
        }

        int nret = 0;
        int offset = sizeof( tag );

        WRITE_STRING_FIELD( tag, 0x00000001, nret, buf, len, offset, name, "SeetaNet_LayerParameter name" );
        WRITE_FIELD( tag, 0x00000002, nret, buf, len, offset, type, "SeetaNet_LayerParameter type" );
        WRITE_FIELD( tag, 0x00000004, nret, buf, len, offset, layer_index, "SeetaNet_LayerParameter layer_index" );
        WRITE_ARRAY_FIELD( tag, 0x00000008, nret, buf, len, offset, bottom, "SeetaNet_LayerParameter bottom" );
        WRITE_ARRAY_FIELD( tag, 0x00000010, nret, buf, len, offset, top, "SeetaNet_LayerParameter top" );
        WRITE_ARRAY_FIELD( tag, 0x00000020, nret, buf, len, offset, top_index, "SeetaNet_LayerParameter top_index" );
        WRITE_ARRAY_FIELD( tag, 0x00000040, nret, buf, len, offset, bottom_index, "SeetaNet_LayerParameter bottom_index" );

        if( msg.get() != nullptr )
        {
            tag |= 0x00000080;
            nret = msg->write( buf + offset, len - offset );
            offset += nret;
        }
        write_tag( buf, sizeof( tag ) );
        return offset;
    }




}




