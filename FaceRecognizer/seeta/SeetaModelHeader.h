#ifndef _SEETA_MODEL_HEADER_H
#define _SEETA_MODEL_HEADER_H

#include <iostream>
#include <cstdint>
#include <string>
#include <cstring>


#include "SeetaNetParseProto.h"

namespace seeta
{
    class FRModelHeader {
    public:
        int32_t feature_size;
        int32_t channels;
        int32_t width;
        int32_t height;
        std::string blob_name;

        /**
         * \brief
         * \param buffer The buffer reading header
         * \return The size read
         */
        int read( const char *buffer, size_t size ) {
            if( size < sizeof( int32_t ) * 5 ) {
                std::cout << "FRModelHeader parse failed" << std::endl;
                exit( -1 );
            }
            int offset = 0;
            memcpy( &feature_size, buffer + offset, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            memcpy( &channels, buffer + offset, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            memcpy( &width, buffer + offset, sizeof( int32_t ) );
            offset += sizeof( int32_t );
            memcpy( &height, buffer + offset, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            int32_t blob_name_size = 0;
            memcpy( &blob_name_size, buffer + offset, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            if( size < offset + blob_name_size ) {
                std::cout << "FRModelHeader parse blob_name failed" << std::endl;
                exit( -1 );
            }
            blob_name = std::string( buffer + offset, blob_name_size );
            return sizeof( int32_t ) * 5 + blob_name_size;
        }

        int read_ex( const char *buffer, size_t size ) {

            if( size < sizeof( int32_t ) * 5 ) {
                std::cout << "FRModelHeader parse failed" << std::endl;
                exit( -1 );
            }
            int offset = 0;
            offset += ::read( buffer + offset, int(size - offset), feature_size );
            offset += ::read( buffer + offset, int(size - offset), channels );
            offset += ::read( buffer + offset, int(size - offset), width );
            offset += ::read( buffer + offset, int(size - offset), height );
            offset += ::read( buffer + offset, int(size - offset), blob_name );
            return offset;
        }

        /**
         * \brief
         * \param buffer The buffer writing head
         * \return The size wrote
         */
        int write( char *buffer, size_t size ) const {

            if( size < sizeof( int32_t ) * 5 + blob_name.length() ) {
                std::cout << "FRModelHeader write failed" << std::endl;
                exit( -1 );
            }

            int offset = 0;
            memcpy( buffer + offset, &feature_size, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            memcpy( buffer + offset, &channels, sizeof( int32_t ) );
            offset += sizeof( int32_t );
            memcpy( buffer + offset, &width, sizeof( int32_t ) );
            offset += sizeof( int32_t );
            memcpy( buffer + offset, &height, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            auto blob_name_size = int(blob_name.length());
            memcpy( buffer + offset, &blob_name_size, sizeof( int32_t ) );
            offset += sizeof( int32_t );

            memcpy( buffer + offset, blob_name.data(), blob_name_size );
            offset += blob_name_size;

            return offset;
        }

        int write_ex( char *buffer, size_t size ) const {

            if( size < sizeof( int32_t ) * 5 + blob_name.length() ) {
                std::cout << "FRModelHeader write failed" << std::endl;
                exit( -1 );
            }

            int offset = 0;
            offset += ::write( buffer + offset, int(size - offset), feature_size );
            offset += ::write( buffer + offset, int(size - offset), channels );
            offset += ::write( buffer + offset, int(size - offset), width );
            offset += ::write( buffer + offset, int(size - offset), height );
            offset += ::write( buffer + offset, int(size - offset), blob_name );
            return offset;
        }
    };
}

#endif
