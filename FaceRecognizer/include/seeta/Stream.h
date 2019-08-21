#ifndef _SEETANET_STREAM_H_
#define _SEETANET_STREAM_H_

#include "CStream.h"
#include <cstring>
#include <cstdio>

#ifndef __SEETA_NOEXCEPT
    #if __cplusplus >= 201103L
        #define __SEETA_NOEXCEPT noexcept
    #else
        #define __SEETA_NOEXCEPT
    #endif
#endif

namespace seeta
{

    class FileNotAccess : public std::exception {
    public:
        using self = FileNotAccess;
        using supper = std::exception;

        FileNotAccess( const std::string &filename )
            : m_filename( filename )
            , m_msg( "Can not access \"" + filename + "\"" ) {
        }
        const char *what() const __SEETA_NOEXCEPT override {
            return m_msg.c_str();
        }
        const std::string &filename() const {
            return m_filename;
        }
    private:
        std::string m_filename;
        std::string m_msg;
    };

    class StreamWriter {
    public:
        using self = StreamWriter;
        virtual ~StreamWriter() = default;
        virtual size_t write( const char *data, size_t length ) = 0;
    };

    class StreamReader {
    public:
        using self = StreamReader;
        virtual ~StreamReader() = default;
        virtual size_t read( char *data, size_t length ) = 0;
    };

    class CStreamWriter : public StreamWriter {
    public:
        using self = CStreamWriter;
        using supper = StreamWriter;

        CStreamWriter( SeetaStreamWrite *writer, void *obj )
            : m_writer( writer ), m_obj( obj ) {
        }

        size_t write( const char *data, size_t length ) override {
            if( !m_writer ) return 0;
            return m_writer( m_obj, data, length );
        }

    private:
        SeetaStreamWrite *m_writer = nullptr;
        void *m_obj = nullptr;
    };

    class CStreamReader : public StreamReader {
    public:
        using self = CStreamReader;
        using supper = StreamReader;

        CStreamReader( SeetaStreamRead *reader, void *obj )
            : m_reader( reader ), m_obj( obj ) {
        }

        size_t read( char *data, size_t length ) override {
            if( !m_reader ) return 0;
            return m_reader( m_obj, data, length );
        }
    private:
        SeetaStreamRead *m_reader = nullptr;
        void *m_obj = nullptr;
    };

    class FileStream : public StreamWriter, public StreamReader {
    public:
        using self = FileStream;

        enum Mode
        {
            Input   = 0x1,
            Output  = 0x1 << 1,
            Binary  = 0x1 << 2,
        };

        FileStream();
        explicit FileStream( const std::string &path, int mode = Output ) {
            open( path, mode );
        }

        FileStream( FileStream &&other ) {
            std::swap( iofile, other.iofile );
        }

        const FileStream &operator=( FileStream &&other ) {
            std::swap( iofile, other.iofile );
            return *this;
        }

        virtual ~FileStream() {
            close();
        }

        bool open( const std::string &path, int mode = Output ) {
            close();
            std::string mode_str;
            if( mode | Input && mode | Output ) {
                mode_str += "a+";
            }
            else
                if( mode | Input ) {
                    mode_str += "r";
                }
                else {
                    mode_str += "w";
                }
            if( mode | Binary ) mode_str += "b";
            #if _MSC_VER >= 1600
            fopen_s( &iofile, path.c_str(), mode_str.c_str() );
            #else
            iofile = std::fopen( path.c_str(), mode_str.c_str() );
            #endif
            return iofile != nullptr;
        }

        void close() {
            if( iofile != nullptr ) std::fclose( iofile );
        }

        bool is_opened() const {
            return iofile != nullptr;
        }

        size_t write( const char *data, size_t length ) override {
            if( iofile == nullptr ) return 0;
            auto result = std::fwrite( data, 1, length, iofile );
            return size_t( result );
        }

        size_t read( char *data, size_t length ) override {
            if( iofile == nullptr ) return 0;
            auto result = std::fread( data, 1, length, iofile );
            return size_t( result );
        }

    private:
        FileStream( const FileStream &other ) = delete;
        const FileStream &operator=( const FileStream &other ) = delete;

        FILE *iofile = nullptr;
    };

    class FileWriter : public FileStream {
    public:
        using self = FileWriter;
        using supper = FileStream;

        FileWriter() {}
        explicit FileWriter( const std::string &path, int mode = Output )
            : FileStream( path, ( mode & ( !Input ) ) | Output ) {
        }

        bool open( const std::string &path, int mode = Output ) {
            return supper::open( path, ( mode & ( !Input ) ) | Output );
        }
    };
    class FileReader : public FileStream {
    public:
        using self = FileReader;
        using supper = FileStream;

        FileReader() {}
        explicit FileReader( const std::string &path, int mode = Input )
            : FileStream( path, ( mode & ( !Output ) ) | Input ) {
        }

        bool open( const std::string &path, int mode = Input ) {
            return supper::open( path, ( mode & ( !Output ) ) | Input );
        }
    };
}

#endif
