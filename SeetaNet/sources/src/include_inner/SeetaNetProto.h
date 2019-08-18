#ifndef _SEETANET_PROTO_H_
#define _SEETANET_PROTO_H_

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

using std::vector;
using std::string;


namespace seeta
{

    class SeetaNet_BaseMsg {
    public:
        SeetaNet_BaseMsg();
        virtual ~SeetaNet_BaseMsg();
        virtual int read( const char *buf, int len ) = 0;
        virtual int write( char *buf, int len ) = 0;
        int read_tag( const char *buf, int len );
        int write_tag( char *buf, int len );

    public:
        uint32_t tag;

    };

    class SeetaNet_BlobShape : public SeetaNet_BaseMsg {
    public:
        SeetaNet_BlobShape();
        ~SeetaNet_BlobShape();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );


    public:
        vector<uint32_t> dim; //0x00000001
    };


    class SeetaNet_BlobProto : public SeetaNet_BaseMsg {
    public:
        SeetaNet_BlobProto();
        ~SeetaNet_BlobProto();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

    public:
        SeetaNet_BlobShape shape; //0x00000001
        vector<float> data;      //0x00000002
    };



    class SeetaNet_PreluParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_PreluParameter();
        ~SeetaNet_PreluParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        SeetaNet_BlobProto param;  //0x00000001
    };


    class SeetaNet_CropParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_CropParameter();
        ~SeetaNet_CropParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_axis( int32_t value ) {
            axis = value;
            tag |= 0x00000001;
        }
        bool has_axis() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }
    public:
        int32_t axis;                 //0x00000001
        vector<uint32_t> offset;      //0x00000002
    };



    class SeetaNet_ConvolutionParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_ConvolutionParameter();
        ~SeetaNet_ConvolutionParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_dilation_height( uint32_t value ) {
            dilation_height = value;
            tag |= 0x00000004;
        }
        bool has_dilation_height() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }

        void set_dilation_width( uint32_t value ) {
            dilation_width = value;
            tag |= 0x00000008;
        }
        bool has_dilation_width() const {
            return ( ( tag & 0x00000008 ) > 0 );
        }

        void set_num_output( uint32_t value ) {
            num_output = value;
            tag |= 0x00000010;
        }
        bool has_num_output() const {
            return ( ( tag & 0x00000010 ) > 0 );
        }

        void set_pad_height( uint32_t value ) {
            pad_height = value;
            tag |= 0x00000020;
        }
        bool has_pad_height() const {
            return ( ( tag & 0x00000020 ) > 0 );
        }

        void set_pad_width( uint32_t value ) {
            pad_width = value;
            tag |= 0x00000040;
        }
        bool has_pad_width() const {
            return ( ( tag & 0x00000040 ) > 0 );
        }

        void set_kernel_height( uint32_t value ) {
            kernel_height = value;
            tag |= 0x00000080;
        }
        bool has_kernel_height() const {
            return ( ( tag & 0x00000080 ) > 0 );
        }

        void set_kernel_width( uint32_t value ) {
            kernel_width = value;
            tag |= 0x00000100;
        }
        bool has_kernel_width() const {
            return ( ( tag & 0x00000100 ) > 0 );
        }

        void set_stride_height( uint32_t value ) {
            stride_height = value;
            tag |= 0x00000200;
        }
        bool has_stride_height() const {
            return ( ( tag & 0x00000200 ) > 0 );
        }
        void set_stride_width( uint32_t value ) {
            stride_width = value;
            tag |= 0x00000400;
        }
        bool has_stride_width() const {
            return ( ( tag & 0x00000400 ) > 0 );
        }

        void set_group( uint32_t value ) {
            group = value;
            tag |= 0x00000800;
        }
        bool has_group() const {
            return ( ( tag & 0x00000800 ) > 0 );
        }

        void set_axis( uint32_t value ) {
            axis = value;
            tag |= 0x00001000;
        }
        bool has_axis() const {
            return ( ( tag & 0x00001000 )  > 0 );
        }

        void set_force_nd_im2col( bool value ) {
            force_nd_im2col = value;
            tag |= 0x00002000;
        }
        bool has_force_nd_im2col() const {
            return ( ( tag & 0x00002000 ) > 0 );
        }

        void set_tf_padding( const std::string &value ) {
            tf_padding = value;
            tag |= 0x00004000;
        }
        bool has_tf_padding() const {
            return ( ( tag & 0x00004000 ) > 0 );
        }
    public:
        SeetaNet_BlobProto bias_param;                 //0x00000001
        SeetaNet_BlobProto kernel_param;               //0x00000002
        uint32_t dilation_height;                     //0x00000004
        uint32_t dilation_width;                      //0x00000008
        uint32_t num_output;                          //0x00000010
        uint32_t pad_height;                          //0x00000020
        uint32_t pad_width;                           //0x00000040
        uint32_t kernel_height;                       //0x00000080
        uint32_t kernel_width;                        //0x00000100
        uint32_t stride_height;                       //0x00000200
        uint32_t stride_width;                        //0x00000400
        uint32_t group;                               //0x00000800
        int32_t axis;                                 //0x00001000
        bool force_nd_im2col;                         //0x00002000

        // for tf padding setting, supporting VALID and SAME option
        string tf_padding;                            //0x00004000
    };

    class SeetaNet_BatchNormliseParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_BatchNormliseParameter();
        ~SeetaNet_BatchNormliseParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        SeetaNet_BlobProto mean_param;  //0x00000001
        SeetaNet_BlobProto covariance_param;  //0x00000002
    };

    class SeetaNet_ScaleParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_ScaleParameter();
        ~SeetaNet_ScaleParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        SeetaNet_BlobProto scale_param;  //0x00000001
        SeetaNet_BlobProto bias_param;  //0x00000002
    };

    class SeetaNet_ConcatParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_ConcatParameter();
        ~SeetaNet_ConcatParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_concat_dim( uint32_t value ) {
            concat_dim = value;
            tag |= 0x00000001;
        }
        bool has_concat_dim() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_axis( int32_t value ) {
            axis = value;
            tag |= 0x00000002;
        }
        bool has_axis() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }
    public:
        uint32_t concat_dim;  //0x00000001
        int32_t axis;         //0x00000002
    };


    class SeetaNet_EltwiseParameter : public SeetaNet_BaseMsg {
    public:
        enum EltwiseOp
        {
            PROD = 0,
            SUM = 1,
            MAX = 2
        };
        SeetaNet_EltwiseParameter();
        ~SeetaNet_EltwiseParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_operation( EltwiseOp value ) {
            operation = value;
            tag |= 0x00000001;
        }
        bool has_operation() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_stable_prod_grad( bool value ) {
            stable_prod_grad = value;
            tag |= 0x00000004;
        }
        bool has_stable_prod_grad() const {
            return ( ( tag & 0x00000004 ) > 0 ) ;
        }
    public:
        EltwiseOp operation;         //0x00000001
        vector<float> coeff;         //0x00000002
        bool stable_prod_grad;       //0x00000004
    };


    class SeetaNet_ExpParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_ExpParameter();
        ~SeetaNet_ExpParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_base( float value ) {
            base = value;
            tag |= 0x00000001;
        }
        bool has_base() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_scale( float value ) {
            scale = value;
            tag |= 0x00000002;
        }
        bool has_scale() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

        void set_shift( float value ) {
            shift = value;
            tag |= 0x00000004;
        }
        bool has_shift() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }
    public:
        float base;       //0x00000001
        float scale;      //0x00000002
        float shift;      //0x00000004
    };


    class SeetaNet_MemoryDataParameterProcess: public SeetaNet_BaseMsg {
    public:
        SeetaNet_MemoryDataParameterProcess();
        ~SeetaNet_MemoryDataParameterProcess();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_batch_size( uint32_t value ) {
            batch_size = value;
            tag |= 0x00000001;
        }
        bool has_batch_size() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_channels( uint32_t value ) {
            channels = value;
            tag |= 0x00000002;
        }
        bool has_channels() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

        void set_height( uint32_t value ) {
            height = value;
            tag |= 0x00000004;
        }
        bool has_height() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }

        void set_width( uint32_t value ) {
            width = value;
            tag |= 0x00000008;
        }
        bool has_width() const {
            return ( ( tag & 0x00000008 ) > 0 );
        }

        void set_new_height( uint32_t value ) {
            new_height = value;
            tag |= 0x00000010;
        }
        bool has_new_height() const {
            return ( ( tag & 0x00000010 ) > 0 );
        }

        void set_new_width( uint32_t value ) {
            new_width = value;
            tag |= 0x00000020;
        }
        bool has_new_width() const {
            return ( ( tag & 0x00000020 ) > 0 );
        }

        void set_scale( float value ) {
            scale = value;
            tag |= 0x00000040;
        }
        bool has_scale() const {
            return ( ( tag & 0x00000040 ) > 0 );
        }

        void set_crop_size_height( uint32_t value ) {
            crop_size_height = value;
            tag |= 0x00000200;
        }
        bool has_crop_size_height() const {
            return ( ( tag & 0x00000200 ) > 0 );
        }

        void set_crop_size_width( uint32_t value ) {
            crop_size_width = value;
            tag |= 0x00000400;
        }
        bool has_crop_size_width() const {
            return ( ( tag & 0x00000400 ) > 0 );
        }

        void set_prewhiten( bool value ) {
            prewhiten = value;
            tag |= 0x00001000;
        }
        bool has_prewhiten() const {
            return ( ( tag & 0x00001000 ) > 0 );
        }
    public:
        uint32_t batch_size;       //0x00000001
        uint32_t channels;         //0x00000002
        uint32_t height;           //0x00000004

        uint32_t width;            //0x00000008
        uint32_t new_height;       //0x00000010
        uint32_t new_width;        //0x00000020
        float    scale;            //0x00000040
        SeetaNet_BlobProto mean_file;           //0x00000080
        vector<float> mean_value;          //0x00000100
        uint32_t crop_size_height; //0x00000200
        uint32_t crop_size_width;  //0x00000400
        // for channels swap supprt push [2, 1, 0] means convert BGR2RGB
        vector<uint32_t> channel_swaps;    //0x00000800
        // for prewhiten action after above actions
        bool     prewhiten;        //0x00001000
    };

    class SeetaNet_TransformationParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_TransformationParameter();
        ~SeetaNet_TransformationParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_scale( float value ) {
            scale = value;
            tag |= 0x00000001;
        }
        bool has_scale() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_mirror( bool value ) {
            mirror = value;
            tag |= 0x00000002;
        }
        bool has_mirror() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

        void set_crop_height( uint32_t value ) {
            crop_height = value;
            tag |= 0x00000004;
        }
        bool has_crop_height() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }

        void set_crop_width( uint32_t value ) {
            crop_width = value;
            tag |= 0x00000008;
        }
        bool has_crop_width() const {
            return ( ( tag & 0x00000008 ) > 0 );
        }

        void set_mean_file( const string &value ) {
            mean_file = value;
            tag |= 0x00000010;
        }
        bool has_mean_file() const {
            return ( ( tag & 0x00000010 ) > 0 );
        }

        void set_mean_value( float value ) {
            mean_value = value;
            tag |= 0x00000020;
        }
        bool has_mean_value() const {
            return ( ( tag & 0x00000020 ) > 0 );
        }

        void set_force_color( bool value ) {
            force_color = value;
            tag |= 0x00000040;
        }
        bool has_force_color() const {
            return ( ( tag & 0x00000040 ) > 0 );
        }

        void set_force_gray( bool value ) {
            force_gray = value;
            tag |= 0x00000080;
        }
        bool has_force_gray() const {
            return ( ( tag & 0x00000080 ) > 0 );
        }
    public:
        float scale;                //0x00000001
        bool mirror;                //0x00000002
        uint32_t crop_height;       //0x00000004
        uint32_t crop_width;        //0x00000008

        string mean_file;           //0x00000010
        float mean_value;           //0x00000020
        bool force_color;           //0x00000040
        bool force_gray;            //0x00000080
    };


    class SeetaNet_InnerProductParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_InnerProductParameter();
        ~SeetaNet_InnerProductParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_num_output( uint32_t value ) {
            num_output = value;
            tag |= 0x00000001;
        }
        bool has_num_output() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_axis( int32_t value ) {
            axis = value;
            tag |= 0x00000002;
        }
        bool has_axis() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

        void set_transpose( bool value ) {
            transpose = value;
            tag |= 0x00000004;
        }
        bool has_transpose() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }
    public:
        uint32_t num_output;            //0x00000001
        int32_t axis;                   //0x00000002
        bool  transpose;                //0x00000004
        SeetaNet_BlobProto bias_param;   //0x00000008
        SeetaNet_BlobProto Inner_param;  //0x00000010
    };



    class SeetaNet_LRNParameter: public SeetaNet_BaseMsg {
    public:
        enum NormRegion
        {
            ACROSS_CHANNELS = 0,
            WITHIN_CHANNEL = 1
        };

        SeetaNet_LRNParameter();
        ~SeetaNet_LRNParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_local_size( uint32_t value ) {
            local_size = value;
            tag |= 0x00000001;
        }
        bool has_local_size() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_alpha( float value ) {
            alpha = value;
            tag |= 0x00000002;
        }
        bool has_alpha() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

        void set_beta( float value ) {
            beta = value;
            tag |= 0x00000004;
        }
        bool has_beta() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }

        void set_norm_region( NormRegion value ) {
            norm_region = value;
            tag |= 0x00000008;
        }
        bool has_norm_region() const {
            return ( ( tag & 0x00000008 ) > 0 );
        }

        void set_k( float value ) {
            k = value;
            tag |= 0x00000010;
        }
        bool has_k() const {
            return ( ( tag & 0x00000010 ) > 0 );
        }
    public:
        uint32_t local_size;      //0x00000001
        float alpha;              //0x00000002
        float  beta;              //0x00000004
        NormRegion norm_region;   //0x00000008
        float k;                  //0x00000010
    };


    class SeetaNet_PoolingParameter: public SeetaNet_BaseMsg {
    public:
        enum PoolMethod
        {
            MAX = 0,
            AVE = 1,
            STOCHASTIC = 2
        };

        SeetaNet_PoolingParameter();
        ~SeetaNet_PoolingParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );


        void set_pool( PoolMethod value ) {
            pool = value;
            tag |= 0x00000001;
        }
        bool has_pool() const {
            return ( ( tag & 0x00000001 )  > 0 );
        }

        void set_pad_height( uint32_t value ) {
            pad_height = value;
            tag |= 0x00000002;
        }
        bool has_pad_height() const {
            return ( ( tag & 0x00000002 )  > 0 );
        }

        void set_pad_width( uint32_t value ) {
            pad_width = value;
            tag |= 0x00000004;
        }
        bool has_pad_width() const {
            return ( ( tag & 0x00000004 )  > 0 );
        }

        void set_kernel_height( uint32_t value ) {
            kernel_height = value;
            tag |= 0x00000008;
        }
        bool has_kernel_height() const {
            return ( ( tag & 0x00000008 )  > 0 );
        }

        void set_kernel_width( uint32_t value ) {
            kernel_width = value;
            tag |= 0x00000010;
        }
        bool has_kernel_width() const {
            return ( ( tag & 0x00000010 ) > 0 );
        }

        void set_stride_height( uint32_t value ) {
            stride_height = value;
            tag |= 0x00000020;
        }
        bool has_stride_height() const {
            return ( ( tag & 0x00000020 ) > 0 );
        }
        void set_stride_width( uint32_t value ) {
            stride_width = value;
            tag |= 0x00000040;
        }
        bool has_stride_width() const {
            return ( ( tag & 0x00000040 ) > 0 );
        }

        void set_global_pooling( bool value ) {
            global_pooling = value;
            tag |= 0x00000080;
        }
        bool has_global_pooling() const {
            return ( ( tag & 0x00000080 ) > 0 );
        }

        void set_valid( bool value ) {
            valid = value;
            tag |= 0x00000100;
        }
        bool has_valid() const {
            return ( ( tag & 0x00000100 ) > 0 );
        }


        void set_tf_padding( const std::string &value ) {
            tf_padding = value;
            tag |= 0x00000200;
        }
        bool has_tf_padding() const {
            return ( ( tag & 0x00000200 ) > 0 );
        }
    public:

        PoolMethod pool;         //0x00000001
        uint32_t pad_height;     //0x00000002
        uint32_t pad_width;      //0x00000004
        uint32_t kernel_height;  //0x00000008
        uint32_t kernel_width;   //0x00000010
        uint32_t stride_height;  //0x00000020
        uint32_t stride_width;   //0x00000040
        bool global_pooling;     //0x00000080
        bool valid;              //0x00000100
        string tf_padding;       //0x00000200
    };


    class SeetaNet_PowerParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_PowerParameter();
        ~SeetaNet_PowerParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_power( float value ) {
            power = value;
            tag |= 0x00000001;
        }
        bool has_power() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_scale( float value ) {
            scale = value;
            tag |= 0x00000002;
        }
        bool has_scale() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }
        void set_shift( float value ) {
            shift = value;
            tag |= 0x00000004;
        }
        bool has_shift() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }
    public:
        float power;            //0x00000001
        float scale;            //0x00000002
        float shift;            //0x00000004
    };

    class SeetaNet_ReLUParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_ReLUParameter();
        ~SeetaNet_ReLUParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_negative_slope( float value ) {
            negative_slope = value;
            tag |= 0x00000001;
        }
        bool has_negative_slope() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_max( float value ) {
            max = value;
            tag |= 0x00000002;
        }
        bool has_max() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

    public:
        float negative_slope;   //0x00000001
        float max;              //0x00000002
    };

    class SeetaNet_SoftmaxParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_SoftmaxParameter();
        ~SeetaNet_SoftmaxParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_axis( int32_t value ) {
            axis = value;
            tag |= 0x00000001;
        }
        bool has_axis() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }
    public:
        int32_t axis;                    //0x00000001
    };

    class SeetaNet_SliceParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_SliceParameter();
        ~SeetaNet_SliceParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );

        void set_axis( int32_t value ) {
            axis = value;
            tag |= 0x00000001;
        }
        bool has_axis() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }
        void set_slice_dim( uint32_t value ) {
            slice_dim = value;
            tag |= 0x00000004;
        }
        bool has_slice_dim() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }
    public:
        int32_t axis;                    //0x00000001
        vector<uint32_t> slice_point;    //0x00000002
        uint32_t slice_dim;              //0x00000004
    };

    class SeetaNet_SigmoidParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_SigmoidParameter();
        ~SeetaNet_SigmoidParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
    };

    class SeetaNet_SplitParameter: public SeetaNet_BaseMsg {
    public:
        SeetaNet_SplitParameter();
        ~SeetaNet_SplitParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
    };

    class SeetaNet_SpaceToBatchNDLayer: public SeetaNet_BaseMsg {
    public:
        SeetaNet_SpaceToBatchNDLayer();
        ~SeetaNet_SpaceToBatchNDLayer();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        // size should be 2, like [2, 2]
        vector<int32_t> block_shape;     //0x00000001
        // size should be 2x2, like [1, 1, 2, 2]
        vector<int32_t> paddings;        //0x00000002
    };

    class SeetaNet_BatchToSpaceNDLayer: public SeetaNet_BaseMsg {
    public:
        SeetaNet_BatchToSpaceNDLayer();
        ~SeetaNet_BatchToSpaceNDLayer();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        // size should be 2, like [2, 2]
        vector<int32_t> block_shape;     //0x00000001
        // size should be 2x2, like [1, 1, 2, 2]
        vector<int32_t> crops;           //0x00000002
    };

    class SeetaNet_ReshapeLayer: public SeetaNet_BaseMsg {
    public:
        SeetaNet_ReshapeLayer();
        ~SeetaNet_ReshapeLayer();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        vector<int32_t> shape;      //0x00000001
        // for tf, NCHW -> NHWC
        vector<int32_t> permute;    //0x00000002
    };

    class SeetaNet_RealMulLayer: public SeetaNet_BaseMsg {
    public:
        SeetaNet_RealMulLayer();
        ~SeetaNet_RealMulLayer();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        SeetaNet_BlobProto y;        //0x00000001
    };

    class SeetaNet_ShapeIndexPatchLayer: public SeetaNet_BaseMsg {
    public:
        SeetaNet_ShapeIndexPatchLayer();
        ~SeetaNet_ShapeIndexPatchLayer();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
    public:
        //{h,w}
        vector<int32_t> origin_patch;      //0x00000001
        //{h,w}
        vector<int32_t> origin;            //0x00000002
    };


    class SeetaNet_LayerParameter : public SeetaNet_BaseMsg {
    public:
        SeetaNet_LayerParameter();
        ~SeetaNet_LayerParameter();
        virtual int read( const char *buf, int len );
        virtual int write( char *buf, int len );
        void set_name( const string &value ) {
            name = value;
            tag |= 0x00000001;
        }
        bool has_name() const {
            return ( ( tag & 0x00000001 ) > 0 );
        }

        void set_type( uint32_t value ) {
            type = value;
            tag |= 0x00000002;
        }
        bool has_type() const {
            return ( ( tag & 0x00000002 ) > 0 );
        }

        void set_layer_index( uint32_t value ) {
            layer_index = value;
            tag |= 0x00000004;
        }
        bool has_layer_index() const {
            return ( ( tag & 0x00000004 ) > 0 );
        }

    public:
        string name;                        //0x00000001
        uint32_t type;                      //0x00000002
        uint32_t layer_index;               //0x00000004
        vector<string> bottom;              //0x00000008
        vector<string> top;                 //0x00000010
        vector<uint32_t> top_index;         //0x00000020
        vector<uint32_t> bottom_index;      //0x00000040
        std::shared_ptr<SeetaNet_BaseMsg> msg;              //0x00000080
    };



}
#endif
