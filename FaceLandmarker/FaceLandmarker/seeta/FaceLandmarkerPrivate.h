#ifndef _SEETA_FACELANDMARKER_H_
#define _SEETA_FACELANDMARKER_H_

#include <memory>
#include <vector>

#include "CStruct.h"

#include <SeetaNetForward.h>



class FaceLandmarkerPrivate {
public:
    FaceLandmarkerPrivate( const char *model_path = nullptr, SeetaDevice device = SEETA_DEVICE_AUTO, int gpuid = 0 );
    ~FaceLandmarkerPrivate() {
        if( model_ != nullptr ) SeetaReleaseModel( model_ );
        model_ = nullptr;
        if( seeta_net_ != nullptr ) SeetaReleaseNet( seeta_net_ );
        seeta_net_ = nullptr;
    }

    void LoadModel( const char *model_path, SeetaDevice device, int gpuid = 0 );

    void LoadModel( const char *buffer, int len, SeetaDevice device, int gpuid = 0 );
    int LandmarkNum() const {
        return landmark_num_;
    }

    // need a cropped face
    bool PredictLandmark( const SeetaImageData &src_img, SeetaPointF *landmarks, int *masks ) const;
    bool PredictLandmark( const SeetaImageData &src_img, std::vector<SeetaPointF> &landmarks, std::vector<int> &masks ) const;

    bool PointDetectLandmarks( const SeetaImageData &src_img, const SeetaRect &face_info, SeetaPointF *landmarks, int *masks ) const; // interface used before

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int landmark_num_;
    float x_move_;
    float y_move_;
    float expand_size_;


    /////////////////////
    SeetaNet_Model *model_ = nullptr;
    SeetaNet_Net   *seeta_net_ = nullptr;
    SeetaNet_SharedParam *param_ = nullptr;
    SeetaNet_DEVICE_TYPE type_;
    int gpuid_ = 0;
    bool isLoadModel() const {
        return seeta_net_ != nullptr;
    }
    void ShowModelInputShape() const;

    // need a cropped face image with correct size
    // corresponding to the model input size
    // output is in range [0,1]
    bool Predict( const SeetaImageData &src_img, std::vector<SeetaPointF> &landmarks, std::vector<int> &masks ) const;

    static void CropFace( const unsigned char *src_img, int src_width, int src_height, int src_channels,
                          unsigned char *dst_img, int min_x, int min_y, int max_x, int max_y );

    static bool ResizeImage( const unsigned char *src_im, int src_width, int src_height, int src_channels,
                             unsigned char *dst_im, int dst_width, int dst_height, int dst_channels );
};

#endif
