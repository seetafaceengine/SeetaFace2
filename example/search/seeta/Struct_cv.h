#pragma once

#include <opencv2/core/core.hpp>

#include <seeta/CStruct.h>

namespace seeta
{
    namespace cv
    {
        // using namespace ::cv;
        class ImageData : public SeetaImageData {
        public:
            ImageData( const ::cv::Mat &mat )
                : cv_mat( mat.clone() ) {
                width = cv_mat.cols;
                height = cv_mat.rows;
                channels = cv_mat.channels();
                data = cv_mat.data;
            }

            ImageData( int width, int height, int channels = 3 )
                : cv_mat( height, width, CV_8UC( channels ) ) {
                width = cv_mat.cols;
                height = cv_mat.rows;
                channels = cv_mat.channels();
                data = cv_mat.data;
            }
            ImageData( const SeetaImageData &img )
                : cv_mat( img.height, img.width, CV_8UC( img.channels ), img.data ) {
                width = cv_mat.cols;
                height = cv_mat.rows;
                channels = cv_mat.channels();
                data = cv_mat.data;
            }
            ImageData()
                : cv_mat() {
                width = cv_mat.cols;
                height = cv_mat.rows;
                channels = cv_mat.channels();
                data = cv_mat.data;
            }
            bool empty() const {
                return cv_mat.empty();
            }
            operator ::cv::Mat() const {
                return cv_mat.clone();
            }
            ::cv::Mat toMat() const {
                return cv_mat.clone();
            }
        private:
            ::cv::Mat cv_mat;
        };
    }
}
