#include "seeta/FaceDetector.h"
#include <fstream>
#include <array>
#include <iostream>
#include "CStruct.h"
#include "FaceDetectorPrivate.h"

namespace seeta
{
    namespace v2
    {

        ////////////////////////////////////////////////////////////
        FaceDetector::FaceDetector( const SeetaModelSetting &setting )
        {
            m_impl = new FaceDetectorPrivate( setting.model[0], setting.device, setting.id );
        }

        FaceDetector::FaceDetector( const SeetaModelSetting &setting, int core_width, int core_height )
        {
            FaceDetectorPrivate::CoreSize coresize( core_width, core_height );
            m_impl = new FaceDetectorPrivate( setting.model[0], coresize, setting.device, setting.id );
        }
        FaceDetector::~FaceDetector()
        {
            FaceDetectorPrivate *ptr = ( FaceDetectorPrivate * )m_impl;
            delete ptr;
            m_impl = nullptr;
        }

        SeetaFaceInfoArray FaceDetector::detect( const SeetaImageData &image )  const
        {
            FaceDetectorPrivate *ptr = ( FaceDetectorPrivate * )m_impl;
            return ptr->Detect( image );
        }

        void FaceDetector::set( FaceDetector::Property property, double value )
        {
            FaceDetectorPrivate *ptr = ( FaceDetectorPrivate * )m_impl;
            std::cout << "property:" << property << ", value:" << value << std::endl;
            switch( property )
            {
                default:
                    break;
                case FaceDetector::PROPERTY_THRESHOLD1:
                {
                    ptr->SetScoreThresh1( float( value ) );
                    break;
                }
                case FaceDetector::PROPERTY_THRESHOLD2:
                {
                    ptr->SetScoreThresh2( float( value ) );
                    break;
                }
                case FaceDetector::PROPERTY_THRESHOLD3:
                {
                    ptr->SetScoreThresh3( float( value ) );
                    break;
                }
                case FaceDetector::PROPERTY_MIN_FACE_SIZE:
                    ptr->SetMinFaceSize( int32_t( value ) );
                    break;
            }

        }

        double FaceDetector::get( FaceDetector::Property property ) const
        {
            FaceDetectorPrivate *ptr = ( FaceDetectorPrivate * )m_impl;
            switch( property )
            {
                default:
                    return 0;
                case FaceDetector::PROPERTY_THRESHOLD1:
                    return ptr->GetScoreThresh1();

                case FaceDetector::PROPERTY_THRESHOLD2:
                    return ptr->GetScoreThresh2();
                case FaceDetector::PROPERTY_THRESHOLD3:
                    return ptr->GetScoreThresh3();

                case FaceDetector::PROPERTY_MIN_FACE_SIZE:
                    return ptr->GetMinFaceSize();
            }

        }
    }
}

