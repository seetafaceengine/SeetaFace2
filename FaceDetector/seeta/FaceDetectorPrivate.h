#ifndef _FACE_DETECTOR_PRIVATE_H_
#define _FACE_DETECTOR_PRIVATE_H_


#include <cstdint>
#include <vector>

#include "CFaceInfo.h"
#include "CStruct.h"


/** @class FaceDetectorPrivate FaceDetectorPrivate.h
*  @brief The face detector.
*/
class FaceDetectorPrivate {
public:

    class CoreSize {
    public:
        CoreSize() : width( -1 ), height( -1 ) {}
        CoreSize( int width, int height ) : width( width ), height( height ) {}

        int width;
        int height;
    };


    /**
     * \brief Construct detector
     * \param [in] model_path 
     * \note 
     */
    explicit FaceDetectorPrivate( const char *model_path );

    /**
     * \brief Construct detector
     * \param [in] model_path 
     * \param [in] device 
     */
    explicit FaceDetectorPrivate( const char *model_path, SeetaDevice device, int gpuid );

    /**
     * \brief Construct detector
     * \param [in] model_path 
     * \param [in] core_size size of detector core, more big mean more memory needed.
     * \note 
     */
    explicit FaceDetectorPrivate( const char *model_path, const CoreSize &core_size );

    /**
     * \briefConstruct detector
     * \param [in] model_path 
     * \param [in] core_size size of detector core, more big mean more memory needed.
     * \param [in] device 
     */
    explicit FaceDetectorPrivate( const char *model_path, const CoreSize &core_size, SeetaDevice device, int gpuid );

    ~FaceDetectorPrivate();

    /**
     * \brief Detect face
     * \param [in] img input image in BGR color in HWC format
     * \return list of detected face
     * \note This API not support thread calling, FaceDetectorPrivate object.
     * \seet VIPLFaceInfo, VIPLImageData
     */
    SeetaFaceInfoArray Detect( const SeetaImageData &img );

    /**
     * \brief Set min face size
     * \param [in] size scale like sqrt(H * W)
     * \note min value is 20
     */
    void SetMinFaceSize( int32_t size );

    /**
     * \brief 
     * \param [in] factor 
     * \note min value is 1.414
     */
    void SetImagePyramidScaleFactor( float factor );

    /**
     * \brief Set thresh of each stage of CascadeCNN
     * \param [in] thresh1 
     * \param [in] thresh2 
     * \param [in] thresh3 
     * \note suggest: 0.62, 0.47, 0.985
     */
    void SetScoreThresh( float thresh1, float thresh2, float thresh3 );

    /**
    * \brief Get min face size
    * \return size 
    */
    int32_t GetMinFaceSize() const;

    /**
    * \brief 
    * \return factor 
    */
    float GetImagePyramidScaleFactor() const;

    /**
     * \brief Get thresh of each stage of CascadeCNN
     * \param [out] thresh1 
     * \param [out] thresh2 
     * \param [out] thresh3 
     * \note set nullptr if no return value need
     */
    void GetScoreThresh( float *thresh1, float *thresh2, float *thresh3 ) const;

    float GetScoreThresh1() const;
    float GetScoreThresh2() const;
    float GetScoreThresh3() const;

    void SetScoreThresh1( float thresh1 );
    void SetScoreThresh2( float thresh2 );
    void SetScoreThresh3( float thresh3 );
    /**
     * \brief if stable face in video
     * \param stable 
     */
    void SetVideoStable( bool stable = true );

    /**
     * \brief 
     * \return 
     */
    bool GetVideoStable() const;

    CoreSize GetCoreSize() const;

private:
    FaceDetectorPrivate( const FaceDetectorPrivate &other ) = delete;
    const FaceDetectorPrivate &operator=( const FaceDetectorPrivate &other ) = delete;

private:
    void *impl_;
    std::vector<SeetaFaceInfo> m_pre_faces;
    static int m_threads;
};

#endif  // VIPL_FACE_DETECTOR_H_
