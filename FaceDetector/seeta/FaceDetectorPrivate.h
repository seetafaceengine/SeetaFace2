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
     * \brief 构造人脸检测器
     * \param [in] model_path 检测器路径
     * \note 默认会以 AUTO 模式使用计算设备
     */
    explicit FaceDetectorPrivate( const char *model_path );

    /**
     * \brief 构造人脸检测器
     * \param [in] model_path 检测器路径
     * \param [in] device 使用的计算设备
     */
    explicit FaceDetectorPrivate( const char *model_path, SeetaDevice device, int gpuid );

    /**
     * \brief 构造人脸检测器
     * \param [in] model_path 检测器路径
     * \note 默认会以 AUTO 模式使用计算设备
     */
    explicit FaceDetectorPrivate( const char *model_path, const CoreSize &core_size );

    /**
     * \brief 构造人脸检测器
     * \param [in] model_path 检测器路径
     * \param [in] device 使用的计算设备
     */
    explicit FaceDetectorPrivate( const char *model_path, const CoreSize &core_size, SeetaDevice device, int gpuid );

    ~FaceDetectorPrivate();

    /**
     * \brief 检测人脸
     * \param [in] img 输入图像，需要 RGB 彩色通道
     * \return 检测到的人脸（VIPLFaceInfo）数组
     * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 FaceDetectorPrivate 的对象分别调用检测函数
     * \seet VIPLFaceInfo, VIPLImageData
     */
    SeetaFaceInfoArray Detect( const SeetaImageData &img );

    /**
     * \brief 设置最小人脸
     * \param [in] size 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
     * \note 最下人脸为 20，小于 20 的值会被忽略
     */
    void SetMinFaceSize( int32_t size );

    /**
     * \brief 设置图像金字塔的缩放比例
     * \param [in] factor 缩放比例
     * \note 该值最小为 1.414，小于 1.414 的值会被忽略
     */
    void SetImagePyramidScaleFactor( float factor );

    /**
     * \brief 设置级联网路网络的三级阈值
     * \param [in] thresh1 第一级阈值
     * \param [in] thresh2 第二级阈值
     * \param [in] thresh3 第三级阈值
     * \note 默认推荐为：0.62, 0.47, 0.985
     */
    void SetScoreThresh( float thresh1, float thresh2, float thresh3 );

    /**
    * \brief 获取最小人脸
    * \return size 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
    */
    int32_t GetMinFaceSize() const;

    /**
    * \brief 获取图像金字塔的缩放比例
    * \return factor 缩放比例
    */
    float GetImagePyramidScaleFactor() const;

    /**
     * \brief 获取级联网路网络的三级阈值
     * \param [out] thresh1 第一级阈值
     * \param [out] thresh2 第二级阈值
     * \param [out] thresh3 第三级阈值
     * \note 可以设置为 nullptr，表示不取该值
     */
    void GetScoreThresh( float *thresh1, float *thresh2, float *thresh3 ) const;

    float GetScoreThresh1() const;
    float GetScoreThresh2() const;
    float GetScoreThresh3() const;

    void SetScoreThresh1( float thresh1 );
    void SetScoreThresh2( float thresh2 );
    void SetScoreThresh3( float thresh3 );
    /**
     * \brief 是否以稳定模式输出人脸检测结果
     * \param stable 是否稳定
     * \note 默认是不以稳定模型工作的
     * \note 只有在视频中连续跟踪时，才使用此方法
     */
    void SetVideoStable( bool stable = true );

    /**
     * \brief 获取当前是否是稳定工作模式
     * \return 是否稳定
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
