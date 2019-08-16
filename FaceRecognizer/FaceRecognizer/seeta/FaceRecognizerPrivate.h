#ifndef _SEETA_FACE_RECOGNIZER_H
#define _SEETA_FACE_RECOGNIZER_H



#include "Struct.h"
#include <vector>



class FaceRecognizerModel {
public:
    friend class FaceRecognizerPrivate;

    FaceRecognizerModel( const char *model_path, int device );
    ~FaceRecognizerModel();
private:
    void *m_impl;
};

class FaceRecognizerPrivate {
public:
    class Param;
    const Param *GetParam() const;

    explicit FaceRecognizerPrivate( const Param *param );

    explicit  FaceRecognizerPrivate( const FaceRecognizerModel &model );



    /**
    * \brief 构造识别器
    * \param [in] modelPath 识别器模型路径
    * \note 识别器模型一般为 FaceRecognizerPrivate5.0.XXX.dat
    * \note 如果只是使用人脸裁剪的部分，则不需要加载模型
    * \note 默认会以 AUTO 模式使用计算设备
    */
    explicit FaceRecognizerPrivate( const char *modelPath = NULL );

    /**
    * \brief 构造识别器
    * \param [in] modelPath 识别器模型路径
    * \param [in] device 使用的计算设备
    * \note 识别器模型一般为 FaceRecognizerPrivate5.0.XXX.dat
    * \note 如果只是使用人脸裁剪的部分，则不需要加载模型
    */
    explicit FaceRecognizerPrivate( const char *modelPath, SeetaDevice device, int gupid = 0 );

    /**
    * \brief 构造识别器
    * \param [in] modelBuffer 识别模型的内存对象
    * \param [in] bufferLength 识别模型的内存长度
    * \note 识别器模型一般为 FaceRecognizerPrivate5.0.XXX.dat
    * \note 如果只是使用人脸裁剪的部分，则不需要加载模型
    */
    explicit FaceRecognizerPrivate( const char *modelBuffer, size_t bufferLength, SeetaDevice device = SEETA_DEVICE_AUTO, int gpuid = 0 );

    ~FaceRecognizerPrivate();

    /**
    * \brief 为识别器加载模型，会卸载原加载模型
    * \param [in] modelPath 识别模型路径
    * \return 加载成功后返回真
    * \note 此函数是为了在构造的时候没有加载模型的情况下调用
    * \note 默认会以 AUTO 模式使用计算设备
    */
    bool LoadModel( const char *modelPath );

    /**
    * \brief 为识别器加载模型，会卸载原加载模型
    * \param [in] modelPath 识别模型路径
    * \param [in] device 使用的计算设备
    * \return 加载成功后返回真
    * \note 此函数是为了在构造的时候没有记载模型的情况下调用
    */
    bool LoadModel( const char *modelPath, SeetaDevice device, int gpuid = 0 );

    /**
    * \brief 为识别器加载模型，会卸载原加载模型
    * \param [in] modelBuffer 识别模型的内存对象
    * \param [in] bufferLength 识别模型的内存长度
    * \param [in] device 使用的计算设备
    * \return 加载成功后返回真
    * \note 此函数是为了在构造的时候没有记载模型的情况下调用
    */
    bool LoadModel( const char *modelBuffer, size_t bufferLength, SeetaDevice device = SEETA_DEVICE_AUTO, int gpuid = 0 );

    /**
     * \brief 获取识别器提取特征维度
     * \return 特征维度
     */
    uint32_t GetFeatureSize();

    /**
     * \brief 获取用于识别而裁剪的人脸图片宽度
     * \return 人脸图片宽度
     */
    uint32_t GetCropWidth();

    /**
    * \brief 获取用于识别而裁剪的人脸图片高度
    * \return 人脸图片高度
    */
    uint32_t GetCropHeight();

    /**
    * \brief 获取用于识别而裁剪的人脸图片通道数
    * \return 人脸图片通道数
    */
    uint32_t GetCropChannels();

    /**
     * \brief 裁剪人脸
     * \param [in] srcImg 原始图像，彩色
     * \param [in] llpoint 原始图像中人脸特征点，5个
     * \param [out] dstImg 目标图像，根据裁剪信息预先申请好内存
     * \param [in] posNum 人脸姿态，保留参数，此版本无意义
     * \return 只有裁剪成功后才返回真
     */
    bool CropFace( const SeetaImageData &srcImg,
                   const SeetaPointF *llpoint,
                   SeetaImageData &dstImg,
                   uint8_t posNum = 1 );

    /**
     * \brief 在裁剪好的人脸上提取特征
     * \param [in] cropImg 裁剪好的人脸（彩色），使用 CropFace 或者对应方法
     * \param [out] feats 待提取特征存放空间
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeature( const SeetaImageData &cropImg,
                         float *feats );

    /**
     * \brief 在裁剪好的人脸上提取特征（归一化的）
     * \param [in] cropImg 裁剪好的人脸（彩色），使用 CropFace 或者对应方法
     * \param [out] feats 待提取特征存放空间
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeatureNormalized( const SeetaImageData &cropImg,
                                   float *feats );

    /**
     * \brief 在原始图像上，根据定位点提取特征
     * \param [in] srcImg 原始图像，彩色
     * \param [in] llpoint 原始图像中人脸特征点，5个
     * \param [out] feats 待提取特征存放空间
     * \param [in] posNum 人脸姿态，保留参数，此版本无意义
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeatureWithCrop( const SeetaImageData &srcImg,
                                 const SeetaPointF *llpoint,
                                 float *feats,
                                 uint8_t posNum = 1 );

    /**
     * \brief 在原始图像上，根据定位点提取特征（归一化的）
     * \param [in] srcImg 原始图像，彩色
     * \param [in] llpoint 原始图像中人脸特征点，5个
     * \param [out] feats 待提取特征存放空间
     * \param [in] posNum 人脸姿态，保留参数，此版本无意义
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeatureWithCropNormalized( const SeetaImageData &srcImg,
                                           const SeetaPointF *llpoint,
                                           float *feats,
                                           uint8_t posNum = 1 );

    /**
     * \brief 计算特征 fc1 和 fc2 的相似度
     * \param [in] fc1 特征向量1
     * \param [in] fc2 特征向量2
     * \param [in] dim 特征维度
     * \return 相似度
     * \note 默认特征维度应该是 GetFeatureSize() 的返回值，如果不是，则需要传入对应的特征长度
     */
    float CalcSimilarity( const float *fc1,
                          const float *fc2,
                          long dim = -1 );

    /**
     * \brief 计算特征 fc1 和 fc2 的相似度
     * \param [in] fc1 特征向量1（归一化的）
     * \param [in] fc2 特征向量2（归一化的）
     * \param [in] dim 特征维度
     * \return 相似度
     * \note 默认特征维度应该是 GetFeatureSize() 的返回值，如果不是，则需要传入对应的特征长度
     * \note 此计算相似度函数必须输入归一化的特征向量，由后缀 Normalized 的函数提取的特征
     */
    float CalcSimilarityNormalized( const float *fc1,
                                    const float *c2,
                                    long dim = -1 );

    /**
     * \brief 设置最大单次输入样本数
     * \param max_batch 最大样本数
     * \return 返回之前的设置
     * \note 注意此函数设置之后，所有的识别器对象都会改成这个设置，暂时不支持对每个对象设置
     */
    static int SetMaxBatchGlobal( int max_batch );

    /**
     * \brief 获取当前识别器单次输入最大样本数
     * \return 最大样本数
     */
    int GetMaxBatch();

    /**
    * \brief 设置最识别器处理线程数
    * \param core_number 处理线程数
    * \return 返回之前的设置
    * \note 注意此函数设置之后，所有的识别器对象都会改成这个设置，暂时不支持对每个对象设置
    */
    static int SetCoreNumberGlobal( int core_number );

    /**
    * \brief 获取当前识别器处理线程数
    * \return 处理线程数
    */
    int GetCoreNumber();

    /**
     * \brief 批量提取特征
     * \param faces 裁剪好的每一张人脸
     * \param [out] feats 存放提取特征的空间，大小为 faces.size() * GetFeatureSize()，第 i * GetFeatureSize() 下标开始的 GetFeatureSize() 个特征是第i张人脸特征
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeature( const std::vector<SeetaImageData> &faces, float *feats, bool normalization = false );

    /**
     * \brief 批量提取特征（归一化的）
     * \param faces 裁剪好的每一张人脸
     * \param [out] feats 存放提取特征的空间，大小为 faces.size() * GetFeatureSize()，第 i * GetFeatureSize() 下标开始的 GetFeatureSize() 个特征是第i张人脸特征
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeatureNormalized( const std::vector<SeetaImageData> &faces, float *feats );

    /**
     * \brief 批量提取特征
     * \param images 原始图像
     * \param points 原始图像上的每一个人脸，大小为 images.size() * 5，第 i * 5 下标开始的 5 点是第 i 张图片上的特征点
     * \param [out] feats 存放提取特征的空间，大小为 images.size() * GetFeatureSize()，第 i * GetFeatureSize() 下标开始的 GetFeatureSize() 个特征是第i张人脸特征
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeatureWithCrop( const std::vector<SeetaImageData> &images, const std::vector<SeetaPointF> &points, float *feats, bool normalization = false );

    /**
     * \brief 批量提取特征（归一化的）
     * \param images 原始图像
     * \param points 原始图像上的每一个人脸，大小为 images.size() * 5，第 i * 5 下标开始的 5 点是第 i 张图片上的特征点
     * \param [out] feats 存放提取特征的空间，大小为 images.size() * GetFeatureSize()，第 i * GetFeatureSize() 下标开始的 GetFeatureSize() 个特征是第i张人脸特征
     * \return 只有提取成功后才返回真
     */
    bool ExtractFeatureWithCropNormalized( const std::vector<SeetaImageData> &images, const std::vector<SeetaPointF> &points, float *feats );

private:
    FaceRecognizerPrivate( const FaceRecognizerPrivate &other ) = delete;
    const FaceRecognizerPrivate &operator=( const FaceRecognizerPrivate &other ) = delete;

private:
    class Recognizer;
    friend class FaceRecognizerModel;
    Recognizer *recognizer;
};
#endif // _SEETA_FACE_RECOGNIZER_H
