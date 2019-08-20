#ifndef _SEETA_COMMON_ALIGNMENT_H
#define _SEETA_COMMON_ALIGNMENT_H
#include <cstdint>

/**
 * \brief 进行缩放时采样的算法
 */
enum SAMPLING_TYPE
{
    LINEAR, ///< 线性采样
    BICUBIC ///< Cubic 采样
};

/**
 * \brief 处于边缘时的边缘填充算法
 */
enum PADDING_TYPE
{
    ZERO_PADDING,           ///< 0 填充
    NEAREST_PADDING,        ///< 复制填充
};

/**
 * \brief 通用人脸裁剪接口
 * \param image_data 输入图像， 以 height * width * channels 的次序存放字节表示的灰度值（0 到 255）
 * \param image_width 图片宽度
 * \param image_height 图片高度
 * \param image_channels 图片通道数，彩色图片一般为 3，灰度图片一般为 1
 * \param crop_data 输出图像（Crop 好的数据），[crop_width + pad_left + pad_right, crop_height + pad_top + pad_bottom, image_channels] 大小的数据
 * \param crop_width 输出宽度，对于标准人脸最后获取的宽度，\b 并不绝对是输出图像的宽度，受pad参数影响
 * \param crop_height 输出高度，对于标准人脸最后获取的高度，\b 并不绝对是输出图像的高度，受pad参数影响
 * \param points 定位的特征点，以 {(x1, y1), (x2, y2), ...} 的格式存放
 * \param points_num 特征点数量
 * \param mean_shape 平局人脸模型，以 {(x1, y1), (x2, y2), ...} 的格式存放
 * \param mean_shape_width 平均人脸模型的宽度
 * \param mean_shape_height 平均人脸模型的高度
 * \param pad_top 向上扩展，可以为负（表示向内缩）
 * \param pad_bottom 向下扩展，可以为负（表示向内缩）
 * \param pad_left 向左扩展，可以为负（表示向内缩）
 * \param pad_right 向右扩展，可以为负（表示向内缩）
 * \param final_points 人脸裁剪后对应特征点的坐标，以 {(x1, y1), (x2, y2), ...} 的格式存放
 * \param type 缩放时插值使用的方法
 * \return 人脸裁剪是否成功
 * \note 最后裁剪出来的人脸大小为 [crop_width + pad_left + pad_right, crop_height + pad_top + pad_bottom] 的大小
 */
bool face_crop_core(
    const uint8_t *image_data, int image_width, int image_height, int image_channels,
    uint8_t *crop_data, int crop_width, int crop_height,
    const float *points, int points_num,
    const float *mean_shape, int mean_shape_width, int mean_shape_height,
    int pad_top = 0, int pad_bottom = 0, int pad_left = 0, int pad_right = 0,
    float *final_points = nullptr,
    SAMPLING_TYPE type = LINEAR );

bool face_crop_core_ex(
    const uint8_t *image_data, int image_width, int image_height, int image_channels,
    uint8_t *crop_data, int crop_width, int crop_height,
    const float *points, int points_num,
    const float *mean_shape, int mean_shape_width, int mean_shape_height,
    int pad_top = 0, int pad_bottom = 0, int pad_left = 0, int pad_right = 0,
    float *final_points = nullptr,
    SAMPLING_TYPE type = LINEAR,
    PADDING_TYPE ptype = ZERO_PADDING );

#endif // _SEETA_COMMON_ALIGNMENT_H
