//
// Created by seetadev on 2019/10/31.
//

#include "ClarityQuality.h"

#include "seeta/DataHelper.h"
#include "seeta/ImageProcess.h"

/**
 * @see The Blur Effect: Perception and Estimation with a New No-Reference Perceptual Blur Metric
 * @param data
 * @param width
 * @param height
 * @return
 */
float reblur(const unsigned char *data, int width, int height)
{
    float blur_val = 0.0f;
    float kernel[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
    float *BVer = new float[width * height];
    float *BHor = new float[width * height];

    float filter_data = 0.0;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (i < 4 || i > height - 5)
            {
                BVer[i * width + j] = data[i * width + j];
            }
            else
            {
                filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] + kernel[2] * data[(i - 2) * width + j] +
                              kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i)* width + j] + kernel[5] * data[(i + 1) * width + j] +
                              kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] + kernel[8] * data[(i + 4) * width + j];
                BVer[i * width + j] = filter_data;
            }

            if (j < 4 || j > width - 5)
            {
                BHor[i * width + j] = data[i * width + j];
            }
            else
            {
                filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] + kernel[2] * data[i * width + (j - 2)] +
                              kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + (j + 1)] +
                              kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] + kernel[8] * data[i * width + (j + 4)];
                BHor[i * width + j] = filter_data;
            }

        }
    }

    float D_Fver = 0.0;
    float D_FHor = 0.0;
    float D_BVer = 0.0;
    float D_BHor = 0.0;
    float s_FVer = 0.0;
    float s_FHor = 0.0;
    float s_Vver = 0.0;
    float s_VHor = 0.0;
    for (int i = 1; i < height; ++i)
    {
        for (int j = 1; j < width; ++j)
        {
            D_Fver = std::abs((float)data[i * width + j] - (float)data[(i - 1) * width + j]);
            s_FVer += D_Fver;
            D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
            s_Vver += std::max((float)0.0, D_Fver - D_BVer);

            D_FHor = std::abs((float)data[i * width + j] - (float)data[i * width + (j - 1)]);
            s_FHor += D_FHor;
            D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
            s_VHor += std::max((float)0.0, D_FHor - D_BHor);
        }
    }
    float b_FVer = (s_FVer - s_Vver) / s_FVer;
    float b_FHor = (s_FHor - s_VHor) / s_FHor;
    blur_val = std::max(b_FVer, b_FHor);

    delete[] BVer;
    delete[] BHor;

    return blur_val;
}

float evaluate_clarity(const SeetaImageData &image, const SeetaRect &info)
{
    if (!image.data || info.width < 9 || info.height < 9) return 0.0;
    seeta::Image color_data(image.data, image.width, image.height, image.channels);
    seeta::Image gray_data = seeta::gray(color_data);

    seeta::Image src_data = seeta::crop(gray_data, seeta::Rect(info.x, info.y, info.width, info.height));
    float blur_val = reblur(src_data.data(), src_data.width(), src_data.height());
    float clarity = 1.0f - blur_val;

    float T1 = 0.0f;
    float T2 = 1.0f;
    if (clarity <= T1)
    {
        clarity = 0.0;
    }
    else if (clarity >= T2)
    {
        clarity = 1.0;
    }
    else
    {
        clarity = (clarity - T1) / (T2 - T1);
    }

    return clarity;
}


