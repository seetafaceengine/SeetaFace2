#pragma warning(disable: 4819)

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>

seeta::ImageData crop_face(
        const seeta::FaceDetector &FD,
        const seeta::FaceLandmarker &PD,
        seeta::FaceRecognizer &FR,
        const SeetaImageData &original_image) 
{
    // detect face
    // detect points
    // crop face

    seeta::ImageData face(0, 0, 0);
    auto faces = FD.detect(original_image);

    if (faces.size == 0) 
		return face;

    auto points = PD.mark(original_image, faces.data[0].pos);
    face = FR.CropFace(original_image, points.data());
    return face;
}

int main()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;

    seeta::ModelSetting FD_model( "./model/fd_2_00.dat", device, id );
    seeta::ModelSetting PD_model( "./model/pd_2_00_pts5.dat", device, id );

    std::string test_image = "1.jpg";
    std::string crop_image_name = "crop_face.png";

	seeta::FaceDetector FD(FD_model);
	seeta::FaceLandmarker PD(PD_model);

	// construct FR with on model, only for crop face.
	seeta::FaceRecognizer FR;

	FD.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 80);

	seeta::cv::ImageData image = cv::imread(test_image);

	if (image.empty()) 
	{
	    std::cerr << "Can not open image: " << test_image << std::endl;
	    return 1;
	}

	auto face = crop_face(FD, PD, FR, image);

	if (face.width == 0 || face.height == 0) 
	{
	    std::cerr << "Can not detect any faces in image: " << test_image << std::endl;
	    return 2;
	}

	seeta::cv::ImageData cv_face = face;

	cv::imwrite(crop_image_name, cv_face.toMat());
	std::cout << "Save cropped image: " << crop_image_name << std::endl;

	return EXIT_SUCCESS;
}
