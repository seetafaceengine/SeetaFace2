#pragma warning(disable: 4819)

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <array>
#include <map>
#include <iostream>

int test_image(seeta::FaceDetector &FD, seeta::FaceLandmarker &FL)
{
	std::string image_path = "1.jpg";
	std::cout << "Loading image: " << image_path << std::endl;
	auto frame = cv::imread(image_path);
	seeta::cv::ImageData simage = frame;

	if (simage.empty()) {
		std::cerr << "Can not open image: " << image_path << std::endl;
		return EXIT_FAILURE;
	}


	auto faces = FD.detect(simage);

	for (int i = 0; i < faces.size; ++i)
	{
		auto &face = faces.data[i];
		auto points = FL.mark(simage, face.pos);

		cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(128, 128, 255), 3);
		for (auto &point : points)
		{
			cv::circle(frame, cv::Point(point.x, point.y), 3, CV_RGB(128, 255, 128), -1);
		}
	}

	auto output_path = image_path + ".pts81.png";
	cv::imwrite(output_path, frame);
	std::cerr << "Saving result into: " << output_path << std::endl;

	return EXIT_SUCCESS;
}

int main()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;
    seeta::ModelSetting FD_model( "./model/fd_2_00.dat", device, id );
    seeta::ModelSetting FL_model( "./model/pd_2_00_pts81.dat", device, id );

	seeta::FaceDetector FD(FD_model);
	seeta::FaceLandmarker FL(FL_model);

	FD.set(seeta::FaceDetector::PROPERTY_VIDEO_STABLE, 1);

	int camera_id = 0;
	cv::VideoCapture capture(camera_id);
	if (!capture.isOpened())
	{
		std::cerr << "Can not open camera(" << camera_id << "), testing image..." << std::endl;
		return test_image(FD, FL);
	}

	auto video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	auto video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	std::cout << "Open camera(" << camera_id << ")" << std::endl;

	cv::Mat frame;
	while (capture.isOpened())
	{
		capture.grab();
		capture.retrieve(frame);

		if (frame.empty()) break;

		seeta::cv::ImageData simage = frame;

		auto faces = FD.detect(simage);

		for (int i = 0; i < faces.size; ++i)
		{
			auto &face = faces.data[i];
			auto points = FL.mark(simage, face.pos);

			cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(128, 128, 255), 3);
			for (auto &point : points)
			{
				cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
			}
		}

		cv::imshow("Frame", frame);
		auto key = cv::waitKey(20);
		if (key == 27)
		{
			break;
		}
	}

	return EXIT_SUCCESS;
}
