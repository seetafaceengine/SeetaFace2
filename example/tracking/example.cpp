#pragma warning(disable: 4819)

#include <seeta/FaceTracker.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>

int main()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;

    int id = 0;
    seeta::ModelSetting FD_model( "./model/fd_2_00.dat", device, id );
	seeta::FaceTracker FD(FD_model);
	FD.set(seeta::FaceTracker::PROPERTY_VIDEO_STABLE, 1);

	int camera_id = 1;
	cv::VideoCapture capture(camera_id);

	if (!capture.isOpened())
	{
		std::cerr << "Can not open camera(" << camera_id << ")" << std::endl;
		return -1;
	}

	// auto video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	// auto video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	std::cout << "Open camera(" << camera_id << ")" << std::endl;

	cv::Mat frame;

	while (capture.isOpened())
	{
		capture.grab();
		capture.retrieve(frame);

		if (frame.empty()) 
			break;

		seeta::cv::ImageData simage = frame;

		auto faces = FD.track(simage);

		for (int i = 0; i < faces.size; ++i) 
		{
            auto &face = faces.data[i];

            cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height),
                          CV_RGB(128, 128, 255), 3);
            cv::putText(frame, std::to_string(face.PID), cv::Point(face.pos.x, face.pos.y - 5), 3, 1,
                        CV_RGB(255, 128, 128));
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
