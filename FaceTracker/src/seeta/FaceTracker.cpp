//
// Created by kier on 2019-09-16.
//

#include <seeta/FaceDetector.h>
#include <vector>

#include <algorithm>
#include <queue>

#include <seeta/CTrackingFaceInfo.h>
#include <seeta/FaceTracker.h>

#include <iostream>

namespace seeta
{
	namespace v2
	{
		struct TrackedFace
		{
			SeetaRect pos;
			int PID = 0;
			float conf = 0;
			int frame_no = 0;
		};

		class FaceTracker::Implement
		{
		public:
			Implement();
			~Implement();

			void bind(const std::shared_ptr<seeta::FaceDetector> pFD);

			void refresh();

			const std::vector<TrackedFace> &Detect(const SeetaImageData &image, int frame_no = -1) const;

			SeetaTrackingFaceInfoArray DetectV2(const SeetaImageData &image, int frame_no = -1) const;

			seeta::FaceDetector &FD() { return *pFD; }

			const seeta::FaceDetector &FD() const { return *pFD; }

		private:
			std::shared_ptr<seeta::FaceDetector> pFD;
			mutable std::vector<TrackedFace> pre_tracked_faces;
			mutable int max_PID = 0;
			float min_score = 0.3f;
			float max_score = 0.5f;

			mutable int frame_no = 0;
			mutable std::vector<SeetaTrackingFaceInfo> tracked_faces;
		};
	}

	using namespace v2;
}

namespace seeta
{
	namespace v2
	{
		FaceTracker::Implement::Implement() {}
		FaceTracker::Implement::~Implement() {}

		void FaceTracker::Implement::bind(const std::shared_ptr<seeta::FaceDetector> pFD)
		{
			this->pFD = pFD;
			this->refresh();
		}

		void FaceTracker::Implement::refresh()
		{
			this->max_PID = 0;
			this->frame_no = 0;
		}

		struct ScoredTrackedFace
		{
			ScoredTrackedFace(const TrackedFace &tracked_face) : face(tracked_face) {}

			TrackedFace face;
			float iou_score = 0;
		};

		static float IoU(const SeetaRect &w1, const SeetaRect &w2)
		{
			int xOverlap = std::max(0, std::min(w1.x + w1.width - 1, w2.x + w2.width - 1) - std::max(w1.x, w2.x) + 1);
			int yOverlap = std::max(0, std::min(w1.y + w1.height - 1, w2.y + w2.height - 1) - std::max(w1.y, w2.y) + 1);
			int intersection = xOverlap * yOverlap;
			int unio = w1.width * w1.height + w2.width * w2.height - intersection;

			return float(intersection) / unio;
		}

		const std::vector<TrackedFace> &FaceTracker::Implement::Detect(const SeetaImageData &image, int frame_no) const
		{
			if (!this->pFD)
			{
				pre_tracked_faces.clear();
				return pre_tracked_faces;
			}

			if (frame_no < 0)
			{
				frame_no = this->frame_no;
				++this->frame_no;
			}

			int num = 0;
			auto face_array = this->pFD->detect(image);
			std::vector<SeetaRect> faces;
			num = int(face_array.size);

			for (int i = 0; i < num; ++i)
			{
				faces.push_back(face_array.data[i].pos);
			}

			// prepare scored trakced faces
			std::deque<ScoredTrackedFace> scored_tracked_faces(pre_tracked_faces.begin(), pre_tracked_faces.end());
			std::vector<TrackedFace> now_trakced_faces;

			for (int i = 0; i < num; ++i)
			{
				auto &face = faces[i];

				for (auto &scored_tracked_face : scored_tracked_faces)
				{
					scored_tracked_face.iou_score = IoU(scored_tracked_face.face.pos, face);
					std::cout << scored_tracked_face.iou_score << std::endl;
				}

				if (scored_tracked_faces.size() > 1)
				{
					std::partial_sort(scored_tracked_faces.begin(), scored_tracked_faces.begin() + 1,
						scored_tracked_faces.end(), [](const ScoredTrackedFace &a, const ScoredTrackedFace &b)
					{
						return a.iou_score > b.iou_score;
					});
				}

				if (!scored_tracked_faces.empty() && scored_tracked_faces.front().iou_score > this->min_score)
				{
					ScoredTrackedFace matched_face = scored_tracked_faces.front();
					scored_tracked_faces.pop_front();
					TrackedFace &tracked_face = matched_face.face;

					if (matched_face.iou_score < max_score)
					{
						tracked_face.pos.x = (tracked_face.pos.x + face.x) / 2;
						tracked_face.pos.y = (tracked_face.pos.y + face.y) / 2;
						tracked_face.pos.width = (tracked_face.pos.width + face.width) / 2;
						tracked_face.pos.height = (tracked_face.pos.height + face.height) / 2;
					}
					else
					{
						tracked_face.pos = face;
					}

					tracked_face.conf = face_array.data[i].score;
					tracked_face.frame_no = frame_no;
					now_trakced_faces.push_back(tracked_face);
				}
				else
				{
					TrackedFace tracked_face;
					tracked_face.pos = face;
					tracked_face.PID = max_PID;
					tracked_face.conf = face_array.data[i].score;
					tracked_face.frame_no = frame_no;
					max_PID++;
					now_trakced_faces.push_back(tracked_face);
				}
			}

			pre_tracked_faces = now_trakced_faces;

			return pre_tracked_faces;
		}

		SeetaTrackingFaceInfoArray FaceTracker::Implement::DetectV2(const SeetaImageData &image, int frame_no) const
		{
			auto &faces = Detect(image, frame_no);
			tracked_faces.clear();

			for (auto &face : faces)
			{
				SeetaTrackingFaceInfo info;
				info.PID = face.PID;
				info.score = face.conf;
				info.frame_no = face.frame_no;
				info.pos = face.pos;

				tracked_faces.push_back(info);
			}

			SeetaTrackingFaceInfoArray result = { nullptr, 0 };
			result.data = tracked_faces.data();
			result.size = tracked_faces.size();

			return result;
		}
	}

	FaceTracker::FaceTracker(const SeetaModelSetting &setting)
	{
		m_impl = new Implement;
		m_impl->bind(std::make_shared<seeta::FaceDetector>(setting));
		m_impl->FD().set(seeta::FaceDetector::Property(PROPERTY_VIDEO_STABLE), 0);
	}

	FaceTracker::FaceTracker(const SeetaModelSetting &setting, int core_width, int core_height)
	{
		m_impl = new Implement;
		m_impl->bind(std::make_shared<seeta::FaceDetector>(setting, core_width, core_height));
		m_impl->FD().set(seeta::FaceDetector::Property(PROPERTY_VIDEO_STABLE), 0);
	}

	FaceTracker::~FaceTracker()
	{
		delete m_impl;
	}

	SeetaTrackingFaceInfoArray FaceTracker::track(const SeetaImageData &image, int frame_no) const
	{
		return m_impl->DetectV2(image, frame_no);
	}

	void FaceTracker::set(FaceTracker::Property property, double value)
	{
		if (property == PROPERTY_VIDEO_STABLE)
			return;

		m_impl->FD().set(seeta::FaceDetector::Property(property), value);
	}

	double FaceTracker::get(FaceTracker::Property property) const
	{
		return m_impl->FD().get(seeta::FaceDetector::Property(property));
	}
}

