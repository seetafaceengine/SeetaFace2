#ifndef _SEETA_FACE_RECOGNIZER_H
#define _SEETA_FACE_RECOGNIZER_H

#include <Struct.h>
#include <vector>

class FaceRecognizerModel
{
public:
	friend class FaceRecognizerPrivate;

	FaceRecognizerModel(const char *model_path, int device);
	~FaceRecognizerModel();

private:
	void *m_impl;
};

class FaceRecognizerPrivate
{
public:
	class Param;
	const Param *GetParam() const;

	explicit FaceRecognizerPrivate(const Param *param);
	explicit  FaceRecognizerPrivate(const FaceRecognizerModel &model);

	/**
	* \brief Construct recognizer
	* \param [in] modelPath
	*/
	explicit FaceRecognizerPrivate(const char *modelPath = NULL);

	/**
	* \brief Construct recognizer
	* \param [in] modelPath
	* \param [in] device ONLY CPU supported
	*/
	explicit FaceRecognizerPrivate(const char *modelPath, SeetaDevice device, int gupid = 0);

	/**
	* \brief Construct recognizer
	* \param [in] modelBuffer
	* \param [in] bufferLength
	* \param [in] device ONLY CPU supported
	*/
	explicit FaceRecognizerPrivate(const char *modelBuffer, size_t bufferLength, SeetaDevice device = SEETA_DEVICE_AUTO, int gpuid = 0);

	~FaceRecognizerPrivate();

	/**
	* \brief Reload model
	* \param [in] modelPath
	* \return true if succeed
	*/
	bool LoadModel(const char *modelPath);

	/**
	* \brief Reload model
	* \param [in] modelPath
	* \param [in] device ONLY CPU supported
	* \return true if succeed
	*/
	bool LoadModel(const char *modelPath, SeetaDevice device, int gpuid = 0);

	/**
	* \brief
	* \param [in] modelBuffer
	* \param [in] bufferLength
	* \param [in] device ONLY CPU supported
	* \return true if succeed
	*/
	bool LoadModel(const char *modelBuffer, size_t bufferLength, SeetaDevice device = SEETA_DEVICE_AUTO, int gpuid = 0);

	/**
	 * \brief Get feature size
	 * \return
	 */
	uint32_t GetFeatureSize();

	/**
	 * \brief Get crop face width
	 * \return
	 */
	uint32_t GetCropWidth();

	/**
	* \brief Get crop face height
	* \return
	*/
	uint32_t GetCropHeight();

	/**
	* \brief Get crop face channels
	* \return
	*/
	uint32_t GetCropChannels();

	/**
	 * \brief cropface
	 * \param [in] srcImg source image, BGR color in HWC format
	 * \param [in] llpoint 5 landmarks
	 * \param [out] dstImg cropped face [GetCropHeight(), GetCropWidth(), GetCropChannels()]
	 * \param [in] posNum NOTUSED
	 * \return return true if succeed
	 */
	bool CropFace(const SeetaImageData &srcImg, const SeetaPointF *llpoint, SeetaImageData &dstImg, uint8_t posNum = 1);

	/**
	 * \brief extrace face from cropped face
	 * \param [in] cropImg @sa CropFace
	 * \param [out] feats size [GetFeatureSize()]
	 * \return return true if succeed
	 */
	bool ExtractFeature(const SeetaImageData &cropImg, float *feats);

	/**
	 * \brief extrace face from cropped face, features will be normalized.
	 * \param [in] cropImg @sa CropFace
	 * \param [out] feats size [GetFeatureSize()]
	 * \return return true if succeed
	 */
	bool ExtractFeatureNormalized(const SeetaImageData &cropImg, float *feats);

	/**
	 * \brief extrace face from orignal face
	 * \param [in] srcImg source image, BGR color in HWC format
	 * \param [in] llpoint 5 landmarks
	 * \param [out] feats size [GetFeatureSize()]
	 * \param [in] posNum NOTUSED
	 * \return return true if succeed
	 */
	bool ExtractFeatureWithCrop(const SeetaImageData &srcImg, const SeetaPointF *llpoint, float *feats, uint8_t posNum = 1);

	/**
	 * \brief extrace face from orignal face, features will be normalized.
	 * \param [in] srcImg source image, BGR color in HWC format
	 * \param [in] llpoint 5 landmarks
	 * \param [out] feats size [GetFeatureSize()]
	 * \param [in] posNum NOTUSED
	 * \return return true if succeed
	 */
	bool ExtractFeatureWithCropNormalized(const SeetaImageData &srcImg, const SeetaPointF *llpoint, float *feats, uint8_t posNum = 1);

	/**
	 * \brief calculate similarity
	 * \param [in] fc1 features1
	 * \param [in] fc2 features2
	 * \param [in] dim feature length, -1 means GetFeatureSize()
	 * \return similarity
	 */
	float CalcSimilarity(const float *fc1, const float *fc2, long dim = -1);

	/**
	 * \brief calculate normalized features similarity
	 * \param [in] fc1 features1
	 * \param [in] fc2 features2
	 * \param [in] dim feature length, -1 means GetFeatureSize()
	 * \return similarity
	 */
	float CalcSimilarityNormalized(const float *fc1, const float *c2, long dim = -1);

	static int SetMaxBatchGlobal(int max_batch);

	int GetMaxBatch();

	static int SetCoreNumberGlobal(int core_number);

	int GetCoreNumber();

	/**
	 * \brief extrace batch images features
	 * \param faces each cropped face
	 * \param [out] feats size [faces.size() * GetFeatureSize()]
	 * \return ture if succeed
	 */
	bool ExtractFeature(const std::vector<SeetaImageData> &faces, float *feats, bool normalization = false);

	/**
	 * \brief extrace batch images normalized features
	 * \param faces each cropped face
	 * \param [out] feats size [faces.size() * GetFeatureSize()]
	 * \return ture if succeed
	 */
	bool ExtractFeatureNormalized(const std::vector<SeetaImageData> &faces, float *feats);

	/**
	 * \brief extrace batch images features
	 * \param [in] images each images
	 * \param [in] points each each points, size[images.size() * 5]
	 * \param [out] feats size [faces.size() * GetFeatureSize()]
	 * \param normalization each cropped face
	 * \return ture if succeed
	 */
	bool ExtractFeatureWithCrop(const std::vector<SeetaImageData> &images, const std::vector<SeetaPointF> &points, float *feats, bool normalization = false);

	/**
	* \brief extrace batch images normalized features
	* \param [in] images each images
	* \param [in] points each each points, size[images.size() * 5]
	* \param [out] feats size [faces.size() * GetFeatureSize()]
	* \param normalization each cropped face
	* \return ture if succeed
	*/
	bool ExtractFeatureWithCropNormalized(const std::vector<SeetaImageData> &images, const std::vector<SeetaPointF> &points, float *feats);

private:
	FaceRecognizerPrivate(const FaceRecognizerPrivate &other) = delete;
	const FaceRecognizerPrivate &operator=(const FaceRecognizerPrivate &other) = delete;

private:
	class Recognizer;
	friend class FaceRecognizerModel;
	Recognizer *recognizer;
};

#endif // _SEETA_FACE_RECOGNIZER_H
