#include "ImageConcat.h"
#include "ImageBlend.h"
#include "Profiling.h"

#if defined(IM_FUNCTION_USE_TRACY)
#include <tracy/Tracy.hpp>
#include <chrono>
#include <thread>
#include <cstdio>

namespace
{
	// 延迟上传。
	void tracyWaitForProfilerUpload()
	{
		std::fprintf(stderr, "Tracy: waiting for profiler connection (up to 120s)...\n");
		for (int i = 0; i < 12000; ++i)
		{
			if (tracy::GetProfiler().IsConnected())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				return;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		std::fprintf(stderr, "Tracy: no connection, exiting.\n");
	}
}
#endif

// 融合测试
void testBlend()
{
	ZoneScopedN("testBlend");
	cv::Mat img1 = cv::imread("data\\1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("data\\2.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img3 = cv::imread("data\\3.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img4 = cv::imread("data\\4.jpg", cv::IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty() || img3.empty() || img4.empty()) return;

	std::vector<cv::Mat> images = { img1, img2, img3, img4 };

	{
		ZoneScopedN("blendMaxValue");
		ImageBlend blend;
		if (blend.setImages(images) == ImageBlendError::OK
			&& blend.setConfig(ImageBlendMode::MaxValue, MaxValueConfig{}) == ImageBlendError::OK
			&& blend.execute() == ImageBlendError::OK)
		{
			const ImageBlendOutputMap outputs = blend.getResult();
			auto resultIt = outputs.find(ImageBlendOutputName::Result);
			if (resultIt != outputs.end())
				cv::imwrite("blend_max.jpg", resultIt->second);
		}
	}

	{
		ZoneScopedN("blendWeightedAverage");
		ImageBlend blend;
		if (blend.setImages(images) == ImageBlendError::OK
			&& blend.setConfig(ImageBlendMode::WeightedAverage, WeightedAverageConfig{ { 0.4, 0.3, 0.2, 0.1 } }) == ImageBlendError::OK
			&& blend.execute() == ImageBlendError::OK)
		{
			const ImageBlendOutputMap outputs = blend.getResult();
			auto resultIt = outputs.find(ImageBlendOutputName::Result);
			if (resultIt != outputs.end())
				cv::imwrite("blend_avg.jpg", resultIt->second);
		}
	}
}

static cv::Vec3f ringLightDirection(float slantRad, float azimuthRad)
{
	return cv::Vec3f(
		std::sin(slantRad) * std::cos(azimuthRad),
		std::sin(slantRad) * std::sin(azimuthRad),
		std::cos(slantRad));
}

static LightSystem buildRingLightSystem(size_t count, float slantDeg)
{
	LightSystem sys;
	sys.lights.resize(count);
	const float slantRad = slantDeg * static_cast<float>(CV_PI) / 180.f;
	for (size_t i = 0; i < count; ++i)
	{
		const float azimuthRad = static_cast<float>(2.0 * CV_PI * i / count);
		sys.lights[i].geo.dir = ringLightDirection(slantRad, azimuthRad);
	}
	return sys;
}

static cv::Mat normalMapToBgr8(const cv::Mat& normalMap)
{
	cv::Mat vis;
	normalMap.convertTo(vis, CV_32FC3, 0.5, 0.5);
	vis.convertTo(vis, CV_8UC3, 255.0);
	return vis;
}

static cv::Mat floatMapToGray8(const cv::Mat& map)
{
	cv::Mat vis;
	cv::normalize(map, vis, 0, 255, cv::NORM_MINMAX);
	vis.convertTo(vis, CV_8UC1);
	return vis;
}

static cv::Mat gradientMapToGray8(const cv::Mat& gradientMap)
{
	std::vector<cv::Mat> channels(2);
	cv::split(gradientMap, channels);
	cv::Mat magnitude;
	cv::magnitude(channels[0], channels[1], magnitude);
	// 固定线性映射（与法线 RGB 一致），避免全图 min-max 把局部浅凹痕压没
	cv::Mat clamped;
	cv::min(magnitude, 1.0, clamped);
	cv::max(clamped, 0.0, clamped);
	cv::Mat vis;
	clamped.convertTo(vis, CV_32F, 0.5, 0.5);
	vis.convertTo(vis, CV_8UC1, 255.0);
	return vis;
}

static PhotometricStereoOutputFlags photometricStereoOutputsWithAuxMaps()
{
	PhotometricStereoOutputFlags outputs;
	outputs.syntheticLightDir = cv::Vec3f(0.f, 0.f, 1.f);
	outputs.computeGradientMap = true;
	outputs.computeHeightMap = true;
	outputs.computeCurvatureMap = true;
	outputs.heightIntegrationScale = 0.5f;
	return outputs;
}

static void writePhotometricStereoOutputs(ImageBlend& blend, const std::string& pathPrefix)
{
	const ImageBlendOutputMap outputs = blend.getResult();

	auto resultIt = outputs.find(ImageBlendOutputName::Result);
	if (resultIt != outputs.end())
		cv::imwrite(pathPrefix + ".jpg", resultIt->second);

	auto normalIt = outputs.find(ImageBlendOutputName::NormalMap);
	if (normalIt != outputs.end())
		cv::imwrite(pathPrefix + "_normal.jpg", normalMapToBgr8(normalIt->second));

	auto albedoIt = outputs.find(ImageBlendOutputName::AlbedoMap);
	if (albedoIt != outputs.end())
		cv::imwrite(pathPrefix + "_albedo.jpg", floatMapToGray8(albedoIt->second));

	auto gradientIt = outputs.find(ImageBlendOutputName::GradientMap);
	if (gradientIt != outputs.end())
		cv::imwrite(pathPrefix + "_gradient.jpg", gradientMapToGray8(gradientIt->second));

	auto heightIt = outputs.find(ImageBlendOutputName::HeightMap);
	if (heightIt != outputs.end())
		cv::imwrite(pathPrefix + "_height.jpg", floatMapToGray8(heightIt->second));

	auto curvatureIt = outputs.find(ImageBlendOutputName::CurvatureMap);
	if (curvatureIt != outputs.end())
		cv::imwrite(pathPrefix + "_curvature.jpg", floatMapToGray8(curvatureIt->second));
}

// 光度立体融合测试
void testPhotometricStereo()
{
	ZoneScopedN("testPhotometricStereo");
	cv::Mat img1 = cv::imread("data\\1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("data\\2.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img3 = cv::imread("data\\3.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img4 = cv::imread("data\\4.jpg", cv::IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty() || img3.empty() || img4.empty()) return;

	std::vector<cv::Mat> images = { img1, img2, img3, img4 };

	PhotometricStereoOutputFlags outputs = photometricStereoOutputsWithAuxMaps();

	LightSystem lightSystem = buildRingLightSystem(images.size(), 45.f);

	ImageBlend blend;
	if (blend.setImages(images) != ImageBlendError::OK) return;
	if (blend.setConfig(ImageBlendMode::PhotometricStereo, PhotometricStereoConfig{ lightSystem, outputs }) != ImageBlendError::OK) return;
	if (blend.execute() != ImageBlendError::OK) return;

	writePhotometricStereoOutputs(blend, "blend_ps");
}

// 标定测试：平板标定 -> 光度立体融合
void testFlatCalibration()
{
	ZoneScopedN("testFlatCalibration");
	cv::Mat img1 = cv::imread("data\\1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("data\\2.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img3 = cv::imread("data\\3.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img4 = cv::imread("data\\4.jpg", cv::IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty() || img3.empty() || img4.empty()) return;

	std::vector<cv::Mat> images = { img1, img2, img3, img4 };

	std::vector<LightSource> lights;
	const float flatPitchDeg = 45.f; // 手动测量的共用俯仰角 [0, 90)，度
	if (ImageBlend::executeFlatCalibration(images, flatPitchDeg, lights) != ImageBlendError::OK) return;

	LightSystem lightSystem;
	lightSystem.lights = lights;

	PhotometricStereoOutputFlags outputs = photometricStereoOutputsWithAuxMaps();

	ImageBlend blend;
	if (blend.setImages(images) != ImageBlendError::OK) return;
	if (blend.setConfig(ImageBlendMode::PhotometricStereo, PhotometricStereoConfig{ lightSystem, outputs }) != ImageBlendError::OK) return;
	if (blend.execute() != ImageBlendError::OK) return;

	writePhotometricStereoOutputs(blend, "blend_ps_calib");
}

// 漫反射球标定测试（需漫反射标定球图像；当前 data 非球体时会标定失败并跳过）
void testSphereCalibration()
{
	ZoneScopedN("testSphereCalibration");
	cv::Mat img1 = cv::imread("data\\1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("data\\2.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img3 = cv::imread("data\\3.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img4 = cv::imread("data\\4.jpg", cv::IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty() || img3.empty() || img4.empty()) return;

	std::vector<cv::Mat> images = { img1, img2, img3, img4 };

	std::vector<LightSource> lights;
	if (ImageBlend::executeSphereCalibration(images, lights) != ImageBlendError::OK) return;

	LightSystem lightSystem;
	lightSystem.lights = lights;

	PhotometricStereoOutputFlags outputs = photometricStereoOutputsWithAuxMaps();

	ImageBlend blend;
	if (blend.setImages(images) != ImageBlendError::OK) return;
	if (blend.setConfig(ImageBlendMode::PhotometricStereo, PhotometricStereoConfig{ lightSystem, outputs }) != ImageBlendError::OK) return;
	if (blend.execute() != ImageBlendError::OK) return;

	writePhotometricStereoOutputs(blend, "blend_ps_sphere");
}


void testConcat()
{
	cv::Mat img1 = cv::imread("data\\1.jpg", -1);
	cv::Mat img2 = cv::imread("data\\2.jpg", -1);
	cv::Mat img3 = cv::imread("data\\3.jpg", -1);
	cv::Mat img4 = cv::imread("data\\4.jpg", -1);
	std::vector<cv::Mat> images;
	images.push_back(img1);
	images.push_back(img2);
	images.push_back(img3);
	images.push_back(img4);
	ImageConcat concat;
	ImageConcatError res = concat.setInput(images, 2, 2);
	res = concat.execute();
	cv::Mat result = concat.getResult();
	/*cv::imshow("img", result);
	cv::waitKey(0);*/
	cv::imwrite("test.jpg", result);

}

int main()
{
	FrameMark;
	// testConcat();
	testBlend();
	testPhotometricStereo();
	// testFlatCalibration();
	// testSphereCalibration();
	FrameMark;
#if defined(IM_FUNCTION_USE_TRACY)
	tracyWaitForProfilerUpload();
#endif
	return 0;
}