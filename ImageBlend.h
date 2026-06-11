#pragma once
#include <opencv2/opencv.hpp>
#include <variant>
#include <vector>
#include "LightTypes.h"

enum class ImageBlendError
{
	OK,
	NullImg,
	ParameterError,
	ImageNumberError,
	SizeMismatch,
	NotGrayscale,
	ImageTypeDifferent,
	WeightError,
	LightNumberError,
	LightParameterError,
	ConfigMismatch,
};

enum class ImageBlendMode
{
	// 最大值融合
	MaxValue,
	// 加权平均融合
	WeightedAverage,
	// 光度立体法融合
	PhotometricStereo,
};
// 最大值融合配置
struct MaxValueConfig
{
};

// 加权平均融合配置
struct WeightedAverageConfig
{
	std::vector<double> weights;
	// setConfig 内根据 weights 预计算，用户无需填写
	std::vector<float> normalizedWeights;
};

// 光度立体法输出选项（高度图/曲率图较慢，按需计算）
struct PhotometricStereoOutputFlags
{
	bool computeHeightMap = false;
	bool computeCurvatureMap = false;
	// 合成图使用的虚拟光方向（单位向量）
	cv::Vec3f syntheticLightDir{ 0.f, 0.f, 1.f };
	// 灰度阈值（float 域），低于此值的观测视为阴影不参与拟合；0 表示不过滤
	float intensityThreshold = 0.f;
};

// 光度立体法融合配置
struct PhotometricStereoConfig
{
	LightSystem lightSystem;
	PhotometricStereoOutputFlags outputs;
};

using ImageBlendConfig = std::variant<
	MaxValueConfig,
	WeightedAverageConfig,
	PhotometricStereoConfig
>;

class ImageBlend
{
public:
	ImageBlend();
	~ImageBlend();
	// 设置输入图片
	ImageBlendError setImages(const std::vector<cv::Mat>& images);
	// 设置融合模式和配置
	ImageBlendError setConfig(ImageBlendMode mode, const ImageBlendConfig& config);
	// 执行融合
	ImageBlendError execute();
	// 平板标定：输入>=4张灰度图与共用俯仰角 pitchDeg（度，[0,90)），输出与输入等数量的 LightSource
	// 偏航角由亮区质心相对图像中心估计，俯仰角由 pitchDeg 给定，不使用相机内参
	ImageBlendError executeFlatCalibration(
		const std::vector<cv::Mat>& images,
		float pitchDeg,
		std::vector<LightSource>& outLights);
	// 漫反射球标定：输入>=4张灰度图，输出与输入等数量的 LightSource（按序一一对应）
	// 光方向由球内灰度加权质心处的球面法线估计（Lambert 模型），不使用相机内参
	ImageBlendError executeSphereCalibration(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights);
	// 获取融合结果
	cv::Mat getResult();
	// 获取法线图
	cv::Mat getNormalMap();
	// 获取高度图
	cv::Mat getHeightMap();
	// 获取梯度图
	cv::Mat getGradientMap();
	// 获取漫反射图
	cv::Mat getAlbedoMap();
	// 获取曲率图
	cv::Mat getCurvatureMap();
private:
	void releaseOutputs();
	void releaseAuxMaps();
	void initPhotometricStereoMaps(const cv::Size& size, const PhotometricStereoOutputFlags& flags);
	ImageBlendError validateImages(const std::vector<cv::Mat>& images, size_t minCount = 1);
	ImageBlendError validateConfig(ImageBlendMode blendMode, const ImageBlendConfig& config);
	ImageBlendError calibrateFlatPanel(
		const std::vector<cv::Mat>& images,
		float pitchDeg,
		std::vector<LightSource>& outLights);
	ImageBlendError calibrateSphere(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights);
	void executeMaxValue();
	void executeWeightedAverage(const WeightedAverageConfig& cfg);
	void executePhotometricStereo(const PhotometricStereoConfig& cfg);

	bool imagesSet;
	bool configSet;
	ImageBlendMode mode;
	ImageBlendConfig config;
	std::vector<cv::Mat> imgs;
	cv::Mat result;
	cv::Mat normalMap;
	cv::Mat heightMap;
	cv::Mat gradientMap;
	cv::Mat albedoMap;
	cv::Mat curvatureMap;
};
