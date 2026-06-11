#include "ImageBlend.h"
#include "Profiling.h"
#include <cmath>
#include <algorithm>

namespace
{
	// 单位向量容差
	constexpr float kUnitVectorTolerance = 1e-3f;
	// 漫反射系数容差
	constexpr float kAlbedoEpsilon = 1e-4f;
	// 法线Z分量容差
	constexpr float kNormalZEpsilon = 1e-3f;
	// 平板亮区比例
	constexpr float kFlatBrightRatio = 0.15f;
	constexpr int kFlatMinBrightPixels = 64;
	constexpr float kSphereBodyBrightRatio = 0.12f;
	constexpr float kSphereHighlightRatio = 0.85f;
	constexpr int kSphereMinBodyPixels = 128;
	constexpr int kSphereMinHighlightPixels = 8;
	constexpr float kSphereMinRadius = 8.f;

	// 平板亮区
	struct FlatBrightRegion
	{
		cv::Point2f centroid;
		float meanIntensity = 0.f;
		bool valid = false;
	};

	// 提取平板亮区
	FlatBrightRegion extractFlatBrightRegion(const cv::Mat& gray)
	{
		FlatBrightRegion stats;
		cv::Mat floatImg;
		gray.convertTo(floatImg, CV_32F);

		double maxVal = 0.0;
		cv::minMaxLoc(floatImg, nullptr, &maxVal);
		if (maxVal <= 0.0) return stats;

		const float threshold = static_cast<float>(maxVal * kFlatBrightRatio);
		double sumW = 0.0;
		double sumX = 0.0;
		double sumY = 0.0;
		int count = 0;

		for (int y = 0; y < floatImg.rows; ++y)
		{
			const float* row = floatImg.ptr<float>(y);
			for (int x = 0; x < floatImg.cols; ++x)
			{
				const float v = row[x];
				if (v < threshold) continue;

				sumW += v;
				sumX += v * x;
				sumY += v * y;
				++count;
			}
		}

		if (count < kFlatMinBrightPixels || sumW <= 0.0)
			return stats;

		stats.centroid = cv::Point2f(static_cast<float>(sumX / sumW), static_cast<float>(sumY / sumW));
		stats.meanIntensity = static_cast<float>(sumW / count);
		stats.valid = true;
		return stats;
	}

	// 平板亮区质心转换为光源方向
	cv::Vec3f flatCentroidToLightDir(const cv::Point2f& centroid, const cv::Size& size, bool& ok)
	{
		ok = false;
		const float cx = size.width * 0.5f;
		const float cy = size.height * 0.5f;
		cv::Vec3f dir(centroid.x - cx, centroid.y - cy, 1.f);
		const float norm = cv::norm(dir);
		if (norm < 1e-6f) return cv::Vec3f();

		dir /= norm;
		if (dir[2] < 0.f) dir = -dir;
		ok = true;
		return dir;
	}

	// 反射向量
	cv::Vec3f reflectVector(const cv::Vec3f& v, const cv::Vec3f& n)
	{
		return v - 2.f * v.dot(n) * n;
	}

	// 球高光
	struct SphereHighlight
	{
		cv::Point2f centroid;
		cv::Point2f center;
		float radius = 0.f;
		float meanIntensity = 0.f;
		bool valid = false;
	};

	// 提取球高光
	SphereHighlight extractSphereHighlight(const cv::Mat& gray)
	{
		SphereHighlight stats;
		cv::Mat floatImg;
		gray.convertTo(floatImg, CV_32F);

		double maxVal = 0.0;
		cv::minMaxLoc(floatImg, nullptr, &maxVal);
		if (maxVal <= 0.0) return stats;

		cv::Mat bodyMask;
		cv::compare(floatImg, static_cast<float>(maxVal * kSphereBodyBrightRatio), bodyMask, cv::CMP_GE);
		bodyMask.convertTo(bodyMask, CV_8UC1);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(bodyMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if (contours.empty()) return stats;

		size_t bestIdx = 0;
		double bestArea = 0.0;
		for (size_t i = 0; i < contours.size(); ++i)
		{
			const double area = cv::contourArea(contours[i]);
			if (area > bestArea)
			{
				bestArea = area;
				bestIdx = i;
			}
		}
		if (bestArea < kSphereMinBodyPixels) return stats;

		cv::Point2f center;
		float radius = 0.f;
		cv::minEnclosingCircle(contours[bestIdx], center, radius);
		if (radius < kSphereMinRadius) return stats;

		stats.center = center;
		stats.radius = radius;

		cv::Mat sphereMask = cv::Mat::zeros(floatImg.size(), CV_8UC1);
		cv::circle(sphereMask, center, static_cast<int>(std::round(radius)), cv::Scalar(255), cv::FILLED);

		const float highlightThreshold = static_cast<float>(maxVal * kSphereHighlightRatio);
		double sumW = 0.0;
		double sumX = 0.0;
		double sumY = 0.0;
		int count = 0;

		for (int y = 0; y < floatImg.rows; ++y)
		{
			const float* row = floatImg.ptr<float>(y);
			const uchar* maskRow = sphereMask.ptr<uchar>(y);
			for (int x = 0; x < floatImg.cols; ++x)
			{
				if (maskRow[x] == 0) continue;

				const float v = row[x];
				if (v < highlightThreshold) continue;

				sumW += v;
				sumX += v * x;
				sumY += v * y;
				++count;
			}
		}

		if (count < kSphereMinHighlightPixels || sumW <= 0.0) return stats;

		stats.centroid = cv::Point2f(static_cast<float>(sumX / sumW), static_cast<float>(sumY / sumW));
		stats.meanIntensity = static_cast<float>(sumW / count);
		stats.valid = true;
		return stats;
	}

	// 球高光法线
	cv::Vec3f sphereNormalAtHighlight(const SphereHighlight& highlight, bool& ok)
	{
		ok = false;
		const float rx = (highlight.centroid.x - highlight.center.x) / highlight.radius;
		const float ry = (highlight.centroid.y - highlight.center.y) / highlight.radius;
		const float rr2 = rx * rx + ry * ry;
		if (rr2 >= 1.f) return cv::Vec3f();

		const float rz = std::sqrt(1.f - rr2);
		cv::Vec3f normal(rx, ry, rz);
		const float norm = cv::norm(normal);
		if (norm < 1e-6f) return cv::Vec3f();

		normal /= norm;
		ok = true;
		return normal;
	}

	// 球高光转换为光源方向
	cv::Vec3f highlightToLightDir(
		const SphereHighlight& highlight,
		const cv::Size& size,
		bool& ok)
	{
		ok = false;

		bool normalOk = false;
		const cv::Vec3f normal = sphereNormalAtHighlight(highlight, normalOk);
		if (!normalOk) return cv::Vec3f();

		bool rayOk = false;
		const cv::Vec3f viewRay = flatCentroidToLightDir(highlight.centroid, size, rayOk);
		if (!rayOk) return cv::Vec3f();

		const cv::Vec3f viewToCamera = -viewRay;
		cv::Vec3f lightDir = reflectVector(viewToCamera, normal);
		const float norm = cv::norm(lightDir);
		if (norm < 1e-6f) return cv::Vec3f();

		lightDir /= norm;
		if (lightDir[2] < 0.f) lightDir = -lightDir;
		ok = true;
		return lightDir;
	}

	bool isZeroVector(const cv::Vec3f& dir)
	{
		return dir[0] == 0.f && dir[1] == 0.f && dir[2] == 0.f;
	}

	bool isUnitVector(const cv::Vec3f& dir)
	{
		return std::abs(dir.dot(dir) - 1.f) <= kUnitVectorTolerance;
	}

	// 构建缩放后的光源矩阵
	cv::Mat buildScaledLightMatrix(const std::vector<LightSource>& lights)
	{
		const int k = static_cast<int>(lights.size());
		cv::Mat L(k, 3, CV_32F);
		for (int i = 0; i < k; ++i)
		{
			const float scale = lights[i].calib.intensity * lights[i].calib.attenuation;
			const cv::Vec3f& dir = lights[i].geo.dir;
			L.at<float>(i, 0) = dir[0] * scale;
			L.at<float>(i, 1) = dir[1] * scale;
			L.at<float>(i, 2) = dir[2] * scale;
		}
		return L;
	}

	// 计算光源矩阵的伪逆
	cv::Mat computeLightPseudoInverse(const cv::Mat& L)
	{
		cv::Mat LtL = L.t() * L;
		cv::Mat LtLInv;
		cv::invert(LtL, LtLInv, cv::DECOMP_SVD);
		return LtLInv * L.t();
	}

	// 光度立体法求解法线图和漫反射图
	void solvePhotometricStereoMaps(
		const std::vector<cv::Mat>& floatImgs,
		const cv::Mat& M,
		const std::vector<LightSource>& lights,
		float intensityThreshold,
		cv::Mat& normalOut,
		cv::Mat& albedoOut)
	{
		ZoneScopedN("solvePhotometricStereoMaps");
		const int k = static_cast<int>(floatImgs.size());
		const int rows = floatImgs[0].rows;
		const int cols = floatImgs[0].cols;
		const int pixelCount = rows * cols;

		cv::Mat I(k, pixelCount, CV_32F);
		for (int i = 0; i < k; ++i)
		{
			cv::Mat flat = floatImgs[i].reshape(1, 1);
			flat.copyTo(I.row(i));
		}

		cv::Mat G = M * I;
		normalOut.create(rows, cols, CV_32FC3);
		albedoOut.create(rows, cols, CV_32FC1);

		const float* gx = G.ptr<float>(0);
		const float* gy = G.ptr<float>(1);
		const float* gz = G.ptr<float>(2);

		for (int y = 0; y < rows; ++y)
		{
			cv::Vec3f* normalRow = normalOut.ptr<cv::Vec3f>(y);
			float* albedoRow = albedoOut.ptr<float>(y);
			for (int x = 0; x < cols; ++x)
			{
				const int idx = y * cols + x;
				cv::Vec3f g(gx[idx], gy[idx], gz[idx]);

				if (intensityThreshold > 0.f)
				{
					int validCount = 0;
					for (int i = 0; i < k; ++i)
					{
						if (floatImgs[i].at<float>(y, x) > intensityThreshold)
							++validCount;
					}
					if (validCount < 3)
					{
						normalRow[x] = cv::Vec3f(0.f, 0.f, 1.f);
						albedoRow[x] = 0.f;
						continue;
					}

					cv::Mat Lsub(validCount, 3, CV_32F);
					cv::Mat Isub(validCount, 1, CV_32F);
					int row = 0;
					for (int i = 0; i < k; ++i)
					{
						const float v = floatImgs[i].at<float>(y, x);
						if (v <= intensityThreshold) continue;

						const float scale = lights[i].calib.intensity * lights[i].calib.attenuation;
						const cv::Vec3f& dir = lights[i].geo.dir;
						Lsub.at<float>(row, 0) = dir[0] * scale;
						Lsub.at<float>(row, 1) = dir[1] * scale;
						Lsub.at<float>(row, 2) = dir[2] * scale;
						Isub.at<float>(row, 0) = v;
						++row;
					}
					cv::Mat gMat;
					cv::solve(Lsub, Isub, gMat, cv::DECOMP_SVD);
					g = cv::Vec3f(gMat.at<float>(0, 0), gMat.at<float>(1, 0), gMat.at<float>(2, 0));
				}

				const float rho = cv::norm(g);
				if (rho < kAlbedoEpsilon)
				{
					normalRow[x] = cv::Vec3f(0.f, 0.f, 1.f);
					albedoRow[x] = 0.f;
					continue;
				}

				normalRow[x] = g / rho;
				albedoRow[x] = rho;
			}
		}
	}

	// 计算法线图的梯度
	void computeGradientFromNormal(const cv::Mat& normalMap, cv::Mat& gradientOut)
	{
		gradientOut.create(normalMap.size(), CV_32FC2);
		for (int y = 0; y < normalMap.rows; ++y)
		{
			const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
			cv::Vec2f* gradRow = gradientOut.ptr<cv::Vec2f>(y);
			for (int x = 0; x < normalMap.cols; ++x)
			{
				const cv::Vec3f& n = normalRow[x];
				if (std::abs(n[2]) < kNormalZEpsilon)
				{
					gradRow[x] = cv::Vec2f(0.f, 0.f);
					continue;
				}
				gradRow[x] = cv::Vec2f(-n[0] / n[2], -n[1] / n[2]);
			}
		}
	}

	// 从梯度图积分高度图
	void integrateHeightFromGradient(const cv::Mat& gradientMap, cv::Mat& heightOut)
	{
		const int rows = gradientMap.rows;
		const int cols = gradientMap.cols;

		std::vector<cv::Mat> gradChannels(2);
		cv::split(gradientMap, gradChannels);
		cv::Mat p = gradChannels[0];
		cv::Mat q = gradChannels[1];

		cv::Mat P, Q;
		cv::dct(p, P);
		cv::dct(q, Q);

		cv::Mat Z = cv::Mat::zeros(rows, cols, CV_32F);
		for (int v = 0; v < rows; ++v)
		{
			for (int u = 0; u < cols; ++u)
			{
				if (u == 0 && v == 0) continue;

				const float wx = static_cast<float>(CV_PI * u / cols);
				const float wy = static_cast<float>(CV_PI * v / rows);
				const float denom = wx * wx + wy * wy;
				Z.at<float>(v, u) = (-wx * P.at<float>(v, u) - wy * Q.at<float>(v, u)) / denom;
			}
		}

		cv::idct(Z, heightOut);
	}

	// 计算法线图的曲率
	void computeCurvatureFromNormal(const cv::Mat& normalMap, cv::Mat& curvatureOut)
	{
		std::vector<cv::Mat> normalChannels(3);
		cv::split(normalMap, normalChannels);

		cv::Mat dnxDx, dnyDy;
		cv::Sobel(normalChannels[0], dnxDx, CV_32F, 1, 0, 3);
		cv::Sobel(normalChannels[1], dnyDy, CV_32F, 0, 1, 3);
		cv::add(dnxDx, dnyDy, curvatureOut);
	}

	// 从高度图计算曲率
	void computeCurvatureFromHeight(const cv::Mat& heightMap, cv::Mat& curvatureOut)
	{
		cv::Laplacian(heightMap, curvatureOut, CV_32F, 3);
	}

	// 渲染合成图
	void renderSyntheticImage(
		const cv::Mat& normalMap,
		const cv::Mat& albedoMap,
		const cv::Vec3f& lightDir,
		int outputType,
		cv::Mat& resultOut)
	{
		resultOut.create(normalMap.size(), outputType);
		for (int y = 0; y < normalMap.rows; ++y)
		{
			const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
			const float* albedoRow = albedoMap.ptr<float>(y);
			for (int x = 0; x < normalMap.cols; ++x)
			{
				const float shade = std::max(0.f, normalRow[x].dot(lightDir));
				const float value = albedoRow[x] * shade;
				resultOut.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
			}
		}
	}
}

ImageBlend::ImageBlend()
	: imagesSet(false)
	, configSet(false)
	, mode(ImageBlendMode::MaxValue)
	, config(MaxValueConfig{})
{
}

ImageBlend::~ImageBlend()
{
}

void ImageBlend::releaseOutputs()
{
	result.release();
	releaseAuxMaps();
}

void ImageBlend::releaseAuxMaps()
{
	normalMap.release();
	heightMap.release();
	gradientMap.release();
	albedoMap.release();
	curvatureMap.release();
}

// 验证输入图片
// - 必须为灰度图，统一类型
// - 光度立体合成最少4张图片
ImageBlendError ImageBlend::validateImages(const std::vector<cv::Mat>& images, size_t minCount)
{
	if (images.size() < minCount) return ImageBlendError::ImageNumberError;

	for (const auto& img : images)
	{
		if (img.empty()) return ImageBlendError::NullImg;
	}

	const cv::Size size = images[0].size();
	const int type = images[0].type();
	for (const auto& img : images)
	{
		if (img.size() != size) return ImageBlendError::SizeMismatch;
		if (img.channels() != 1) return ImageBlendError::NotGrayscale;
		if (img.type() != type) return ImageBlendError::ImageTypeDifferent;
	}

	return ImageBlendError::OK;
}

// 设置输入图片
ImageBlendError ImageBlend::setImages(const std::vector<cv::Mat>& images)
{
	ImageBlendError err = validateImages(images, 1);
	if (err != ImageBlendError::OK) return err;

	imgs = images;
	imagesSet = true;
	configSet = false;
	releaseOutputs();
	return ImageBlendError::OK;
}

// 验证合成配置
ImageBlendError ImageBlend::validateConfig(ImageBlendMode blendMode, const ImageBlendConfig& blendConfig)
{
	switch (blendMode)
	{
	case ImageBlendMode::MaxValue:
		if (!std::holds_alternative<MaxValueConfig>(blendConfig))
			return ImageBlendError::ConfigMismatch;
		break;

	case ImageBlendMode::WeightedAverage:
	{
		if (!std::holds_alternative<WeightedAverageConfig>(blendConfig))
			return ImageBlendError::ConfigMismatch;

		const auto& cfg = std::get<WeightedAverageConfig>(blendConfig);
		if (cfg.weights.empty()) return ImageBlendError::WeightError;
		if (cfg.weights.size() != imgs.size()) return ImageBlendError::WeightError;

		double weightSum = 0.0;
		for (double w : cfg.weights)
		{
			if (w < 0.0) return ImageBlendError::WeightError;
			weightSum += w;
		}
		if (weightSum <= 0.0) return ImageBlendError::WeightError;
		break;
	}

	case ImageBlendMode::PhotometricStereo:
	{
		if (!std::holds_alternative<PhotometricStereoConfig>(blendConfig))
			return ImageBlendError::ConfigMismatch;

		const auto& cfg = std::get<PhotometricStereoConfig>(blendConfig);
		const auto& lights = cfg.lightSystem.lights;
		if (imgs.size() < 4) return ImageBlendError::ImageNumberError;
		if (lights.size() != imgs.size()) return ImageBlendError::LightNumberError;
		if (lights.size() < 4) return ImageBlendError::LightNumberError;

		for (const auto& light : lights)
		{
			if (isZeroVector(light.geo.dir)) return ImageBlendError::LightParameterError;
			if (!isUnitVector(light.geo.dir)) return ImageBlendError::LightParameterError;
			if (light.calib.intensity <= 0.0f) return ImageBlendError::LightParameterError;
			if (light.calib.attenuation <= 0.0f) return ImageBlendError::LightParameterError;
		}

		const cv::Vec3f& synDir = cfg.outputs.syntheticLightDir;
		if (isZeroVector(synDir)) return ImageBlendError::LightParameterError;
		if (!isUnitVector(synDir)) return ImageBlendError::LightParameterError;
		break;
	}
	}

	return ImageBlendError::OK;
}

// 设置融合模式和配置
ImageBlendError ImageBlend::setConfig(ImageBlendMode blendMode, const ImageBlendConfig& blendConfig)
{
	if (!imagesSet) return ImageBlendError::ParameterError;

	ImageBlendError err = validateConfig(blendMode, blendConfig);
	if (err != ImageBlendError::OK) return err;

	mode = blendMode;
	config = blendConfig;
	configSet = true;
	releaseOutputs();
	return ImageBlendError::OK;
}

// 光度立体法：仅分配需要输出的辅助图
void ImageBlend::initPhotometricStereoMaps(const cv::Size& size, const PhotometricStereoOutputFlags& flags)
{
	normalMap = cv::Mat::zeros(size, CV_32FC3);
	gradientMap = cv::Mat::zeros(size, CV_32FC2);
	albedoMap = cv::Mat::zeros(size, CV_32FC1);

	if (flags.computeHeightMap)
		heightMap = cv::Mat::zeros(size, CV_32FC1);

	if (flags.computeCurvatureMap)
		curvatureMap = cv::Mat::zeros(size, CV_32FC1);
}

// 最大值融合
void ImageBlend::executeMaxValue()
{
	ZoneScoped;
	result = imgs[0].clone();
	for (size_t i = 1; i < imgs.size(); ++i)
		cv::max(result, imgs[i], result);
}

// 加权平均融合
void ImageBlend::executeWeightedAverage(const WeightedAverageConfig& cfg)
{
	ZoneScoped;
	double weightSum = 0.0;
	for (double w : cfg.weights) weightSum += w;

	cv::Mat acc;
	imgs[0].convertTo(acc, CV_64F, cfg.weights[0] / weightSum);
	for (size_t i = 1; i < imgs.size(); ++i)
	{
		cv::Mat temp;
		imgs[i].convertTo(temp, CV_64F, cfg.weights[i] / weightSum);
		acc += temp;
	}
	acc.convertTo(result, imgs[0].type());
}

// 光度立体法融合
void ImageBlend::executePhotometricStereo(const PhotometricStereoConfig& cfg)
{
	ZoneScoped;
	const auto& lights = cfg.lightSystem.lights;
	const PhotometricStereoOutputFlags& flags = cfg.outputs;

	initPhotometricStereoMaps(imgs[0].size(), flags);

	std::vector<cv::Mat> floatImgs(imgs.size());
	for (size_t i = 0; i < imgs.size(); ++i)
		imgs[i].convertTo(floatImgs[i], CV_32F);

	const cv::Mat L = buildScaledLightMatrix(lights);
	const cv::Mat M = computeLightPseudoInverse(L);

	solvePhotometricStereoMaps(floatImgs, M, lights, flags.intensityThreshold, normalMap, albedoMap);
	computeGradientFromNormal(normalMap, gradientMap);

	if (flags.computeHeightMap)
		integrateHeightFromGradient(gradientMap, heightMap);

	if (flags.computeCurvatureMap)
	{
		if (flags.computeHeightMap && !heightMap.empty())
			computeCurvatureFromHeight(heightMap, curvatureMap);
		else
			computeCurvatureFromNormal(normalMap, curvatureMap);
	}

	renderSyntheticImage(normalMap, albedoMap, flags.syntheticLightDir, imgs[0].type(), result);
}

// 漫反射平板标定
// 假设平板法线为 (0,0,1)，每张图对应一次打光：
// 1. 亮区加权质心相对图像中心 -> 光源方向 normalize(du, dv, 1)
// 2. 亮区平均灰度 / lz -> 相对光强，最后归一化
ImageBlendError ImageBlend::calibrateFlatPanel(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights)
{
	ZoneScoped;
	const cv::Size size = images[0].size();
	const size_t n = images.size();
	outLights.clear();
	outLights.reserve(n);

	std::vector<float> rawIntensities;
	rawIntensities.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const FlatBrightRegion region = extractFlatBrightRegion(images[i]);
		if (!region.valid)
			return ImageBlendError::ParameterError;

		bool dirOk = false;
		const cv::Vec3f dir = flatCentroidToLightDir(region.centroid, size, dirOk);
		if (!dirOk || dir[2] < kNormalZEpsilon)
			return ImageBlendError::ParameterError;

		LightSource light;
		light.geo.dir = dir;
		light.calib.attenuation = 1.0f;
		light.calib.intensity = region.meanIntensity / dir[2];
		rawIntensities.push_back(light.calib.intensity);
		outLights.push_back(light);
	}

	float intensitySum = 0.f;
	for (float v : rawIntensities) intensitySum += v;
	if (intensitySum <= 1e-6f)
		return ImageBlendError::ParameterError;

	const float intensityMean = intensitySum / static_cast<float>(n);
	for (size_t i = 0; i < n; ++i)
		outLights[i].calib.intensity = rawIntensities[i] / intensityMean;

	return ImageBlendError::OK;
}

// 镜面球标定
// 1. 检测球体轮廓（亮区最大连通域 + 最小外接圆）
// 2. 球内高光加权质心
// 3. 球面法线 + 反射定律求光源方向（视线由高光像素与图像中心计算，不使用相机内参）
// 4. 高光强度归一化
ImageBlendError ImageBlend::calibrateSphere(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights)
{
	ZoneScoped;
	const cv::Size size = images[0].size();

	const size_t n = images.size();
	outLights.clear();
	outLights.reserve(n);

	std::vector<float> rawIntensities;
	rawIntensities.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const SphereHighlight highlight = extractSphereHighlight(images[i]);
		if (!highlight.valid)
			return ImageBlendError::ParameterError;

		bool dirOk = false;
		const cv::Vec3f dir = highlightToLightDir(highlight, size, dirOk);
		if (!dirOk || dir[2] < kNormalZEpsilon)
			return ImageBlendError::ParameterError;

		LightSource light;
		light.geo.dir = dir;
		light.calib.attenuation = 1.0f;
		light.calib.intensity = highlight.meanIntensity;
		rawIntensities.push_back(light.calib.intensity);
		outLights.push_back(light);
	}

	float intensitySum = 0.f;
	for (float v : rawIntensities) intensitySum += v;
	if (intensitySum <= 1e-6f)
		return ImageBlendError::ParameterError;

	const float intensityMean = intensitySum / static_cast<float>(n);
	for (size_t i = 0; i < n; ++i)
		outLights[i].calib.intensity = rawIntensities[i] / intensityMean;

	return ImageBlendError::OK;
}

ImageBlendError ImageBlend::executeFlatCalibration(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights)
{
	ImageBlendError err = validateImages(images, 4);
	if (err != ImageBlendError::OK) return err;

	return calibrateFlatPanel(images, outLights);
}

ImageBlendError ImageBlend::executeSphereCalibration(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights)
{
	ImageBlendError err = validateImages(images, 4);
	if (err != ImageBlendError::OK) return err;

	return calibrateSphere(images, outLights);
}

ImageBlendError ImageBlend::execute()
{
	ZoneScoped;
	if (!imagesSet || !configSet) return ImageBlendError::ParameterError;

	switch (mode)
	{
	case ImageBlendMode::MaxValue:
		releaseAuxMaps();
		executeMaxValue();
		break;
	case ImageBlendMode::WeightedAverage:
		releaseAuxMaps();
		executeWeightedAverage(std::get<WeightedAverageConfig>(config));
		break;
	case ImageBlendMode::PhotometricStereo:
		if (imgs.size() < 4) return ImageBlendError::ImageNumberError;
		releaseAuxMaps();
		executePhotometricStereo(std::get<PhotometricStereoConfig>(config));
		break;
	}

	return ImageBlendError::OK;
}

cv::Mat ImageBlend::getResult()
{
	if (result.empty()) return cv::Mat();
	return result;
}

cv::Mat ImageBlend::getNormalMap()
{
	if (normalMap.empty()) return cv::Mat();
	return normalMap;
}

cv::Mat ImageBlend::getHeightMap()
{
	if (heightMap.empty()) return cv::Mat();
	return heightMap;
}

cv::Mat ImageBlend::getGradientMap()
{
	if (gradientMap.empty()) return cv::Mat();
	return gradientMap;
}

cv::Mat ImageBlend::getAlbedoMap()
{
	if (albedoMap.empty()) return cv::Mat();
	return albedoMap;
}

cv::Mat ImageBlend::getCurvatureMap()
{
	if (curvatureMap.empty()) return cv::Mat();
	return curvatureMap;
}
