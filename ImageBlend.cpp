#include "ImageBlend.h"
#include "ImageBlendSimd.h"
#include "Profiling.h"
#include <cmath>
#include <algorithm>

namespace
{
	// PS
	constexpr float kUnitVectorTolerance = 1e-3f;
	constexpr float kAlbedoEpsilon = 1e-4f;
	constexpr float kNormalZEpsilon = 1e-3f;

	// 平板亮区参数
	constexpr float kFlatBrightRatio = 0.15f;
	constexpr int kFlatMinBrightPixels = 64;

	// 漫反射球亮区参数
	constexpr float kSphereBodyBrightRatio = 0.12f;
	constexpr int kSphereMinBodyPixels = 128;
	constexpr int kSphereMinDiffusePixels = 64;
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

	// 平板标定：手动俯仰角 + 亮区质心偏航角 -> 光源方向
	// pitchDeg: 与 +Z 光轴夹角，度，范围 [0, 90)
	cv::Vec3f flatCentroidPitchToLightDir(
		const cv::Point2f& centroid,
		const cv::Size& size,
		float pitchDeg,
		bool& ok)
	{
		ok = false;
		const float cx = size.width * 0.5f;
		const float cy = size.height * 0.5f;
		const float du = centroid.x - cx;
		const float dv = centroid.y - cy;

		float yaw = 0.f;
		if (du * du + dv * dv > 1e-12f)
			yaw = std::atan2(dv, du);

		const float pitchRad = pitchDeg * (static_cast<float>(CV_PI) / 180.f);
		const float sinPitch = std::sin(pitchRad);
		const float cosPitch = std::cos(pitchRad);
		cv::Vec3f dir(
			sinPitch * std::cos(yaw),
			sinPitch * std::sin(yaw),
			cosPitch);

		if (dir[2] < 0.f) dir = -dir;
		ok = true;
		return dir;
	}

	// 漫反射球亮区
	struct DiffuseSphereRegion
	{
		cv::Point2f centroid;
		cv::Point2f center;
		float radius = 0.f;
		float peakIntensity = 0.f;
		bool valid = false;
	};

	// 提取漫反射球：检测球体轮廓，球内灰度加权质心估计亮区中心
	DiffuseSphereRegion extractDiffuseSphereRegion(const cv::Mat& gray)
	{
		DiffuseSphereRegion stats;
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

		double sumW = 0.0;
		double sumX = 0.0;
		double sumY = 0.0;
		int count = 0;
		float peakIntensity = 0.f;

		for (int y = 0; y < floatImg.rows; ++y)
		{
			const float* row = floatImg.ptr<float>(y);
			const uchar* maskRow = sphereMask.ptr<uchar>(y);
			for (int x = 0; x < floatImg.cols; ++x)
			{
				if (maskRow[x] == 0) continue;

				const float v = row[x];
				if (v <= 0.f) continue;

				sumW += v;
				sumX += v * x;
				sumY += v * y;
				peakIntensity = std::max(peakIntensity, v);
				++count;
			}
		}

		if (count < kSphereMinDiffusePixels || sumW <= 0.0 || peakIntensity <= 0.f)
			return stats;

		stats.centroid = cv::Point2f(static_cast<float>(sumX / sumW), static_cast<float>(sumY / sumW));
		stats.peakIntensity = peakIntensity;
		stats.valid = true;
		return stats;
	}

	// 漫反射球亮区质心处的球面外法线
	cv::Vec3f sphereNormalAtPoint(const DiffuseSphereRegion& region, bool& ok)
	{
		ok = false;
		const float rx = (region.centroid.x - region.center.x) / region.radius;
		const float ry = (region.centroid.y - region.center.y) / region.radius;
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

	// 漫反射球：Lambert 模型下亮区法线即光源方向
	cv::Vec3f diffuseSphereToLightDir(const DiffuseSphereRegion& region, bool& ok)
	{
		ok = false;

		bool normalOk = false;
		cv::Vec3f lightDir = sphereNormalAtPoint(region, normalOk);
		if (!normalOk) return cv::Vec3f();

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

	bool isSupportedInputType(int type)
	{
		return type == CV_8UC1 || type == CV_32FC1;
	}

	struct ScaledLight
	{
		float sx = 0.f;
		float sy = 0.f;
		float sz = 0.f;
	};

	std::vector<ScaledLight> buildScaledLights(const std::vector<LightSource>& lights)
	{
		std::vector<ScaledLight> scaled(lights.size());
		for (size_t i = 0; i < lights.size(); ++i)
		{
			const float scale = lights[i].calib.intensity * lights[i].calib.attenuation;
			const cv::Vec3f& dir = lights[i].geo.dir;
			scaled[i].sx = dir[0] * scale;
			scaled[i].sy = dir[1] * scale;
			scaled[i].sz = dir[2] * scale;
		}
		return scaled;
	}

	void writeNormalAlbedoFromG(float gx, float gy, float gz, cv::Vec3f& normalOut, float& albedoOut)
	{
		const float rho = std::sqrt(gx * gx + gy * gy + gz * gz);
		if (rho < kAlbedoEpsilon)
		{
			normalOut = cv::Vec3f(0.f, 0.f, 1.f);
			albedoOut = 0.f;
			return;
		}

		const float invRho = 1.f / rho;
		normalOut[0] = gx * invRho;
		normalOut[1] = gy * invRho;
		normalOut[2] = gz * invRho;
		albedoOut = rho;
	}

	bool invert3x3(const float a[3][3], float invOut[3][3])
	{
		cv::Mat mat(3, 3, CV_32F);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				mat.at<float>(r, c) = a[r][c];

		cv::Mat invMat;
		if (!cv::invert(mat, invMat, cv::DECOMP_LU))
			return false;

		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				invOut[r][c] = invMat.at<float>(r, c);
		return true;
	}

	void accumulateNormalEquations(
		const float* lRows,
		const float* iValues,
		int rowCount,
		float LtL[3][3],
		float LtI[3])
	{
		for (int r = 0; r < rowCount; ++r)
		{
			const float sx = lRows[r * 3 + 0];
			const float sy = lRows[r * 3 + 1];
			const float sz = lRows[r * 3 + 2];
			const float iv = iValues[r];

			LtL[0][0] += sx * sx;
			LtL[0][1] += sx * sy;
			LtL[0][2] += sx * sz;
			LtL[1][0] += sy * sx;
			LtL[1][1] += sy * sy;
			LtL[1][2] += sy * sz;
			LtL[2][0] += sz * sx;
			LtL[2][1] += sz * sy;
			LtL[2][2] += sz * sz;
			LtI[0] += sx * iv;
			LtI[1] += sy * iv;
			LtI[2] += sz * iv;
		}
	}

	bool solveLeastSquaresPhotometric(
		const float* lRows,
		const float* iValues,
		int rowCount,
		float& gx,
		float& gy,
		float& gz)
	{
		if (rowCount < 3) return false;

		float LtL[3][3] = {};
		float LtI[3] = {};
		accumulateNormalEquations(lRows, iValues, rowCount, LtL, LtI);

		float LtLInv[3][3];
		if (!invert3x3(LtL, LtLInv))
			return false;

		gx = LtLInv[0][0] * LtI[0] + LtLInv[0][1] * LtI[1] + LtLInv[0][2] * LtI[2];
		gy = LtLInv[1][0] * LtI[0] + LtLInv[1][1] * LtI[1] + LtLInv[1][2] * LtI[2];
		gz = LtLInv[2][0] * LtI[0] + LtLInv[2][1] * LtI[1] + LtLInv[2][2] * LtI[2];
		return true;
	}

	// k=4 阴影 mask LUT：16 组 × 3×4 系数，无效 mask 保持全零
	void buildThresholdLutK4(const std::vector<float>& scaledLightsFlat, std::vector<float>& lutOut)
	{
		lutOut.assign(16 * 12, 0.f);
		if (scaledLightsFlat.size() != 12) return;

		for (int mask = 1; mask < 16; ++mask)
		{
			int indices[4];
			float lRows[4 * 3];
			int rowCount = 0;
			for (int i = 0; i < 4; ++i)
			{
				if ((mask & (1 << i)) == 0) continue;
				indices[rowCount] = i;
				const size_t base = static_cast<size_t>(i * 3);
				lRows[rowCount * 3 + 0] = scaledLightsFlat[base + 0];
				lRows[rowCount * 3 + 1] = scaledLightsFlat[base + 1];
				lRows[rowCount * 3 + 2] = scaledLightsFlat[base + 2];
				++rowCount;
			}
			if (rowCount < 3) continue;

			float LtL[3][3] = {};
			float LtLInv[3][3];
			float dummyI[4] = {};
			accumulateNormalEquations(lRows, dummyI, rowCount, LtL, dummyI);
			if (!invert3x3(LtL, LtLInv)) continue;

			float* lutMask = lutOut.data() + mask * 12;
			for (int j = 0; j < rowCount; ++j)
			{
				const float sx = lRows[j * 3 + 0];
				const float sy = lRows[j * 3 + 1];
				const float sz = lRows[j * 3 + 2];
				const int lightIdx = indices[j];
				lutMask[0 * 4 + lightIdx] = LtLInv[0][0] * sx + LtLInv[0][1] * sy + LtLInv[0][2] * sz;
				lutMask[1 * 4 + lightIdx] = LtLInv[1][0] * sx + LtLInv[1][1] * sy + LtLInv[1][2] * sz;
				lutMask[2 * 4 + lightIdx] = LtLInv[2][0] * sx + LtLInv[2][1] * sy + LtLInv[2][2] * sz;
			}
		}
	}

	// k=6 阴影 mask LUT：64 组 × 3×6 系数，无效 mask 保持全零
	void buildThresholdLutK6(const std::vector<float>& scaledLightsFlat, std::vector<float>& lutOut)
	{
		lutOut.assign(64 * 18, 0.f);
		if (scaledLightsFlat.size() != 18) return;

		for (int mask = 1; mask < 64; ++mask)
		{
			int indices[6];
			float lRows[6 * 3];
			int rowCount = 0;
			for (int i = 0; i < 6; ++i)
			{
				if ((mask & (1 << i)) == 0) continue;
				indices[rowCount] = i;
				const size_t base = static_cast<size_t>(i * 3);
				lRows[rowCount * 3 + 0] = scaledLightsFlat[base + 0];
				lRows[rowCount * 3 + 1] = scaledLightsFlat[base + 1];
				lRows[rowCount * 3 + 2] = scaledLightsFlat[base + 2];
				++rowCount;
			}
			if (rowCount < 3) continue;

			float LtL[3][3] = {};
			float LtLInv[3][3];
			float dummyI[6] = {};
			accumulateNormalEquations(lRows, dummyI, rowCount, LtL, dummyI);
			if (!invert3x3(LtL, LtLInv)) continue;

			float* lutMask = lutOut.data() + mask * 18;
			for (int j = 0; j < rowCount; ++j)
			{
				const float sx = lRows[j * 3 + 0];
				const float sy = lRows[j * 3 + 1];
				const float sz = lRows[j * 3 + 2];
				const int lightIdx = indices[j];
				lutMask[0 * 6 + lightIdx] = LtLInv[0][0] * sx + LtLInv[0][1] * sy + LtLInv[0][2] * sz;
				lutMask[1 * 6 + lightIdx] = LtLInv[1][0] * sx + LtLInv[1][1] * sy + LtLInv[1][2] * sz;
				lutMask[2 * 6 + lightIdx] = LtLInv[2][0] * sx + LtLInv[2][1] * sy + LtLInv[2][2] * sz;
			}
		}
	}

	void convertGrayImagesToFloatParallel(
		const std::vector<cv::Mat>& imgs,
		std::vector<cv::Mat>& floatImgsOut)
	{
		ZoneScopedN("convertGrayImagesToFloatParallel");
		const int count = static_cast<int>(imgs.size());
		floatImgsOut.resize(static_cast<size_t>(count));
		cv::parallel_for_(cv::Range(0, count), [&](const cv::Range& range) {
			for (int i = range.start; i < range.end; ++i)
				imgs[static_cast<size_t>(i)].convertTo(floatImgsOut[static_cast<size_t>(i)], CV_32F);
		});
	}

	inline void ensureMatBuffer(cv::Mat& mat, const cv::Size& size, int type)
	{
		mat.create(size, type);
	}

	// 由缩放光源矩阵计算 3×k 伪逆系数，布局为 [row0(k), row1(k), row2(k)]
	bool computeLightPseudoInverseCoeffs(
		const std::vector<ScaledLight>& scaledLights,
		std::vector<float>& inverseMOut)
	{
		const int k = static_cast<int>(scaledLights.size());
		if (k < 3) return false;

		cv::Mat LtL = cv::Mat::zeros(3, 3, CV_32F);
		for (const ScaledLight& light : scaledLights)
		{
			const float sx = light.sx;
			const float sy = light.sy;
			const float sz = light.sz;
			LtL.at<float>(0, 0) += sx * sx;
			LtL.at<float>(0, 1) += sx * sy;
			LtL.at<float>(0, 2) += sx * sz;
			LtL.at<float>(1, 0) += sy * sx;
			LtL.at<float>(1, 1) += sy * sy;
			LtL.at<float>(1, 2) += sy * sz;
			LtL.at<float>(2, 0) += sz * sx;
			LtL.at<float>(2, 1) += sz * sy;
			LtL.at<float>(2, 2) += sz * sz;
		}

		cv::Mat LtLInv;
		if (!cv::invert(LtL, LtLInv, cv::DECOMP_LU))
			return false;

		inverseMOut.resize(static_cast<size_t>(3 * k));
		float* m0 = inverseMOut.data();
		float* m1 = m0 + k;
		float* m2 = m0 + 2 * k;

		for (int i = 0; i < k; ++i)
		{
			const float sx = scaledLights[i].sx;
			const float sy = scaledLights[i].sy;
			const float sz = scaledLights[i].sz;
			m0[i] = LtLInv.at<float>(0, 0) * sx + LtLInv.at<float>(0, 1) * sy + LtLInv.at<float>(0, 2) * sz;
			m1[i] = LtLInv.at<float>(1, 0) * sx + LtLInv.at<float>(1, 1) * sy + LtLInv.at<float>(1, 2) * sz;
			m2[i] = LtLInv.at<float>(2, 0) * sx + LtLInv.at<float>(2, 1) * sy + LtLInv.at<float>(2, 2) * sz;
		}
		return true;
	}

	void solvePhotometricStereoMapsFastScalar(
		const std::vector<cv::Mat>& imgs,
		const std::vector<float>& inverseM,
		cv::Mat& normalOut,
		cv::Mat& albedoOut)
	{
		ZoneScopedN("solvePhotometricStereoMapsFastScalar");
		const int k = static_cast<int>(imgs.size());
		const int rows = imgs[0].rows;
		const int cols = imgs[0].cols;
		const int imgType = imgs[0].type();
		const float* m0 = inverseM.data();
		const float* m1 = m0 + k;
		const float* m2 = m0 + 2 * k;

		cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
			std::vector<const uchar*> u8RowPtrs;
			std::vector<const float*> f32RowPtrs;
			if (imgType == CV_8UC1)
				u8RowPtrs.resize(static_cast<size_t>(k));
			else
				f32RowPtrs.resize(static_cast<size_t>(k));

			for (int y = range.start; y < range.end; ++y)
			{
				if (imgType == CV_8UC1)
				{
					for (int i = 0; i < k; ++i)
						u8RowPtrs[static_cast<size_t>(i)] = imgs[static_cast<size_t>(i)].ptr<uchar>(y);
				}
				else
				{
					for (int i = 0; i < k; ++i)
						f32RowPtrs[static_cast<size_t>(i)] = imgs[static_cast<size_t>(i)].ptr<float>(y);
				}

				cv::Vec3f* normalRow = normalOut.ptr<cv::Vec3f>(y);
				float* albedoRow = albedoOut.ptr<float>(y);
				for (int x = 0; x < cols; ++x)
				{
					float gx = 0.f;
					float gy = 0.f;
					float gz = 0.f;
					for (int i = 0; i < k; ++i)
					{
						const float v = (imgType == CV_8UC1)
							? static_cast<float>(u8RowPtrs[static_cast<size_t>(i)][x])
							: f32RowPtrs[static_cast<size_t>(i)][x];
						gx += m0[i] * v;
						gy += m1[i] * v;
						gz += m2[i] * v;
					}
					writeNormalAlbedoFromG(gx, gy, gz, normalRow[x], albedoRow[x]);
				}
			}
		});
	}

	void solvePhotometricStereoMapsFast(
		const std::vector<cv::Mat>& imgs,
		const std::vector<float>& inverseM,
		cv::Mat& normalOut,
		cv::Mat& albedoOut)
	{
		ZoneScopedN("solvePhotometricStereoMapsFast");
		if (solvePhotometricStereoTryAvx2(imgs, inverseM, normalOut, albedoOut))
			return;
		solvePhotometricStereoMapsFastScalar(imgs, inverseM, normalOut, albedoOut);
	}

	void solvePhotometricStereoMapsThreshold(
		const std::vector<cv::Mat>& floatImgs,
		const std::vector<float>& scaledLightsFlat,
		const std::vector<float>& thresholdLutK4,
		const std::vector<float>& thresholdLutK6,
		float intensityThreshold,
		cv::Mat& normalOut,
		cv::Mat& albedoOut)
	{
		ZoneScopedN("solvePhotometricStereoMapsThreshold");
		const int k = static_cast<int>(floatImgs.size());
		const int rows = floatImgs[0].rows;
		const int cols = floatImgs[0].cols;
		const bool useLutK4 = (k == 4 && thresholdLutK4.size() == 16 * 12);
		const bool useLutK6 = (k == 6 && thresholdLutK6.size() == 64 * 18);
		const int lutStride = useLutK4 ? 4 : (useLutK6 ? 6 : 0);
		const std::vector<float>* lutData = useLutK4 ? &thresholdLutK4 : (useLutK6 ? &thresholdLutK6 : nullptr);

		cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
			std::vector<const float*> imgRowPtrs(static_cast<size_t>(k));
			thread_local float lRows[8 * 3];
			thread_local float iValues[8];

			for (int y = range.start; y < range.end; ++y)
			{
				for (int i = 0; i < k; ++i)
					imgRowPtrs[static_cast<size_t>(i)] = floatImgs[static_cast<size_t>(i)].ptr<float>(y);

				cv::Vec3f* normalRow = normalOut.ptr<cv::Vec3f>(y);
				float* albedoRow = albedoOut.ptr<float>(y);
				for (int x = 0; x < cols; ++x)
				{
					int validCount = 0;
					int mask = 0;
					for (int i = 0; i < k; ++i)
					{
						const float v = imgRowPtrs[static_cast<size_t>(i)][x];
						if (v > intensityThreshold)
						{
							++validCount;
							mask |= (1 << i);
						}
					}
					if (validCount < 3)
					{
						normalRow[x] = cv::Vec3f(0.f, 0.f, 1.f);
						albedoRow[x] = 0.f;
						continue;
					}

					float gx = 0.f;
					float gy = 0.f;
					float gz = 0.f;
					if (lutData)
					{
						const float* lutMask = lutData->data() + mask * (3 * lutStride);
						for (int i = 0; i < k; ++i)
						{
							const float v = imgRowPtrs[static_cast<size_t>(i)][x];
							gx += lutMask[0 * lutStride + i] * v;
							gy += lutMask[1 * lutStride + i] * v;
							gz += lutMask[2 * lutStride + i] * v;
						}
					}
					else
					{
						int row = 0;
						for (int i = 0; i < k; ++i)
						{
							const float v = imgRowPtrs[static_cast<size_t>(i)][x];
							if (v <= intensityThreshold) continue;

							const size_t lightIdx = static_cast<size_t>(i * 3);
							lRows[row * 3 + 0] = scaledLightsFlat[lightIdx + 0];
							lRows[row * 3 + 1] = scaledLightsFlat[lightIdx + 1];
							lRows[row * 3 + 2] = scaledLightsFlat[lightIdx + 2];
							iValues[row] = v;
							++row;
						}
						if (!solveLeastSquaresPhotometric(lRows, iValues, validCount, gx, gy, gz))
						{
							normalRow[x] = cv::Vec3f(0.f, 0.f, 1.f);
							albedoRow[x] = 0.f;
							continue;
						}
					}

					writeNormalAlbedoFromG(gx, gy, gz, normalRow[x], albedoRow[x]);
				}
			}
		});
	}

	// 光度立体法求解法线图和漫反射图
	void solvePhotometricStereoMaps(
		const std::vector<cv::Mat>& imgs,
		const std::vector<float>& inverseM,
		const std::vector<float>& scaledLightsFlat,
		const std::vector<float>& thresholdLutK4,
		const std::vector<float>& thresholdLutK6,
		float intensityThreshold,
		cv::Mat& normalOut,
		cv::Mat& albedoOut)
	{
		ZoneScopedN("solvePhotometricStereoMaps");

		if (intensityThreshold <= 0.f)
		{
			solvePhotometricStereoMapsFast(imgs, inverseM, normalOut, albedoOut);
			return;
		}

		std::vector<cv::Mat> floatImgs;
		if (imgs[0].type() == CV_32FC1)
			floatImgs = imgs;
		else
			convertGrayImagesToFloatParallel(imgs, floatImgs);

		solvePhotometricStereoMapsThreshold(
			floatImgs,
			scaledLightsFlat,
			thresholdLutK4,
			thresholdLutK6,
			intensityThreshold,
			normalOut,
			albedoOut);
	}

	// 计算法线图的梯度
	inline void writeGradientFromNormal(float nx, float ny, float nz, cv::Vec2f& gradOut)
	{
		if (std::abs(nz) < kNormalZEpsilon)
		{
			gradOut = cv::Vec2f(0.f, 0.f);
			return;
		}
		const float invNz = 1.f / nz;
		gradOut[0] = -nx * invNz;
		gradOut[1] = -ny * invNz;
	}

	void computeGradientFromNormalScalar(const cv::Mat& normalMap, cv::Mat& gradientOut)
	{
		const int cols = normalMap.cols;
		cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
				cv::Vec2f* gradRow = gradientOut.ptr<cv::Vec2f>(y);
				for (int x = 0; x < cols; ++x)
				{
					const cv::Vec3f& n = normalRow[x];
					writeGradientFromNormal(n[0], n[1], n[2], gradRow[x]);
				}
			}
		});
	}

	void computeGradientPQFromNormalScalar(
		const cv::Mat& normalMap,
		cv::Mat& pOut,
		cv::Mat& qOut)
	{
		const int cols = normalMap.cols;
		cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
				float* pRow = pOut.ptr<float>(y);
				float* qRow = qOut.ptr<float>(y);
				for (int x = 0; x < cols; ++x)
				{
					const cv::Vec3f& n = normalRow[x];
					if (std::abs(n[2]) < kNormalZEpsilon)
					{
						pRow[x] = 0.f;
						qRow[x] = 0.f;
						continue;
					}
					const float invNz = 1.f / n[2];
					pRow[x] = -n[0] * invNz;
					qRow[x] = -n[1] * invNz;
				}
			}
		});
	}

	void computeGradientFromNormal(const cv::Mat& normalMap, cv::Mat& gradientOut)
	{
		ZoneScopedN("computeGradientFromNormal");
		ensureMatBuffer(gradientOut, normalMap.size(), CV_32FC2);
		if (computeGradientFromNormalTryAvx2(normalMap, gradientOut))
			return;
		computeGradientFromNormalScalar(normalMap, gradientOut);
	}

	void computeGradientFromNormalToPQ(
		const cv::Mat& normalMap,
		cv::Mat& pOut,
		cv::Mat& qOut,
		cv::Mat& gradientOut)
	{
		ZoneScopedN("computeGradientFromNormalToPQ");
		const cv::Size size = normalMap.size();
		pOut.create(size, CV_32F);
		qOut.create(size, CV_32F);
		ensureMatBuffer(gradientOut, size, CV_32FC2);

		const int cols = size.width;
		cv::parallel_for_(cv::Range(0, size.height), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
				float* pRow = pOut.ptr<float>(y);
				float* qRow = qOut.ptr<float>(y);
				cv::Vec2f* gradRow = gradientOut.ptr<cv::Vec2f>(y);
				for (int x = 0; x < cols; ++x)
				{
					const cv::Vec3f& n = normalRow[x];
					writeGradientFromNormal(n[0], n[1], n[2], gradRow[x]);
					pRow[x] = gradRow[x][0];
					qRow[x] = gradRow[x][1];
				}
			}
		});
	}

	inline int borderReflect101(int p, int limit)
	{
		if (limit <= 1)
			return 0;
		while (p < 0 || p >= limit)
		{
			if (p < 0)
				p = -p;
			else
				p = limit - 1 - (p - limit);
		}
		return p;
	}

	inline float normalChannelAt(const cv::Mat& normalMap, int x, int y, int channel)
	{
		x = borderReflect101(x, normalMap.cols);
		y = borderReflect101(y, normalMap.rows);
		return normalMap.ptr<cv::Vec3f>(y)[x][channel];
	}

	inline float normalSobelDxAt(const cv::Mat& normalMap, int x, int y)
	{
		return normalChannelAt(normalMap, x + 1, y - 1, 0) - normalChannelAt(normalMap, x - 1, y - 1, 0)
			+ 2.f * (normalChannelAt(normalMap, x + 1, y, 0) - normalChannelAt(normalMap, x - 1, y, 0))
			+ normalChannelAt(normalMap, x + 1, y + 1, 0) - normalChannelAt(normalMap, x - 1, y + 1, 0);
	}

	inline float normalSobelDyAt(const cv::Mat& normalMap, int x, int y)
	{
		return normalChannelAt(normalMap, x - 1, y + 1, 1) - normalChannelAt(normalMap, x - 1, y - 1, 1)
			+ 2.f * (normalChannelAt(normalMap, x, y + 1, 1) - normalChannelAt(normalMap, x, y - 1, 1))
			+ normalChannelAt(normalMap, x + 1, y + 1, 1) - normalChannelAt(normalMap, x + 1, y - 1, 1);
	}

	inline float normalCurvatureAt(const cv::Mat& normalMap, int x, int y)
	{
		return normalSobelDxAt(normalMap, x, y) + normalSobelDyAt(normalMap, x, y);
	}

	void computeCurvatureFromNormalScalar(const cv::Mat& normalMap, cv::Mat& curvatureOut)
	{
		const int cols = normalMap.cols;
		const int rows = normalMap.rows;
		cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				float* curvRow = curvatureOut.ptr<float>(y);
				for (int x = 0; x < cols; ++x)
					curvRow[x] = normalCurvatureAt(normalMap, x, y);
			}
		});
	}

	void computeCurvatureFromNormal(const cv::Mat& normalMap, cv::Mat& curvatureOut)
	{
		ZoneScopedN("computeCurvatureFromNormal");
		ensureMatBuffer(curvatureOut, normalMap.size(), CV_32FC1);
		computeCurvatureFromNormalScalar(normalMap, curvatureOut);
	}

	void computeGradientAndCurvatureFromNormal(
		const cv::Mat& normalMap,
		cv::Mat& gradientOut,
		cv::Mat& curvatureOut)
	{
		ZoneScopedN("computeGradientAndCurvatureFromNormal");
		ensureMatBuffer(gradientOut, normalMap.size(), CV_32FC2);
		ensureMatBuffer(curvatureOut, normalMap.size(), CV_32FC1);

		const int cols = normalMap.cols;
		const int rows = normalMap.rows;
		cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
				cv::Vec2f* gradRow = gradientOut.ptr<cv::Vec2f>(y);
				float* curvRow = curvatureOut.ptr<float>(y);
				for (int x = 0; x < cols; ++x)
				{
					const cv::Vec3f& n = normalRow[x];
					writeGradientFromNormal(n[0], n[1], n[2], gradRow[x]);
					curvRow[x] = normalCurvatureAt(normalMap, x, y);
				}
			}
		});
	}

	struct PoissonHeightWorkspace
	{
		cv::Size size;
		cv::Mat p;
		cv::Mat q;
		cv::Mat Z;
		cv::Mat C;
		cv::Mat wx;
		cv::Mat wy;
		cv::Mat invDenom;
		cv::Mat wxPwyQ;

		void prepare(const cv::Size& newSize)
		{
			if (newSize == size && !p.empty())
				return;

			size = newSize;
			const int rows = size.height;
			const int cols = size.width;
			p.create(rows, cols, CV_32F);
			q.create(rows, cols, CV_32F);
			Z.create(rows, cols, CV_32F);
			C.create(rows, cols, CV_32F);
			wx.create(rows, cols, CV_32F);
			wy.create(rows, cols, CV_32F);
			invDenom.create(rows, cols, CV_32F);
			wxPwyQ.create(rows, cols, CV_32F);

			for (int v = 0; v < rows; ++v)
			{
				float* wxRow = wx.ptr<float>(v);
				float* wyRow = wy.ptr<float>(v);
				float* invRow = invDenom.ptr<float>(v);
				for (int u = 0; u < cols; ++u)
				{
					if (u == 0 && v == 0)
					{
						wxRow[u] = 0.f;
						wyRow[u] = 0.f;
						invRow[u] = 0.f;
						continue;
					}
					wxRow[u] = static_cast<float>(CV_PI * u / cols);
					wyRow[u] = static_cast<float>(CV_PI * v / rows);
					invRow[u] = 1.f / (wxRow[u] * wxRow[u] + wyRow[u] * wyRow[u]);
				}
			}
		}
	};

	PoissonHeightWorkspace& poissonHeightWorkspace(const cv::Size& size)
	{
		thread_local PoissonHeightWorkspace ws;
		ws.prepare(size);
		return ws;
	}

	inline cv::Size integrationSizeFor(const cv::Size& fullSize, float scale)
	{
		scale = std::clamp(scale, 0.25f, 1.0f);
		if (scale >= 0.999f)
			return fullSize;
		return cv::Size(
			std::max(1, static_cast<int>(std::lround(fullSize.width * scale))),
			std::max(1, static_cast<int>(std::lround(fullSize.height * scale))));
	}

	void integrateHeightFromPQ(PoissonHeightWorkspace& ws, cv::Mat& heightOut)
	{
		ZoneScopedN("integrateHeightFromPQ");
		ensureMatBuffer(heightOut, ws.size, CV_32FC1);

		{
			ZoneScopedN("integrateHeightFromPQ_dct");
			cv::parallel_for_(cv::Range(0, 2), [&](const cv::Range& range) {
				for (int i = range.start; i < range.end; ++i)
				{
					if (i == 0)
						cv::dct(ws.p, ws.p);
					else
						cv::dct(ws.q, ws.q);
				}
			});
		}

		{
			ZoneScopedN("integrateHeightFromPQ_freq");
			cv::multiply(ws.wx, ws.p, ws.wxPwyQ);
			cv::multiply(ws.wy, ws.q, ws.C);
			cv::add(ws.wxPwyQ, ws.C, ws.wxPwyQ);
			cv::multiply(ws.wxPwyQ, ws.invDenom, ws.Z, -1.0);
			ws.Z.at<float>(0, 0) = 0.f;
		}

		{
			ZoneScopedN("integrateHeightFromPQ_idct");
			cv::idct(ws.Z, heightOut);
		}
	}

	// 从高度图计算曲率
	void computeCurvatureFromHeight(const cv::Mat& heightMap, cv::Mat& curvatureOut)
	{
		ZoneScopedN("computeCurvatureFromHeight");
		ensureMatBuffer(curvatureOut, heightMap.size(), CV_32FC1);
		cv::Laplacian(heightMap, curvatureOut, CV_32F, 3);
	}

	void integrateHeightFromNormal(
		const cv::Mat& normalMap,
		cv::Mat& heightOut,
		cv::Mat* curvatureOut = nullptr)
	{
		ZoneScopedN("integrateHeightFromNormal");
		PoissonHeightWorkspace& ws = poissonHeightWorkspace(normalMap.size());
		computeGradientPQFromNormalScalar(normalMap, ws.p, ws.q);
		integrateHeightFromPQ(ws, heightOut);
		if (curvatureOut)
			computeCurvatureFromHeight(heightOut, *curvatureOut);
	}

	// 加权平均：标量单遍扫描
	void weightedAverageSinglePassScalar(
		const std::vector<cv::Mat>& imgs,
		const std::vector<float>& weights,
		cv::Mat& resultOut)
	{
		const size_t n = imgs.size();
		CV_Assert(n == weights.size() && n > 0);

		const cv::Size size = imgs[0].size();
		resultOut.create(size, imgs[0].type());

		const int cols = size.width;
		const int rows = size.height;
		const int imgType = imgs[0].type();

		// 行并行扫描，列并行累加
		cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				if (imgType == CV_8UC1)
				{
					uchar* dstRow = resultOut.ptr<uchar>(y);
					for (int x = 0; x < cols; ++x)
					{
						float sum = 0.f;
						for (size_t i = 0; i < n; ++i)
							sum += weights[i] * static_cast<float>(imgs[i].ptr<uchar>(y)[x]);
						dstRow[x] = cv::saturate_cast<uchar>(sum);
					}
				}
				else
				{
					float* dstRow = resultOut.ptr<float>(y);
					for (int x = 0; x < cols; ++x)
					{
						float sum = 0.f;
						for (size_t i = 0; i < n; ++i)
							sum += weights[i] * imgs[i].ptr<float>(y)[x];
						dstRow[x] = sum;
					}
				}
			}
		});
	}

	// 加权平均：优先 AVX2，不可用时回退标量计算
	void weightedAverageSinglePass(
		const std::vector<cv::Mat>& imgs,
		const std::vector<float>& weights,
		cv::Mat& resultOut)
	{
		if (weightedAverageTryAvx2(imgs, weights, resultOut))
			return;
		weightedAverageSinglePassScalar(imgs, weights, resultOut);
	}

	// 渲染合成图
	void renderSyntheticImageScalar(
		const cv::Mat& normalMap,
		const cv::Mat& albedoMap,
		const cv::Vec3f& lightDir,
		cv::Mat& resultOut)
	{
		ZoneScopedN("renderSyntheticImageScalar");
		const float lx = lightDir[0];
		const float ly = lightDir[1];
		const float lz = lightDir[2];
		const int cols = normalMap.cols;
		const int outputType = resultOut.type();

		cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
			for (int y = range.start; y < range.end; ++y)
			{
				const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
				const float* albedoRow = albedoMap.ptr<float>(y);
				if (outputType == CV_8UC1)
				{
					uchar* dstRow = resultOut.ptr<uchar>(y);
					for (int x = 0; x < cols; ++x)
					{
						const cv::Vec3f& n = normalRow[x];
						const float shade = std::max(0.f, n[0] * lx + n[1] * ly + n[2] * lz);
						dstRow[x] = cv::saturate_cast<uchar>(albedoRow[x] * shade);
					}
				}
				else
				{
					float* dstRow = resultOut.ptr<float>(y);
					for (int x = 0; x < cols; ++x)
					{
						const cv::Vec3f& n = normalRow[x];
						const float shade = std::max(0.f, n[0] * lx + n[1] * ly + n[2] * lz);
						dstRow[x] = albedoRow[x] * shade;
					}
				}
			}
		});
	}

	void renderSyntheticImage(
		const cv::Mat& normalMap,
		const cv::Mat& albedoMap,
		const cv::Vec3f& lightDir,
		int outputType,
		cv::Mat& resultOut)
	{
		ZoneScopedN("renderSyntheticImage");
		resultOut.create(normalMap.size(), outputType);
		if (outputType == CV_8UC1 && renderSyntheticTryAvx2(normalMap, albedoMap, lightDir, resultOut))
			return;
		renderSyntheticImageScalar(normalMap, albedoMap, lightDir, resultOut);
	}
}

ImageBlend::ImageBlend()
	: imagesSet(false)
	, configSet(false)
	, mode(ImageBlendMode::MaxValue)
	, config(MaxValueConfig{})
	, psPrecomputeReady_(false)
{
}

ImageBlend::~ImageBlend()
{
}

void ImageBlend::clearPhotometricStereoPrecompute()
{
	psInverseM_.clear();
	psScaledLightsFlat_.clear();
	psThresholdLutK4_.clear();
	psThresholdLutK6_.clear();
	psPrecomputeReady_ = false;
}

void ImageBlend::updatePhotometricStereoPrecompute(const std::vector<LightSource>& lights)
{
	ZoneScopedN("updatePhotometricStereoPrecompute");
	const std::vector<ScaledLight> scaled = buildScaledLights(lights);
	psScaledLightsFlat_.resize(scaled.size() * 3);
	for (size_t i = 0; i < scaled.size(); ++i)
	{
		psScaledLightsFlat_[i * 3 + 0] = scaled[i].sx;
		psScaledLightsFlat_[i * 3 + 1] = scaled[i].sy;
		psScaledLightsFlat_[i * 3 + 2] = scaled[i].sz;
	}

	if (!computeLightPseudoInverseCoeffs(scaled, psInverseM_))
	{
		clearPhotometricStereoPrecompute();
		return;
	}

	if (lights.size() == 4)
		buildThresholdLutK4(psScaledLightsFlat_, psThresholdLutK4_);
	else
		psThresholdLutK4_.clear();

	if (lights.size() == 6)
		buildThresholdLutK6(psScaledLightsFlat_, psThresholdLutK6_);
	else
		psThresholdLutK6_.clear();

	psPrecomputeReady_ = true;
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
// - 必须为 CV_8UC1 或 CV_32FC1，统一类型
// - 光度立体合成最少4张图片
ImageBlendError ImageBlend::validateImages(const std::vector<cv::Mat>& images, size_t minCount)
{
	if (images.size() < minCount) return ImageBlendError::ImageNumberError;

	for (const auto& img : images)
	{
		if (img.empty()) return ImageBlendError::NullImg;
	}

	const cv::Size size = images[0].size();
	if (images[0].channels() != 1) return ImageBlendError::NotGrayscale;
	const int type = images[0].type();
	if (!isSupportedInputType(type)) return ImageBlendError::ImageTypeDifferent;
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
	weightedAverageWeights_.clear();
	clearPhotometricStereoPrecompute();
	releaseOutputs();
	return ImageBlendError::OK;
}

// 验证配置
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
	if (blendMode == ImageBlendMode::WeightedAverage)
	{
		const auto& cfg = std::get<WeightedAverageConfig>(config);
		double weightSum = 0.0;
		for (double w : cfg.weights) weightSum += w;
		weightedAverageWeights_.resize(cfg.weights.size());
		// 归一化权重
		for (size_t i = 0; i < cfg.weights.size(); ++i)
			weightedAverageWeights_[i] = static_cast<float>(cfg.weights[i] / weightSum);
		clearPhotometricStereoPrecompute();
	}
	else if (blendMode == ImageBlendMode::PhotometricStereo)
	{
		weightedAverageWeights_.clear();
		const auto& cfg = std::get<PhotometricStereoConfig>(config);
		// 预计算用于灰度阈值过滤的查找表
		updatePhotometricStereoPrecompute(cfg.lightSystem.lights);
	}
	else
	{
		weightedAverageWeights_.clear();
		clearPhotometricStereoPrecompute();
	}
	configSet = true;
	releaseOutputs();
	return ImageBlendError::OK;
}

// 光度立体法：分配需可能需要的图（create 复用缓冲，不 zero-fill）
void ImageBlend::initPhotometricStereoMaps(const cv::Size& size, const PhotometricStereoOutputFlags& flags)
{
	ZoneScopedN("initPhotometricStereoMaps");
	ensureMatBuffer(normalMap, size, CV_32FC3);
	ensureMatBuffer(albedoMap, size, CV_32FC1);

	if (flags.computeGradientMap)
		ensureMatBuffer(gradientMap, size, CV_32FC2);
	else
		gradientMap.release();

	if (flags.computeHeightMap)
		ensureMatBuffer(heightMap, size, CV_32FC1);
	else
		heightMap.release();

	if (flags.computeCurvatureMap)
		ensureMatBuffer(curvatureMap, size, CV_32FC1);
	else
		curvatureMap.release();
}

// 最大值融合
void ImageBlend::executeMaxValue()
{
	ZoneScopedN("executeMaxValue");
	result = imgs[0].clone();
	// cv:max底层优化更快
	for (size_t i = 1; i < imgs.size(); ++i)
		cv::max(result, imgs[i], result);
}

// 加权平均融合
void ImageBlend::executeWeightedAverage()
{
	ZoneScopedN("executeWeightedAverage");
	weightedAverageSinglePass(imgs, weightedAverageWeights_, result);
}

// 光度立体法融合
void ImageBlend::executePhotometricStereo(const PhotometricStereoConfig& cfg)
{
	ZoneScopedN("executePhotometricStereo");
	const auto& lights = cfg.lightSystem.lights;
	const PhotometricStereoOutputFlags& flags = cfg.outputs;

	initPhotometricStereoMaps(imgs[0].size(), flags);

	// 预计算
	if (!psPrecomputeReady_)
		updatePhotometricStereoPrecompute(lights);

	solvePhotometricStereoMaps(
		imgs,
		psInverseM_,
		psScaledLightsFlat_,
		psThresholdLutK4_,
		psThresholdLutK6_,
		flags.intensityThreshold,
		normalMap,
		albedoMap);

	if (flags.computeGradientMap && flags.computeCurvatureMap && !flags.computeHeightMap)
	{
		// 同时计算梯度图和Sobel曲率图，不计算高度
		computeGradientAndCurvatureFromNormal(normalMap, gradientMap, curvatureMap);
	}
	// 计算高度
	else if (flags.computeHeightMap)
	{
		ZoneScopedN("integrateHeightPipeline");
		const cv::Size fullSize = normalMap.size();
		const cv::Size integrateSize = integrationSizeFor(fullSize, flags.heightIntegrationScale);
		const bool integrateReduced = integrateSize != fullSize;
		PoissonHeightWorkspace& ws = poissonHeightWorkspace(integrateSize);

		cv::Mat normalForIntegration = normalMap;
		cv::Mat normalReduced;
		if (integrateReduced)
		{
			normalReduced.create(integrateSize, CV_32FC3);
			cv::resize(normalMap, normalReduced, integrateSize, 0, 0, cv::INTER_AREA);
			normalForIntegration = normalReduced;
		}

		if (flags.computeGradientMap)
		{
			if (integrateReduced)
				computeGradientFromNormal(normalMap, gradientMap);
			else
				computeGradientFromNormalToPQ(normalMap, ws.p, ws.q, gradientMap);
		}

		if (integrateReduced || !flags.computeGradientMap)
			computeGradientPQFromNormalScalar(normalForIntegration, ws.p, ws.q);

		cv::Mat heightIntegrated;
		cv::Mat& heightIntegrateOut = integrateReduced ? heightIntegrated : heightMap;
		if (integrateReduced)
			heightIntegrated.create(integrateSize, CV_32FC1);

		integrateHeightFromPQ(ws, heightIntegrateOut);

		if (integrateReduced)
			cv::resize(heightIntegrated, heightMap, fullSize, 0, 0, cv::INTER_CUBIC);

		// Laplacian(~9ms) 替代第二次 IDCT(~80ms)；与 ∇²height 一致
		if (flags.computeCurvatureMap)
			computeCurvatureFromHeight(heightMap, curvatureMap);
	}
	else
	{
		// 只计算梯度图
		if (flags.computeGradientMap)
			computeGradientFromNormal(normalMap, gradientMap);
		
		// 计算曲率图
		if (flags.computeCurvatureMap)
			computeCurvatureFromNormal(normalMap, curvatureMap);
	}

	renderSyntheticImage(
		normalMap,
		albedoMap,
		flags.syntheticLightDir,
		imgs[0].type(),
		result);
}

// 漫反射平板标定
// 假设平板法线为 (0,0,1)，每张图对应一次打光，所有光源共用同一俯仰角 pitchDeg：
// 1. 亮区加权质心 -> 偏航角 yaw = atan2(dv, du)
// 2. 手动俯仰角 pitchDeg 与 yaw 合成 dir = (sin(p)cos(y), sin(p)sin(y), cos(p))
// 3. 亮区平均灰度 / lz -> 相对光强，最后归一化
ImageBlendError ImageBlend::calibrateFlatPanel(
	const std::vector<cv::Mat>& images,
	float pitchDeg,
	std::vector<LightSource>& outLights)
{
	ZoneScoped;
	if (pitchDeg < 0.f || pitchDeg >= 90.f)
		return ImageBlendError::ParameterError;

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
		const cv::Vec3f dir = flatCentroidPitchToLightDir(region.centroid, size, pitchDeg, dirOk);
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

// 漫反射球标定
// 1. 检测球体轮廓（亮区最大连通域 + 最小外接圆）
// 2. 球内灰度加权质心 -> 亮区中心
// 3. Lambert 模型：亮区质心处球面外法线即光源方向
// 4. 球内峰值灰度作为相对光强，最后归一化
ImageBlendError ImageBlend::calibrateSphere(const std::vector<cv::Mat>& images, std::vector<LightSource>& outLights)
{
	ZoneScoped;

	const size_t n = images.size();
	outLights.clear();
	outLights.reserve(n);

	std::vector<float> rawIntensities;
	rawIntensities.reserve(n);

	for (size_t i = 0; i < n; ++i)
	{
		const DiffuseSphereRegion region = extractDiffuseSphereRegion(images[i]);
		if (!region.valid)
			return ImageBlendError::ParameterError;

		bool dirOk = false;
		const cv::Vec3f dir = diffuseSphereToLightDir(region, dirOk);
		if (!dirOk || dir[2] < kNormalZEpsilon)
			return ImageBlendError::ParameterError;

		LightSource light;
		light.geo.dir = dir;
		light.calib.attenuation = 1.0f;
		light.calib.intensity = region.peakIntensity;
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

ImageBlendError ImageBlend::executeFlatCalibration(
	const std::vector<cv::Mat>& images,
	float pitchDeg,
	std::vector<LightSource>& outLights)
{
	ImageBlendError err = validateImages(images, 4);
	if (err != ImageBlendError::OK) return err;

	return calibrateFlatPanel(images, pitchDeg, outLights);
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
		executeWeightedAverage();
		break;
	case ImageBlendMode::PhotometricStereo:
		if (imgs.size() < 4) return ImageBlendError::ImageNumberError;
		executePhotometricStereo(std::get<PhotometricStereoConfig>(config));
		break;
	}

	return ImageBlendError::OK;
}

ImageBlendOutputMap ImageBlend::getResult()
{
	ImageBlendOutputMap outputs;
	if (!result.empty())
		outputs.emplace(ImageBlendOutputName::Result, result);
	if (!normalMap.empty())
		outputs.emplace(ImageBlendOutputName::NormalMap, normalMap);
	if (!heightMap.empty())
		outputs.emplace(ImageBlendOutputName::HeightMap, heightMap);
	if (!gradientMap.empty())
		outputs.emplace(ImageBlendOutputName::GradientMap, gradientMap);
	if (!albedoMap.empty())
		outputs.emplace(ImageBlendOutputName::AlbedoMap, albedoMap);
	if (!curvatureMap.empty())
		outputs.emplace(ImageBlendOutputName::CurvatureMap, curvatureMap);
	return outputs;
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
