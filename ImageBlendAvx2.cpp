#include "ImageBlendSimd.h"
#include "Profiling.h"
#include <immintrin.h>
#include <cmath>

namespace
{
	constexpr float kAlbedoEpsilon = 1e-4f;
	// 字节扩展，加载8字节灰度图像到256位浮点数
	inline __m256 loadU8ToPs8(const uchar* src)
	{
		const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
		const __m128i words = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
		const __m128i dwordsLo = _mm_unpacklo_epi16(words, _mm_setzero_si128());
		const __m128i dwordsHi = _mm_unpackhi_epi16(words, _mm_setzero_si128());
		__m256 values = _mm256_castps128_ps256(_mm_cvtepi32_ps(dwordsLo));
		return _mm256_insertf128_ps(values, _mm_cvtepi32_ps(dwordsHi), 1);
	}
	
	// 反向操作，存储256位浮点数到8字节灰度图像
	inline void storePs8ToU8(uchar* dst, __m256 values)
	{
		alignas(32) float buffer[8];
		_mm256_storeu_ps(buffer, values);
		for (int i = 0; i < 8; ++i)
			dst[i] = cv::saturate_cast<uchar>(buffer[i]);
	}

	inline void writeNormalAlbedoFromG(float gx, float gy, float gz, cv::Vec3f& normalOut, float& albedoOut)
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

	inline void writeNormalAlbedoFromG8(
		__m256 gx,
		__m256 gy,
		__m256 gz,
		cv::Vec3f* normalRow,
		float* albedoRow,
		int x)
	{
		alignas(32) float gxArr[8];
		alignas(32) float gyArr[8];
		alignas(32) float gzArr[8];
		_mm256_storeu_ps(gxArr, gx);
		_mm256_storeu_ps(gyArr, gy);
		_mm256_storeu_ps(gzArr, gz);
		for (int i = 0; i < 8; ++i)
			writeNormalAlbedoFromG(gxArr[i], gyArr[i], gzArr[i], normalRow[x + i], albedoRow[x + i]);
	}

	inline __m256 loadObs8(const cv::Mat& img, int imgType, const uchar* u8Row, const float* f32Row, int x)
	{
		(void)img;
		if (imgType == CV_8UC1)
			return loadU8ToPs8(u8Row + x);
		return _mm256_loadu_ps(f32Row + x);
	}

	inline __m256 computeG8N(const __m256* obs, const float* mRow, int k)
	{
		__m256 sum = _mm256_mul_ps(obs[0], _mm256_set1_ps(mRow[0]));
		for (int i = 1; i < k; ++i)
			sum = _mm256_fmadd_ps(obs[i], _mm256_set1_ps(mRow[i]), sum);
		return sum;
	}

	bool canUseAvx2ForGray8Mats(const std::vector<cv::Mat>& imgs)
	{
		if (!cv::checkHardwareSupport(CV_CPU_AVX2))
			return false;
		if (imgs.empty() || imgs[0].type() != CV_8UC1)
			return false;

		const cv::Size size = imgs[0].size();
		for (const cv::Mat& img : imgs)
		{
			if (img.type() != CV_8UC1 || img.size() != size || !img.isContinuous())
				return false;
		}
		return true;
	}

	bool canUseAvx2ForPsSolve(
		const std::vector<cv::Mat>& imgs,
		const std::vector<float>& inverseM)
	{
		if (!cv::checkHardwareSupport(CV_CPU_AVX2))
			return false;
		const size_t k = imgs.size();
		if ((k != 4 && k != 6) || inverseM.size() != 3 * k)
			return false;

		const int type = imgs[0].type();
		if (type != CV_8UC1 && type != CV_32FC1)
			return false;

		const cv::Size size = imgs[0].size();
		for (const cv::Mat& img : imgs)
		{
			if (img.type() != type || img.size() != size || !img.isContinuous())
				return false;
		}
		return true;
	}

	bool canUseAvx2ForRender(const cv::Mat& normalMap, const cv::Mat& albedoMap)
	{
		if (!cv::checkHardwareSupport(CV_CPU_AVX2))
			return false;
		return normalMap.type() == CV_32FC3
			&& albedoMap.type() == CV_32FC1
			&& normalMap.size() == albedoMap.size()
			&& normalMap.isContinuous()
			&& albedoMap.isContinuous();
	}
}

bool weightedAverageTryAvx2(
	const std::vector<cv::Mat>& imgs,
	const std::vector<float>& weights,
	cv::Mat& resultOut)
{
	const size_t n = imgs.size();
	if (n != weights.size() || n == 0)
		return false;
	// 只能处理8UC1并且支持AVX2
	if (!canUseAvx2ForGray8Mats(imgs))
		return false;

	const cv::Size size = imgs[0].size();
	resultOut.create(size, CV_8UC1);

	const int cols = size.width;
	const int rows = size.height;
	// 按8字节对齐
	const int colsSimd = cols - (cols % 8);

	cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
		for (int y = range.start; y < range.end; ++y)
		{
			uchar* dstRow = resultOut.ptr<uchar>(y);

			int x = 0;
			for (; x < colsSimd; x += 8)
			{
				__m256 sum = _mm256_setzero_ps();
				for (size_t i = 0; i < n; ++i)
				{
					const uchar* srcRow = imgs[i].ptr<uchar>(y);
					const __m256 pixels = loadU8ToPs8(srcRow + x);
					const __m256 w = _mm256_set1_ps(weights[i]);
					sum = _mm256_fmadd_ps(pixels, w, sum);
				}
				storePs8ToU8(dstRow + x, sum);
			}

			for (; x < cols; ++x)
			{
				float sum = 0.f;
				for (size_t i = 0; i < n; ++i)
					sum += weights[i] * static_cast<float>(imgs[i].ptr<uchar>(y)[x]);
				dstRow[x] = cv::saturate_cast<uchar>(sum);
			}
		}
	});

	return true;
}

bool solvePhotometricStereoTryAvx2(
	const std::vector<cv::Mat>& imgs,
	const std::vector<float>& inverseM,
	cv::Mat& normalOut,
	cv::Mat& albedoOut)
{
	ZoneScopedN("solvePhotometricStereoTryAvx2");
	if (!canUseAvx2ForPsSolve(imgs, inverseM))
		return false;

	const int k = static_cast<int>(imgs.size());
	const int imgType = imgs[0].type();
	const int cols = imgs[0].cols;
	const int rows = imgs[0].rows;
	const int colsSimd = cols - (cols % 8);
	const float* m0 = inverseM.data();
	const float* m1 = m0 + k;
	const float* m2 = m0 + 2 * k;

	cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
		for (int y = range.start; y < range.end; ++y)
		{
			const uchar* u8Rows[6] = {};
			const float* f32Rows[6] = {};
			for (int i = 0; i < k; ++i)
			{
				u8Rows[i] = imgs[static_cast<size_t>(i)].ptr<uchar>(y);
				f32Rows[i] = imgs[static_cast<size_t>(i)].ptr<float>(y);
			}

			cv::Vec3f* normalRow = normalOut.ptr<cv::Vec3f>(y);
			float* albedoRow = albedoOut.ptr<float>(y);

			int x = 0;
			for (; x < colsSimd; x += 8)
			{
				__m256 obs[6];
				for (int i = 0; i < k; ++i)
					obs[i] = loadObs8(imgs[static_cast<size_t>(i)], imgType, u8Rows[i], f32Rows[i], x);
				const __m256 gx = computeG8N(obs, m0, k);
				const __m256 gy = computeG8N(obs, m1, k);
				const __m256 gz = computeG8N(obs, m2, k);
				writeNormalAlbedoFromG8(gx, gy, gz, normalRow, albedoRow, x);
			}

			for (; x < cols; ++x)
			{
				float gx = 0.f;
				float gy = 0.f;
				float gz = 0.f;
				for (int i = 0; i < k; ++i)
				{
					const float v = (imgType == CV_8UC1)
						? static_cast<float>(u8Rows[i][x])
						: f32Rows[i][x];
					gx += m0[i] * v;
					gy += m1[i] * v;
					gz += m2[i] * v;
				}
				writeNormalAlbedoFromG(gx, gy, gz, normalRow[x], albedoRow[x]);
			}
		}
	});

	return true;
}

bool renderSyntheticTryAvx2(
	const cv::Mat& normalMap,
	const cv::Mat& albedoMap,
	const cv::Vec3f& lightDir,
	cv::Mat& resultOut)
{
	ZoneScopedN("renderSyntheticTryAvx2");
	if (!canUseAvx2ForRender(normalMap, albedoMap))
		return false;

	resultOut.create(normalMap.size(), CV_8UC1);
	const float lx = lightDir[0];
	const float ly = lightDir[1];
	const float lz = lightDir[2];
	const __m256 lxV = _mm256_set1_ps(lx);
	const __m256 lyV = _mm256_set1_ps(ly);
	const __m256 lzV = _mm256_set1_ps(lz);
	const __m256 zero = _mm256_setzero_ps();

	const int cols = normalMap.cols;
	const int rows = normalMap.rows;
	const int colsSimd = cols - (cols % 8);

	cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
		for (int y = range.start; y < range.end; ++y)
		{
			const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
			const float* albedoRow = albedoMap.ptr<float>(y);
			uchar* dstRow = resultOut.ptr<uchar>(y);

			int x = 0;
			for (; x < colsSimd; x += 8)
			{
				alignas(32) float nx[8];
				alignas(32) float ny[8];
				alignas(32) float nz[8];
				for (int i = 0; i < 8; ++i)
				{
					nx[i] = normalRow[x + i][0];
					ny[i] = normalRow[x + i][1];
					nz[i] = normalRow[x + i][2];
				}

				const __m256 albedo = _mm256_loadu_ps(albedoRow + x);
				const __m256 shade = _mm256_max_ps(
					zero,
					_mm256_fmadd_ps(_mm256_loadu_ps(nx), lxV,
						_mm256_fmadd_ps(_mm256_loadu_ps(ny), lyV, _mm256_mul_ps(_mm256_loadu_ps(nz), lzV))));
				storePs8ToU8(dstRow + x, _mm256_mul_ps(albedo, shade));
			}

			for (; x < cols; ++x)
			{
				const cv::Vec3f& n = normalRow[x];
				const float shade = std::max(0.f, n[0] * lx + n[1] * ly + n[2] * lz);
				dstRow[x] = cv::saturate_cast<uchar>(albedoRow[x] * shade);
			}
		}
	});

	return true;
}

bool computeGradientFromNormalTryAvx2(
	const cv::Mat& normalMap,
	cv::Mat& gradientOut)
{
	if (!cv::checkHardwareSupport(CV_CPU_AVX2))
		return false;
	if (normalMap.type() != CV_32FC3)
		return false;
	if (gradientOut.type() != CV_32FC2 || gradientOut.size() != normalMap.size())
		gradientOut.create(normalMap.size(), CV_32FC2);
	if (!normalMap.isContinuous() || !gradientOut.isContinuous())
		return false;

	const int cols = normalMap.cols;
	const int rows = normalMap.rows;
	const int colsSimd = cols - (cols % 8);
	const __m256 eps = _mm256_set1_ps(1e-3f);
	const __m256 one = _mm256_set1_ps(1.f);
	const __m256 negOne = _mm256_set1_ps(-1.f);

	cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
		for (int y = range.start; y < range.end; ++y)
		{
			const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
			cv::Vec2f* gradRow = gradientOut.ptr<cv::Vec2f>(y);

			int x = 0;
			for (; x < colsSimd; x += 8)
			{
				alignas(32) float nx[8];
				alignas(32) float ny[8];
				alignas(32) float nz[8];
				for (int i = 0; i < 8; ++i)
				{
					nx[i] = normalRow[x + i][0];
					ny[i] = normalRow[x + i][1];
					nz[i] = normalRow[x + i][2];
				}

				const __m256 vnx = _mm256_loadu_ps(nx);
				const __m256 vny = _mm256_loadu_ps(ny);
				const __m256 vnz = _mm256_loadu_ps(nz);
				const __m256 absNz = _mm256_andnot_ps(_mm256_set1_ps(-0.f), vnz);
				const __m256 valid = _mm256_cmp_ps(absNz, eps, _CMP_GT_OQ);
				const __m256 invNz = _mm256_div_ps(one, vnz);
				const __m256 px = _mm256_mul_ps(_mm256_mul_ps(vnx, invNz), negOne);
				const __m256 py = _mm256_mul_ps(_mm256_mul_ps(vny, invNz), negOne);
				const __m256 outPx = _mm256_and_ps(px, valid);
				const __m256 outPy = _mm256_and_ps(py, valid);

				alignas(32) float pxArr[8];
				alignas(32) float pyArr[8];
				_mm256_storeu_ps(pxArr, outPx);
				_mm256_storeu_ps(pyArr, outPy);
				for (int i = 0; i < 8; ++i)
					gradRow[x + i] = cv::Vec2f(pxArr[i], pyArr[i]);
			}

			for (; x < cols; ++x)
			{
				const cv::Vec3f& n = normalRow[x];
				if (std::abs(n[2]) < 1e-3f)
					gradRow[x] = cv::Vec2f(0.f, 0.f);
				else
				{
					const float invNz = 1.f / n[2];
					gradRow[x] = cv::Vec2f(-n[0] * invNz, -n[1] * invNz);
				}
			}
		}
	});

	return true;
}
