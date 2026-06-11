#include "ImageBlendSimd.h"
#include <immintrin.h>

namespace
{
	inline __m256 loadU8ToPs8(const uchar* src)
	{
		const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
		const __m128i words = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
		const __m128i dwordsLo = _mm_unpacklo_epi16(words, _mm_setzero_si128());
		const __m128i dwordsHi = _mm_unpackhi_epi16(words, _mm_setzero_si128());
		__m256 values = _mm256_castps128_ps256(_mm_cvtepi32_ps(dwordsLo));
		return _mm256_insertf128_ps(values, _mm_cvtepi32_ps(dwordsHi), 1);
	}

	inline void storePs8ToU8(uchar* dst, __m256 values)
	{
		alignas(32) float buffer[8];
		_mm256_storeu_ps(buffer, values);
		for (int i = 0; i < 8; ++i)
			dst[i] = cv::saturate_cast<uchar>(buffer[i]);
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
}

bool weightedAverageTryAvx2(
	const std::vector<cv::Mat>& imgs,
	const std::vector<float>& weights,
	cv::Mat& resultOut)
{
	const size_t n = imgs.size();
	if (n != weights.size() || n == 0)
		return false;
	if (!canUseAvx2ForGray8Mats(imgs))
		return false;

	const cv::Size size = imgs[0].size();
	resultOut.create(size, CV_8UC1);

	const int cols = size.width;
	const int rows = size.height;
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
