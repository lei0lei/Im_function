#pragma once

#include <opencv2/core.hpp>
#include <vector>

// ImageBlend SIMD 入口：各融合路径在支持 AVX2 且满足前置条件时尝试 SIMD，
// 成功返回 true 并已写入输出；否则返回 false，由调用方走标量实现。

// 加权平均融合
bool weightedAverageTryAvx2(
	const std::vector<cv::Mat>& imgs,
	const std::vector<float>& weights,
	cv::Mat& resultOut);
