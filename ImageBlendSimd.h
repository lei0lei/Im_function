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

// 光度立体：threshold=0 快路径（k=4 或 k=6，CV_8UC1 或 CV_32FC1，连续内存）
bool solvePhotometricStereoTryAvx2(
	const std::vector<cv::Mat>& imgs,
	const std::vector<float>& inverseM,
	cv::Mat& normalOut,
	cv::Mat& albedoOut);

// 光度立体：合成图渲染（CV_32FC3 法线 + CV_32FC1 albedo，连续内存）
bool renderSyntheticTryAvx2(
	const cv::Mat& normalMap,
	const cv::Mat& albedoMap,
	const cv::Vec3f& lightDir,
	cv::Mat& resultOut);

// 法线图 → 梯度图（CV_32FC3 连续法线，CV_32FC2 连续输出）
bool computeGradientFromNormalTryAvx2(
	const cv::Mat& normalMap,
	cv::Mat& gradientOut);
