#pragma once
#include <opencv2/core.hpp>
#include <vector>

// 光源方向
struct LightGeometry
{
	cv::Vec3f dir{ 0.f, 0.f, 1.f };
};

// 光源校准
struct LightCalibration
{
	float intensity = 1.0f;
	float attenuation = 1.0f;
};

// 光源
struct LightSource
{
	LightGeometry geo;
	LightCalibration calib;
};

// 投影模型
enum class ProjectionModel
{
	Orthographic,
	Perspective,
};

// 相机内参
struct CameraIntrinsics
{
	float fx = 1.0f;
	float fy = 1.0f;
	float cx = 0.0f;
	float cy = 0.0f;

	cv::Matx33f cameraMatrix() const
	{
		return cv::Matx33f(
			fx, 0.f, cx,
			0.f, fy, cy,
			0.f, 0.f, 1.f);
	}
};

// 相机外参
struct CameraExtrinsics
{
	cv::Matx33f R = cv::Matx33f::eye();
	cv::Vec3f t{ 0.f, 0.f, 0.f };
};

// 相机，理想情况下不使用相机参数
struct Camera
{
	CameraIntrinsics intrinsics;
	CameraExtrinsics extrinsics;
	ProjectionModel projection = ProjectionModel::Orthographic;
};

// 光源系统
struct LightSystem
{
	std::vector<LightSource> lights;
	Camera camera;
};

// 将光源方向转为 OpenCV 光度立体法使用的 N×3 CV_32F 矩阵
inline cv::Mat lightDirectionsMat(const std::vector<LightSource>& lights)
{
	const int n = static_cast<int>(lights.size());
	cv::Mat dirs(n, 3, CV_32F);
	for (int i = 0; i < n; ++i)
	{
		const cv::Vec3f& dir = lights[i].geo.dir;
		dirs.at<float>(i, 0) = dir[0];
		dirs.at<float>(i, 1) = dir[1];
		dirs.at<float>(i, 2) = dir[2];
	}
	return dirs;
}
