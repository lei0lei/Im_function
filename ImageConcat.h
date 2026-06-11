#pragma once
#include <opencv2/opencv.hpp>

enum class ImageConcatError
{
	OK,
	NullImg,
	ParameterError,
	ImageNumberError,
	ImageTypeDifferent,
};


class ImageConcat
{
public:
	ImageConcat();
	~ImageConcat();
	ImageConcatError setInput(std::vector<cv::Mat>& images, int rows, int cols);
	ImageConcatError execute();
	cv::Mat getResult();
private:
	int maxWidth;
	int maxHeight;
	int type;
	std::vector<cv::Mat> imgs;
	int rows;
	int cols;
	cv::Mat result;
};

