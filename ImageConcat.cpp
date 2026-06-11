#include "ImageConcat.h"

ImageConcat::ImageConcat()
{
}

ImageConcat::~ImageConcat()
{
}

ImageConcatError ImageConcat::setInput(std::vector<cv::Mat>& images, int rows, int cols)
{
	if (rows < 1)return ImageConcatError::ParameterError;
	if (cols < 1)return ImageConcatError::ParameterError;
	if (images.size() != rows * cols)return ImageConcatError::ImageNumberError;
	
	maxWidth = 0;
	maxHeight = 0;
	for (const auto& img : images)
	{
		if (img.empty()) return ImageConcatError::NullImg;
		maxWidth = std::max(maxWidth, img.cols);
		maxHeight = std::max(maxHeight, img.rows);
	}

	type = images[0].type();
	for (const auto& img : images)
	{
		if (type != img.type())return ImageConcatError::ImageTypeDifferent;
	}
	imgs.clear();
	imgs = images;
	this->rows = rows;
	this->cols = cols;
	return ImageConcatError::OK;
}

ImageConcatError ImageConcat::execute()
{
	int totalWidth = cols * maxWidth;
	int totalHeight = rows * maxHeight;
	result = cv::Mat::zeros(totalHeight, totalWidth, type);
	int count = rows * cols;
	for (int i = 0; i < count; ++i)
	{
		const cv::Mat& img = imgs[i];

		int r = i / cols;  // ÐÐ
		int c = i % cols;  // ÁÐ

		int dstX = c * maxWidth;
		int dstY = r * maxHeight;

		int copyW = std::min(img.cols, maxWidth);
		int copyH = std::min(img.rows, maxHeight);
		img(cv::Rect(0, 0, copyW, copyH)).copyTo(result(cv::Rect(dstX, dstY, copyW, copyH)));
	}
	return ImageConcatError::OK;
}

cv::Mat ImageConcat::getResult()
{
	if (result.empty()) return cv::Mat();
	return result;
}