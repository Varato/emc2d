#pragma once
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
namespace Contrast
{
	inline void Log_Contrast(cv::Mat &img_float)
	{
		int row = img_float.rows;
		int col = img_float.cols;
		float const_coeff = 1e6f;
		for (auto i = 0; i < row; i++)
		{
			for (auto j = 0; j < col; j++)
			{
				img_float.at<float>(i, j) = log(img_float.at<float>(i, j)* const_coeff + 1.0f);;
			}
		}
	}
	inline void Linear_Contrast(cv::Mat &img_float, bool background)
	{
		double min_val, max_val;
		cv::minMaxLoc(img_float, &min_val, &max_val);
		if (background)
		{
			img_float = (img_float - min_val) / (max_val - min_val);
		}
		else
		{
			img_float = img_float /max_val;
		}
	}
}