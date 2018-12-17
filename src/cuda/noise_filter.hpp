#pragma once
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "host_config.h"
#include "fftlib.cuh"
namespace Filter
{
	inline void generate_fftAmp(std::vector<float> frames, int frame_num, int row_frame, int col_frame,
		std::vector<float> &averaged_amp)
	{
		int frame_size = row_frame*col_frame;
		averaged_amp.resize(frame_size);
		std::fill(averaged_amp.begin(), averaged_amp.end(), 0.0f);
		FFTLIB::FFT<float> fourtrans;
		fourtrans.create_plan_2d(row_frame, col_frame, 8);
		std::vector <std::complex<float>> complex_A(frame_size);
		for (auto iframe = 0; iframe < frame_num; iframe++)
		{
			for (auto i = 0; i < frame_size; i++)
			{
				complex_A[i] = std::complex<float>(frames[i + iframe*frame_size], 0.0f);
			}
			fourtrans.forward(complex_A);
			for (auto i = 0; i < frame_size; i++)
			{
				averaged_amp[i] += std::abs(complex_A[i]);
			}
		}
	}
	inline void generate_mask(cv::Mat img_fft, std::vector<float> &mask, int radius)
	{
		float band_value = 0.1;
		int erosion_type = 2;
		int erosion_size = radius;
		int row = img_fft.rows;
		int col = img_fft.cols;
		cv::Mat dilated(img_fft.size(), img_fft.type());
		cv::Mat filtered=cv::Mat::zeros(img_fft.size(), img_fft.type());
		cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2*erosion_size+1, 2*erosion_size+1));
		cv::dilate(img_fft, dilated, element);
		for (auto i=0; i<row; i++)
		{
			for (auto j=0; j< col; j++)
			{
				float temp = img_fft.at<float>(i, j);
				if (temp>band_value&& (temp==dilated.at<float>(i, j)))
				{
					filtered.at<float>(i, j) = 1.0;
				}
			}
		}
		int image_size = col*row;
		cv::dilate(filtered, dilated, element);
		mask.resize(image_size);
		cv::imshow("dilate", dilated);
		memcpy(mask.data(), dilated.data, sizeof(float)*image_size);
	}
	inline void apply_Fourier_Filter(std::vector<float> &model, std::vector<float> mask, int row, int col)
	{
		std::vector <std::complex<float>> complex_A(model.size());
		FFTLIB::FFT<float> fourtrans;
		fourtrans.create_plan_2d(row, col, 8);
		for (auto i = 0; i < model.size(); i++)
		{
			complex_A[i] = std::complex<float>(model[i], 0.0f);
		}
		fourtrans.forward(complex_A);
		for (auto i=0; i<model.size();i++)
		{
			complex_A[i] *= mask[i];
		}
		fourtrans.inverse(complex_A);
		for (auto i = 0; i < model.size(); i++)
		{
			model[i] = std::abs(complex_A[i]) / row / col;
		}
	}
	inline void applyBWLimit(std::vector<float> &matrix_A, int row, int col, int limit)
	{
		std::vector <std::complex<float>> complex_A(matrix_A.size());
		FFTLIB::FFT<float> fourtrans;
		fourtrans.create_plan_2d(row, col, 8);
		for (auto i = 0; i < matrix_A.size(); i++)
		{
			complex_A[i] = std::complex<float>(matrix_A[i], 0.0f);
		}
		fourtrans.forward(complex_A);
		std::vector<int> f_row(row);
		std::vector<int> f_col(col);
		for (auto i = 0; i < row; i++)
		{
			if (i < row / 2)
			{
				f_row[i] = i;
			}
			else
			{
				f_row[i] = i - row;
			}
		}
		for (auto i = 0; i < col; i++)
		{
			if (i < col / 2)
			{
				f_col[i] = i;
			}
			else
			{
				f_col[i] = i - col;
			}
		}
		for (auto i = 0; i < row; i++)
		{
			for (auto j = 0; j < col; j++)
			{
				float radiu = (float)sqrt(f_row[i] * f_row[i] + f_col[j] * f_col[j]);
				if (radiu > limit)
				{
					complex_A[i*col + j] = std::complex<float>(0.0f, 0.0f);
				}
			}
		}
		fourtrans.inverse(complex_A);
		for (auto i = 0; i < matrix_A.size(); i++)
		{
			matrix_A[i] = std::abs(complex_A[i]) / row / col;
		}
	}
}