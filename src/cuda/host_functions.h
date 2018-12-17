#pragma once
#include "mkl.h"
#include "stdio.h"
#include "stdlib.h"
#include <vector>
#include <random>
#include "fftlib.cuh"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#ifndef VOID_LIMIT
#define VOID_LIMIT 0.0000001
#endif // !VOID_LIMIT
/************************************************************************/
/* Function description:                                                */
/* In order to add random Poisson noise to generated one dimensional or */
/* Two dimensional image. The dose is used to control the SNR, and seed */
/* is used to control the generated random number.                      */
/************************************************************************/
void inline addPoissonNoise(std::vector<float> &clean_data, float dose, int seed)
{
	std::mt19937 random_linear_generator;
	std::mt19937 random_poisson_generator;
	random_linear_generator.seed(seed);
	std::uniform_int_distribution<int> rand_seed;
	std::vector<int> seeds;
	seeds.resize(clean_data.size());
	for (int i = 0; i < clean_data.size(); i++)
	{
		seeds[i] = rand_seed(random_linear_generator);
	}
	for (int i = 0; i < clean_data.size(); i++)
	{
		random_poisson_generator.seed(seeds[i]);
		std::poisson_distribution<int> distribution(clean_data[i] * dose);
		clean_data[i] = (float)distribution(random_poisson_generator) / dose;
	}
}
void inline addPoissonNoise(std::vector<float> &clean_data, float dose)
{
	std::mt19937 random_generator;
	for (auto i = 0; i < clean_data.size(); i++)
	{
		std::poisson_distribution<int> distribution(clean_data[i] * dose);
		int Poisson_val = distribution(random_generator);
		clean_data[i] = (float)Poisson_val / dose;
	}
}
void inline addPoissonNoise(cv::Mat &img, float dose)
{
	std::mt19937 random_generator;
	int row = img.rows;
	int col = img.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			std::poisson_distribution<int> distribution(img.at<float>(i, j)*dose);
			int Poisson_val = distribution(random_generator);
			img.at<float>(i, j) = (float)Poisson_val / dose;
		}
	}
}
/************************************************************************/
/* Function description:                                                */
/* Generate the index for sparse matrix  */
/************************************************************************/
void inline generate_sparse_index(std::vector<float> matrix, int row, int col,
	std::vector<float> &values, std::vector<long long> &row_index, std::vector<long long> &col_index)
{
	size_t sparse_num = 0;
	if (row*col != matrix.size())
	{
		std::cout << "dimension not match" << std::endl;
		return;
	}
	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			if (abs(matrix[i*col + j]) > VOID_LIMIT)
			{
				values[sparse_num] = matrix[i*col + j];
				row_index[sparse_num] = (long long)i;
				col_index[sparse_num] = (long long)j;
				sparse_num++;
			}
		}
	}
	values.resize(sparse_num);
	row_index.resize(sparse_num);
	col_index.resize(sparse_num);
	values.shrink_to_fit();
	row_index.shrink_to_fit();
	col_index.shrink_to_fit();
}
void inline generate_sparse_index(std::vector<float> matrix, int row, int col,
	std::vector<float> &values, std::vector<int> &row_index, std::vector<int> &col_index)
{
	size_t sparse_num = 0;
	if (row*col != matrix.size())
	{
		std::cout << "dimension not match" << std::endl;
		return;
	}
	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			if (abs(matrix[i*col + j]) > VOID_LIMIT)
			{
				values[sparse_num] = matrix[i*col + j];
				row_index[sparse_num] = (long long)i;
				col_index[sparse_num] = (long long)j;
				sparse_num++;
			}
		}
	}
	values.resize(sparse_num);
	row_index.resize(sparse_num);
	col_index.resize(sparse_num);
	values.shrink_to_fit();
	row_index.shrink_to_fit();
	col_index.shrink_to_fit();
}
void inline generate_sparse_index(std::vector<float> vector,
	std::vector<float> values, std::vector<int> index)
{
	values.clear();
	index.clear();
	for (auto i = 0; i < vector.size(); i++)
	{
		if (abs(vector[i]) > VOID_LIMIT)
		{
			values.push_back(vector[i]);
			index.push_back(i);
		}
	}
}
/************************************************************************/
/* Dense float type Matrix multiplication based on MKL library          */
/* The input dimension is for the final result of matrix C.             */
/* Both A, B, C are stored along the row direction, row priority.       */
/* The row priority is different from the operation in Fortran.         */
/* The row and column of A and B could be determined by the orientation */
/* flag.                                                                */
/* If width of A is h_C, then the A matrix should be transposed         */
/* Then orientation_A should be false.                                  */
/* If height of B is w_C, then the B matrix should be transposed        */
/* Then orientation_B should be false.                                  */
/************************************************************************/
void inline matrixMul(std::vector<float> matrix_A, bool orientation_A, std::vector<float> matrix_B, int orientation_B,
	std::vector<float> &matrix_C, int row_C, int col_C, int shared_Length)
{
	int max_threads = mkl_get_max_threads();
	mkl_set_num_threads(max_threads);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	CBLAS_TRANSPOSE transpose_A;
	CBLAS_TRANSPOSE transpose_B;
	long long id_A;
	long long id_B;
	if (orientation_A)
	{
		transpose_A = CblasNoTrans;
		id_A = shared_Length;
	}
	else
	{
		transpose_A = CblasTrans;
		id_A = row_C;
	}
	if (orientation_B)
	{
		transpose_B = CblasNoTrans;
		id_B = col_C;
	}
	else
	{
		transpose_B = CblasTrans;
		id_B = shared_Length;
	}
	//reset the value of matrix C. 
	for (auto i = 0; i < matrix_C.size(); i++)
	{
		matrix_C[i] = 0.0f;
	}
	cblas_sgemm(CblasRowMajor, transpose_A, transpose_B,
		row_C, col_C, shared_Length, alpha,
		matrix_A.data(), id_A,
		matrix_B.data(), id_B,
		beta, matrix_C.data(), col_C);
}
/************************************************************************/
/* By default, A is the sparse matrix.                                  */
/* By default, B and C are the dense matrix.                            */
/* Do remember, row_A and col_A belongs to the original storage conditi-*/
/* on of A, and col_B is the width of B and C, matrix B could not be    */
/* transposed, A is transposed or not have influence on the dimension   */
/* of B and C, if not transposed, C: row_A, col_B, B: col_A, col_B      */
/* if transposed, C: col_A, col_B, B: row_A, col_B                      */
/************************************************************************/

void inline matrixMulSparse(std::vector<float> values_A, std::vector<long long> row_index, std::vector<long long> col_index,
	std::vector<float> matrix_B, std::vector<float> &matrix_C, int row_C, int col_C, int shared_length)
{
	int max_threads = mkl_get_max_threads();
	mkl_set_num_threads(max_threads);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	char  trans_A;
	char  matdescra[6];
	long long sparse_length = values_A.size();
	//under determined 
	long long row_matrixC = row_C;
	long long col_matrixC = col_C;
	long long col_matrixA = shared_length;
	matdescra[0] = 'g';
	matdescra[3] = 'c';
	trans_A = 'n';
	for (auto i = 0; i < matrix_C.size(); i++)
	{
		matrix_C[i] = 0.0f;
	}
	//the dimension of B should be w_A * w_B when the matrix is not transposed.
	//the dimension of B should be h_A * w_B when the matrix is transposed.
	//the dimension of C should be h_A * w_B when the matrix is not transposed.
	//the dimension of C should be w_A * w_B when the matrix is transposed.
	mkl_scoomm(&trans_A, &row_matrixC, &col_matrixC, &col_matrixA, &alpha, matdescra,
		values_A.data(), row_index.data(), col_index.data(), &sparse_length, matrix_B.data(), &col_matrixC, &beta,
		matrix_C.data(), &col_matrixC);
}
void inline matrixTrans(std::vector<float> &matrix_A, int row_A, int col_A)
{
	float alpha = 1.0;
	char order = 'r';
	char trans = 't';
	int max_threads = mkl_get_max_threads();
	mkl_set_num_threads(max_threads);
	mkl_simatcopy(order, trans, row_A, col_A, alpha, matrix_A.data(), col_A, row_A);
	std::cout << " transposed" << std::endl;
}
void inline matrixTrans(std::vector<float> matrix_A, std::vector<float> &matrix_B, int row_A, int col_A)
{
	for (int i = 0; i < row_A; i++)
	{
		for (int j = 0; j < col_A; j++)
		{
			matrix_B[j*row_A + i] = matrix_A[i*col_A + j];
		}
	}
}
void inline RadialProfile(std::vector<float> &matrix_A, int row, int col, std::vector<float> &profile)
{
	std::vector <std::complex<float>> complex_A(matrix_A.size());
	FFTLIB::FFT<float> fourtrans;
	float const_coeff=1.0e6;
	int profile_len = (row<col)? row: col;
	profile_len /= 2;
	std::vector<int> profile_num(profile_len,0);
	profile.resize(profile_len);
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
	for (auto i = 0; i < profile_len; i++)
	{
		profile[i] = 0.0f;
	}
	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			float radiu = (float)sqrt(f_row[i] * f_row[i] + f_col[j] * f_col[j]);
			if (radiu>profile_len)
			{
				continue;
			}
			int   temp = (int)round(radiu);
			profile[temp] += log(std::norm(complex_A[i*col + j] )* const_coeff+1.0f);
			profile_num[temp]++;
		}
	}
	for (auto i = 0; i < profile_len; i++)
	{
		if (profile_num[i]>0)
		{
			profile[i] /= profile_num[i];
			profile[i] /= profile[0];
		}
	}
}
void inline AmplitudeAverageDFT(std::vector<float> frames, int row, int col, int frame_num, std::vector<float> &averaged)
{
	int frame_size = row*col;
	FFTLIB::FFT<float> fourierTran;
	cv::Mat img_sum = cv::Mat::zeros(row, col, CV_32FC1);
	fourierTran.create_plan_2d(row, col, 8);
	averaged.resize(frame_size);
	for (auto i = 0; i < frame_size;i++)
	{
		averaged[i] = 0.0f;
	}
	for (auto iframe = 0; iframe < frame_num; iframe++)
	{
		size_t start_index=iframe*frame_size;
		std::vector<std::complex<float>> imgData(frame_size);
		for (auto i = 0; i < frame_size; i++)
		{
			imgData[i] = std::complex<float>(frames[i+start_index], 0.0f);
		}
		fourierTran.forward(imgData);
		FFTLIB::fftshift2(imgData, row, col);
		for (auto i = 0; i < frame_size; i++)
		{
			averaged[i] +=std::abs(imgData[i]);
		}
	}
	for (auto i = 0; i < frame_size; i++)
	{
		averaged[i] /= frame_num;
	}
}
void inline DFTLogrithm(std::vector<float> &frames, int row, int col)
{
	float coeff = 1e6f;
	int frame_size = row*col;
	for (auto i = 0; i < frame_size; i++)
	{
		frames[i] = std::log(frames[i]*coeff+1.0f);
	}
}
void inline PickMaximums(cv::Mat img_float, std::vector<cv::Point> &maximums)
{
	float band_value = 0.1f;
	int erosion_type = 2;
	int erosion_size = 3;
	cv::Mat dialated(img_float.size(), img_float.type());
	cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));
	cv::dilate(img_float, dialated, element);
	cv::imshow("dilated", dialated);
	cv::imwrite("dilated.bmp", dialated * 255);
	maximums.clear();
	for (auto i = 0; i < img_float.rows;i++)
	{
		for (auto j = 0; j < img_float.cols; j++)
		{
			float temp= img_float.at<float>(i, j);
			if (temp<1.0f &&temp>band_value && temp==dialated.at<float>(i,j))
			{
				maximums.push_back(cv::Point(j, i));
			}
		}
	}
}
void inline fftAmplitude(cv::Mat img_real, std::vector<float> &amplitude)
{
	int row = img_real.rows;
	int col = img_real.cols;
	int frame_size = row*col;
	FFTLIB::FFT<float> fourierTran;
	fourierTran.create_plan_2d(row, col, 8);
	amplitude.resize(frame_size);
	//firstly make a Fourier transform on the data. 
	std::vector<std::complex<float>> imgData(frame_size);
	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col;j++)
		{
			imgData[i*col+j] = std::complex<float>(img_real.at<float>(i, j), 0.0f);
		}
	}
	fourierTran.forward(imgData);
	FFTLIB::fftshift2(imgData, row, col);
	for (auto i = 0; i < frame_size; i++)
	{
		amplitude[i] += std::abs(imgData[i]);
	}
}
double inline qualityFactorCalc(cv::Mat img_reference, cv::Mat img_modified, std::vector<cv::Point> &maximums)
{
	double sum1=0.0;
	double sum2=0.0;
	cv::Rect ROI(0,0, 100, 100);
	cv::Scalar tempVal2 = cv::mean(img_modified(ROI));
	cv::Scalar tempVal1 = cv::mean(img_reference(ROI));
	double meanValue1 = tempVal1[0];
	double meanValue2 = tempVal2[0];
	for (auto imax = 0; imax< maximums.size(); imax++ )
	{
		cv::Point temp = maximums[imax];
		for (auto i = -2; i < 3; i++)
		{
			for (auto j = -2; j < 3; j++)
			{
				sum1 += (img_reference.at<float>(temp.y+i, temp.x+j) - meanValue1);
				sum2 += (img_modified.at<float>(temp.y + i, temp.x + j)-meanValue2);
			}
		}
	}
	return sum2 / sum1;
}
