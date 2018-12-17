#include <cstdlib>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "EMCRec.h"
#include "host_functions.h"
#include "host_functions.h"
#include "noise_filter.hpp"
#include "contrast.hpp"
#include "reconstructor.hpp"


void run_random_Poisson_CPU()
{
	cv::Mat img = cv::imread("mof.bmp", 0);
	int row_drift = 5;
	int col_drift = 5;
	int frame_num = 100;
	std::ofstream fileOut;
	fileOut.open("error.txt", std::ios::app);
	for (auto i=1; i<= 10; i++)
	{
		float dose = 0.02*i;
		EMC::Constructor emc_const;
		emc_const.init(img, row_drift, col_drift, frame_num, dose);
		emc_const.demo_data_gene();
		emc_const.real_solution_gene();
		emc_const.recon_from_random_guess();
		fileOut << emc_const.model_error_gene() << std::endl;
	}
}
//void k2test()
//{
//	cv::Mat img_frames;
//	double min_value, max_value;
//	cv::FileStorage fs("frames.yml", cv::FileStorage::READ);
//	fs["frames"] >> img_frames;
//	std::cout << img_frames.rows << std::endl;
//	std::cout << img_frames.cols << std::endl;
//	std::vector<float> model;
//	std::vector<float> frames;
//	int col_frame = img_frames.cols;
//	int row_frame = col_frame;
//	int row_drift = 11;
//	int col_drift = 11;
//	int row = row_frame + 2 * row_drift;
//	int col = col_frame + 2 * col_drift;
//	cv::Rect ROI(col_drift, row_drift, col_frame, row_frame);
//	cv::Mat img_float=cv::Mat::zeros(row, col, CV_32FC1);
//	cv::Mat img_ROI = img_float(ROI);
//	cv::Mat approximation=cv::imread("Aligned.jpg", 0);
//	approximation.convertTo(img_ROI, CV_32FC1);
//	cv::minMaxLoc(img_ROI, &min_value, &max_value);
//	img_float = img_float / max_value;
//	cv::imshow("initial", img_float);
//	cv::waitKey(0);
//	int frame_num = img_frames.rows / col_frame;
//	int frame_size = img_frames.rows*img_frames.cols;
//	frames.resize(frame_size);
//	memcpy(frames.data(), img_frames.data, sizeof(float)*frame_size);
//	EMC::EMCRec emcback;
//	emcback.importFrame(frames, frame_num, row_frame, col_frame, row_drift, col_drift, 50);
//	emcback.applyDriftLimit(45.0f*CV_PI / 180.0f, 1.0);
//	std::vector<float> ini_model;
//	ini_model.resize(row*col);
//	memcpy( ini_model.data(), img_float.data, sizeof(float)*row*col);
//	emcback.runEMC(ini_model, true);
//	//get the established model.
//	model.resize(row * col);
//	emcback.exportModel(model, row, col);
//	memcpy(img_float.data, model.data(), sizeof(float)*row*col);
//	cv::minMaxLoc(img_float(ROI), &min_value, &max_value);
//	img_float = img_float / max_value;
//	cv::imshow("corrected", img_float(ROI));
//	cv::imwrite("D:/corrected.bmp", img_float(ROI) * 255);
//	cv::waitKey(0);
//}
int main()
{
	run_random_Poisson_CPU();
	system("pause");
	return 0;
}
