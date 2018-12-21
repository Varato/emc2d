#pragma once
#include <cstdlib>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include "EMCRec.h"
#include "host_functions.h"
#include "host_functions.h"
#include "noise_filter.hpp"
#include "contrast.hpp"
#include "EMC_Generator.h"
namespace EMC
{
	class Constructor
	{
	public:
		inline void init(cv::Mat img_i, int row_drift_i, int col_drift_i, int frame_num_i, float dose_i)
		{
			row = img_i.rows; col = img_i.cols;
			row_drift = row_drift_i;  col_drift = col_drift_i;
			row_frame = row - 2 * row_drift;
			col_frame = col - 2 * col_drift;
			frame_num = frame_num_i;
			dose = dose_i;
			img_i.convertTo(img_float, CV_32FC1);
			model.resize(row * col);
			ROI=cv::Rect(col_drift, row_drift, col_frame, row_frame);
		}
		inline void real_solution_gene()
		{
			EMC::EMCRec emcback;
			std::vector<float> ini_model;
			emcback.importFrame(frames, frame_num, row_frame, col_frame, row_drift, col_drift, 80);
			emcback.importMask(mask);
			emcback.applyDriftLimit(45.0f*CV_PI / 180.0f, 0.5);
			emcback.gene_solution(prob_mat);
			emcback.exportModel(model_solution, row, col);
		}
		inline void mask_gene()
		{
			std::vector<float> averaged_amp;
			cv::Mat amp_img(row_frame, col_frame, CV_32FC1);
			cv::Mat amp_img_full(row, col, CV_32FC1);
			Filter::generate_fftAmp(frames, frame_num, row_frame, col_frame, averaged_amp);
			FFTLIB::fftshift2(averaged_amp, row_frame, col_frame);
			memcpy(amp_img.data, averaged_amp.data(), sizeof(float)*row_frame*col_frame);
			Contrast::Log_Contrast(amp_img);
			Contrast::Linear_Contrast(amp_img, true);
			//transfer the data from the sub frame to the full frame.
			cv::Mat sub_image = amp_img_full(ROI);
			amp_img.copyTo(sub_image);
			Filter::generate_mask(amp_img_full, mask, 5);
			FFTLIB::fftshift2(mask, row, col);
			cv::imshow("mask", amp_img_full);
			cv::waitKey(0);
		}
		inline void demo_data_gene()
		{
			double min_value, max_value;
			cv::minMaxLoc(img_float, &min_value, &max_value);
			img_float = img_float / max_value;
			EMC::EMC_Generator emc_gene;
			memcpy(model.data(), img_float.data, sizeof(float)*row*col);
			emc_gene.importModel(model, row, col);
			emc_gene.applyDriftLimit(45.0f*CV_PI / 180.0f, 0.5);
			emc_gene.generateFrames(frames, frame_num, row_drift, col_drift);
			emc_gene.exportProb(prob_mat, row_prob, col_prob);
			addPoissonNoise(frames, dose);
		}
		inline void recon_from_solution()
		{
			cv::Mat model_result(img_float.size(), img_float.type());
			double min_value, max_value;
			EMC::EMCRec emcback;
			emcback.importFrame(frames, frame_num, row_frame, col_frame, row_drift, col_drift, 20);
			std::vector<float> ini_model;
			ini_model.resize(row*col);
			memcpy(ini_model.data(), img_float.data, sizeof(float)*row*col);
			/*memcpy(ini_model.data(), model_solution.data(), sizeof(float)*row*col);*/
			emcback.applyDriftLimit(45.0f*CV_PI / 180.0f, 1.0);
			emcback.runEMC(ini_model, true);
			emcback.exportModel(model, row, col);
			emcback.exportProb(prob_mat, row_prob, col_prob);
			cv::Mat img_prob(row_prob, col_prob, CV_32FC1);
			memcpy(img_prob.data, prob_mat.data(), sizeof(float)*row_prob*col_prob);
			cv::imshow("prob_log", img_prob);
			//show the model.
			memcpy(model_result.data, model.data(), sizeof(float)*row*col);
			cv::minMaxLoc(model_result(ROI), &min_value, &max_value);
			model_result = model_result / max_value;
			cv::imshow("corrected_solution", model_result(ROI));
			cv::imwrite("D:/corrected_solution"+ std::to_string(frame_num)+ " "+ std::to_string(dose)+ ".bmp", model_result(ROI) * 255);
			cv::imwrite("D:/prob_solution.bmp", img_prob * 255);
			cv::waitKey(0);
		}
		inline void recon_from_random_guess()
		{
			cv::Mat model_result(img_float.size(), img_float.type());
			double min_value, max_value;
			EMC::EMCRec emcback;
			emcback.importFrame(frames, frame_num, row_frame, col_frame, row_drift, col_drift, 60);
			emcback.applyDriftLimit(45.0f*CV_PI / 180.0f, 1.0);
			std::vector<float> ini_model;
			ini_model.clear();
			emcback.runEMC(ini_model, true);
			emcback.exportModel(model, row, col);
			emcback.exportProb(prob_mat, row_prob, col_prob);
			cv::Mat img_prob(row_prob, col_prob, CV_32FC1);
			memcpy(img_prob.data, prob_mat.data(), sizeof(float)*row_prob*col_prob);
			cv::imshow("prob_log", img_prob);
			//show the model.
			memcpy(model_result.data, model.data(), sizeof(float)*row*col);
			cv::minMaxLoc(model_result(ROI), &min_value, &max_value);
			model_result = model_result / max_value;
			cv::imshow("corrected", model_result(ROI));
			cv::imwrite("D:/corrected.bmp", model_result(ROI) * 255);
			cv::imwrite("D:/prob.bmp", img_prob * 255);
			cv::waitKey(0);
		}
		inline double model_error_gene()
		{
			double error_sum;
			for (auto i = 0; i < row*col; i++)
			{
				error_sum += abs(model[i] - model_solution[i]);
			}
			//error_sum /= model_size;
			return error_sum;
		}
	private:
		int row, col;
		int row_drift, col_drift;
		int row_frame, col_frame;
		int row_prob,  col_prob;
		int frame_num;
		float dose;
		std::vector<float> model;
		std::vector<float> model_solution;
		std::vector<float> frames;
		std::vector<float> prob_mat;
		std::vector<float> mask;
		cv::Mat img_float;
		cv::Rect ROI;
	};
}