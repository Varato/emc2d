#pragma once

#include <random>
#include <vector>
#include <iostream>

#include "frames.hpp"


namespace emc
{
	class EmcCorrector
	{
	public:
		EmcCorrector();
		EmcCorrector(const FrameSet& frames, int row_drift, int col_drift, int iteration_num);
		~EmcCorrector();
		void importFrames(const FrameSet&);
		void importModel(std::vector<float> model, int row, int col);
		void importMask(std::vector<float> mask);
		void applyDriftLimit(float drift_angle, float drift_width);
		void generateFrames(std::vector<float> &frames, int frame_num, int row_drift, int col_drift);
		void runEMC(std::vector<float> init_model, bool mode);
		void exportModel(std::vector<float> &model, int &row, int &col);
		void exportProb(std::vector<float> &prob_mat, int &row, int &col);
		void gene_solution(const std::vector<float> &prob_mat);
		float getSparseRatio();
	private:
		std::vector<float> mask;
		std::vector<float> model;
		std::vector<float> old_model;
		std::vector<float> patterns;
		std::vector<float> prob;
		std::vector<float> weights;
		std::vector<float> frames;
		std::vector<float> pattern_sum;
		std::vector<int>   pattern_drifts;
		std::vector<float> sparse_values;
		std::vector<long long> row_index;
		std::vector<long long> col_index;
		int row;          int col;
		int row_pattern;  int col_pattern;
		int row_drift;    int col_drift;
		int pattern_size; int model_size;
		int pattern_num;  int frame_num;
		int iteration_num; bool angle_limit_mode;
		float drift_angle; float drift_width;
		float nonzero_ratio; int iter; bool Fourier_filter;
		size_t pattern_volume; size_t nonzero_length;
		void initialize();
		void initialSparse();
		void initialPattern();
		void initialModel();
		void generatePatternDrift();
		void expand();
		void expandSparse();
		void maximize();
		void maximizeSparse();
		void compress();
		void compressSparse();
		void update_prob();
		void update_weight();
		void generatePatternSum();
		void generateRandomArray(std::vector<float> &randomArray);
		void expandPatterns();
		void model_error();
	};
}

