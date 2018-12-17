#pragma once
#include "stdlib.h"
#include <random>
#include <vector>
#include <iostream>
namespace EMC
{
	class EMC_Generator
	{
	public:
		EMC_Generator()
		{
			row = 0;
			col = 0;
			row_drift = 0;
			col_drift = 0;
			row_pattern = 0;
			col_pattern = 0;
			angle_limit_mode = false;
		}
		void importModel(std::vector<float> model, int row, int col)
		{
			this->model.resize(model.size());
			this->model.assign(model.begin(), model.end());
			this->row = row;
			this->col = col;
		}
		void applyDriftLimit(float drift_angle, float drift_width)
		{
			this->drift_angle = drift_angle;
			this->drift_width = drift_width;
			this->angle_limit_mode = true;
		}
		void generateFrames(std::vector<float> &frames, int frame_num, int row_drift, int col_drift)
		{
			if (row == 0 || col == 0)
			{
				std::cout << "please input the model and dimension." << std::endl;
				return;
			}
			this->row_drift = row_drift;
			this->col_drift = col_drift;
			this->row_pattern = row - 2 * row_drift;
			this->col_pattern = col - 2 * col_drift;
			this->pattern_size = this->row_pattern*this->col_pattern;
			this->frame_num = frame_num;
			initialPattern();
			expandPatterns();
			frames.resize(pattern_size*frame_num);
			std::fill(prob.begin(), prob.end(), 0.0f);
			//randomly select the frames from the generated patterns
			std::uniform_int_distribution<int> linearDistri(0, pattern_num - 1);
			std::mt19937_64 random_Engine;
			for (int i = 0; i < frame_num; i++)
			{
				int index = linearDistri(random_Engine);
				prob[pattern_num*i + index] = 1.0;
				memcpy(&frames[i*pattern_size], &patterns[index*pattern_size], sizeof(float)*pattern_size);
			}
		}
		void exportProb(std::vector<float> &prob_mat, int &row, int &col)
		{
			row = this->frame_num;
			col = this->pattern_num;
			prob_mat.resize(this->prob.size());
			prob_mat.assign(this->prob.begin(), this->prob.end());
		}
	private:
		std::vector<float> model;
		std::vector<float> patterns;
		std::vector<float> prob;
		std::vector<float> frames;
		std::vector<float> pattern_sum;
		std::vector<int>   pattern_drifts;
		int row;          int col;
		int row_pattern;  int col_pattern;
		int row_drift;    int col_drift;
		int pattern_size; int model_size;
		int pattern_num;  int frame_num;
		bool angle_limit_mode;
		float drift_angle; float drift_width;
		size_t pattern_volume; size_t nonzero_length;
		void initialPattern()
		{
			generatePatternDrift();
			pattern_num = pattern_drifts.size() / 2;
			pattern_volume = pattern_size*pattern_num;
			patterns.resize(pattern_volume);
			pattern_sum.resize(pattern_num);
			prob.resize(frame_num*pattern_num);
		}
		void generatePatternDrift()
		{
			pattern_drifts.clear();
			for (auto i = -row_drift; i < row_drift + 1; i++)
			{
				for (auto j = -col_drift; j < col_drift + 1; j++)
				{
					if (angle_limit_mode)
					{
						float pl_distance = std::sin(drift_angle)*j + std::cos(drift_angle)*i;
						if (abs(pl_distance) < drift_width)
						{
							pattern_drifts.push_back(i + row_drift);
							pattern_drifts.push_back(j + col_drift);
						}
					}
					else
					{
						pattern_drifts.push_back(i + row_drift);
						pattern_drifts.push_back(j + col_drift);
					}
				}
			}
		}
		void expandPatterns()
		{
			size_t pattern_position;
			//the generated pattern is in row priority. 
			for (auto iPattern = 0; iPattern < pattern_num; iPattern++)
			{
				pattern_position = iPattern*pattern_size;
				for (auto i = 0; i < row_pattern; i++)
				{
					for (auto j = 0; j < col_pattern; j++)
					{
						int temporaty_index = i*col_pattern + j;
						int model_index = (pattern_drifts[iPattern * 2] + i)*col + pattern_drifts[iPattern * 2 + 1] + j;
						patterns[temporaty_index + pattern_position] = model[model_index];
					}
				}
			}
		}
	};
}

