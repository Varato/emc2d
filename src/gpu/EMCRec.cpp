#include "EMCRec.h"
#include "host_functions.h"
#include "noise_filter.hpp"
using namespace EMC;
EMCRec::EMCRec()
{
	row = 0;
	col = 0;
	row_drift = 0;
	col_drift = 0;
	row_pattern = 0;
	col_pattern = 0;
	angle_limit_mode = false;
	Fourier_filter = false;
}
EMCRec::~EMCRec()
{
	frames.clear();
	patterns.clear();
	weights.clear();
	prob.clear();
	pattern_sum.clear();
	model.clear();
	old_model.clear();
	sparse_values.clear();
	row_index.clear();
	col_index.clear();
}
EMCRec::EMCRec(std::vector<float> frames, int frame_num, int row_frame, int col_frame, int row_drift, int col_drift, int iteration_num)
{
	importFrame(frames, frame_num, row_frame, col_frame, row_drift, col_drift, iteration_num);
}
void EMCRec::importFrame(std::vector<float> frames, int frame_num, int row_frame, int col_frame, int row_drift, int col_drift, int iteration_num)
{
	this->row_pattern = row_frame;
	this->col_pattern = col_frame;
	this->row_drift = row_drift;
	this->col_drift = col_drift;
	this->frame_num = frame_num;
	this->iteration_num = iteration_num;
	this->pattern_size = this->row_pattern*this->col_pattern;
	if ((frames.size() / frame_num) != this->pattern_size)
	{
		std::cout << "data size not match" << std::endl;
		return;
	}
	//copy the data to the frames contained by the EMC class.
	this->frames.resize(frames.size());
	this->frames.assign(frames.begin(), frames.end());
}
void EMCRec::applyDriftLimit(float drift_angle, float drift_width)
{
	this->drift_angle = drift_angle;
	this->drift_width = drift_width;
	this->angle_limit_mode = true;
}
void EMCRec::initialize()
{
	initialModel();
	initialPattern();
	initialSparse();
	std::cout << "complete initialization" << std::endl;
	std::cout << "row_pattern: " << row_pattern<< std::endl;
	std::cout << "pattern_num: " << pattern_num<< std::endl;
	std::cout << "model row: " <<row<< std::endl;
	std::cout << "frame size: " <<pattern_size<< std::endl;
	std::cout << "frame num: " << frame_num << std::endl;

}
void EMCRec::initialModel()
{
	row = 2 * row_drift + row_pattern;
	col = 2 * col_drift + col_pattern;
	model_size = row*col;
	//check the validity of the frame size;
	model.resize(model_size);
	old_model.resize(model_size);
	weights.resize(model_size);
}
void EMCRec::initialPattern()
{
	generatePatternDrift();
	pattern_num = pattern_drifts.size()/2;
	pattern_volume = pattern_size*pattern_num;
	patterns.resize(pattern_volume);
	pattern_sum.resize(pattern_num);
	prob.resize(frame_num*pattern_num);
}
void EMCRec::initialSparse()
{
	//acquire the sparse information
	sparse_values.resize(frames.size());
	row_index.resize(frames.size());
	col_index.resize(frames.size());
	//calculate the sparsity of the input frames, and establish the coordinate compression format.
	generate_sparse_index(frames, frame_num, pattern_size, sparse_values, row_index, col_index);
	nonzero_length = sparse_values.size();
	nonzero_ratio = ((float)nonzero_length) / frame_num / pattern_size;
}
void EMCRec::generatePatternDrift()
{
	pattern_drifts.clear();
	for (auto i = -row_drift; i < row_drift + 1; i++)
	{
		for (auto j = -col_drift; j < col_drift + 1; j++)
		{
			if (angle_limit_mode)
			{
				float pl_distance=std::sin(drift_angle)*j + std::cos(drift_angle)*i;
				if (abs(pl_distance)<drift_width)
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
float EMCRec::getSparseRatio()
{
	return nonzero_ratio;
}
//this expansion is only for the generation of export frames.
void EMCRec::expandPatterns()
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
void EMCRec::generatePatternSum()
{
	//assign the value of the model to the old model for further comparison.
	old_model.assign(model.begin(), model.end());
	//the pattern sum is generated for each pattern. 
	for (auto iPattern = 0; iPattern < pattern_num; iPattern++)
	{
		double sum = 0.0;
		for (auto i = 0; i < row_pattern; i++)
		{
			for (auto j = 0; j < col_pattern; j++)
			{
				int model_index = (pattern_drifts[iPattern * 2] + i)*col + pattern_drifts[iPattern * 2 + 1] + j;
				sum+=model[model_index];
			}
		}
		pattern_sum[iPattern] = (float)sum;
	}
	//generate the log patterns.
	for (auto i = 0; i < model_size; i++)
	{
		model[i] = (float)std::log((double)model[i] + 0.000000001);
	}
}
void EMCRec::expand()
{
	generatePatternSum();
	expandPatterns();
}
void EMCRec::expandSparse()
{
	generatePatternSum();
	//the generated pattern is stored in column major matrix serving for further sparse matrix multiplication. 
	for (auto iPattern = 0; iPattern < pattern_num; iPattern++)
	{
		for (auto i = 0; i < row_pattern; i++)
		{
			for (auto j = 0; j < col_pattern; j++)
			{
				int temporaty_index = i*col_pattern + j;
				int model_index = (pattern_drifts[iPattern * 2] + i)*col + pattern_drifts[iPattern * 2 + 1] + j;
				patterns[temporaty_index*pattern_num + iPattern] = model[model_index];
			}
		}
	}
}
void EMCRec::maximize()
{
	//assigning the value of the logarithm probability
	//calculate the likelihood for each frame.
	//the dimension of frames is frame_num * pattern_size, row_oriented. 
	//the dimension of patterns is pattern_num * pattern_size, row_oriented. 
	//the dimension of log is frame_num * pattern_num, row_oriented. 
	matrixMul(frames, true, patterns, false, prob, frame_num, pattern_num, pattern_size);
	//update the value of log-prob matrix and the weight value.
	update_prob();
	update_weight();
	//the dimension of prob is frame_num * pattern_num, row_oriented. 
	//the dimension of frames is frame_num * pattern_size, row_oriented. 
	//the dimension of patterns is pattern_num * pattern_size, row_oriented. 
	matrixMul(prob, false, frames, true, patterns, pattern_num, pattern_size, frame_num);
}
void EMCRec::maximizeSparse()
{
	//sparse matrix multiplication. The difference is brought by whether the pattern matrix is row major or column major. 
	//the dimension of patterns is  pattern_size *pattern_num, col oriented, different from the previous one. 
	//the dimension of log is frame_num * pattern_num, row_oriented.
	matrixMulSparse(sparse_values, row_index, col_index, patterns, prob, frame_num, pattern_num, pattern_size);
	update_prob();
	update_weight();
	matrixMulSparse(sparse_values, col_index, row_index, prob, patterns, pattern_size, pattern_num, frame_num);
}
void EMCRec::compress()
{
	size_t pattern_position;
	for (auto iPattern = 0; iPattern < pattern_num; iPattern++)
	{
		pattern_position = iPattern*pattern_size;
		for (auto i = 0; i < row_pattern; i++)
		{
			for (auto j = 0; j < col_pattern; j++)
			{
				int temporaty_index = i*col_pattern + j;
				int model_index = (pattern_drifts[iPattern * 2] + i)*col + pattern_drifts[iPattern * 2 + 1] + j;
				model[model_index] += patterns[temporaty_index + pattern_position];
			}
		}
	}
	//update the content of the model via the weight value. 
	for (auto i = 0; i < model_size; i++)
	{
		if (weights[i] > 0)
		{
			model[i] = model[i] / weights[i];
		}
	}
}
void EMCRec::compressSparse()
{
	for (auto iPattern = 0; iPattern < pattern_num; iPattern++)
	{
		for (auto i = 0; i < row_pattern; i++)
		{
			for (auto j = 0; j < col_pattern; j++)
			{
				int temporaty_index = i*col_pattern + j;
				int model_index = (pattern_drifts[iPattern * 2] + i)*col + pattern_drifts[iPattern * 2 + 1] + j;
				model[model_index] += patterns[temporaty_index*pattern_num + iPattern];
			}
		}
	}
	//update the content of the model via the weight value. 
	for (auto i = 0; i < model_size; i++)
	{
		if (weights[i] > 0)
		{
			model[i] = model[i] / weights[i];
		}
	}
}
void EMCRec::update_weight()
{
	// update the value of pattern. 
	for (auto i = 0; i < pattern_num; i++)
	{
		pattern_sum[i] = 0.0f;
		for (auto j = 0; j < frame_num; j++)
		{
			pattern_sum[i] += prob[i + j*pattern_num];
		}
	}
	for (auto i = 0; i < model_size; i++)
	{
		weights[i] = 0.0f;
	}

	for (auto iPattern = 0; iPattern < pattern_num; iPattern++)
	{
		for (auto i = 0; i < row_pattern; i++)
		{
			for (auto j = 0; j < col_pattern; j++)
			{
				int model_index = (pattern_drifts[iPattern * 2] + i)*col + pattern_drifts[iPattern * 2 + 1] + j;
				weights[model_index] += pattern_sum[iPattern];
			}
		}
	}
	//reset the value of model.
	for (auto i = 0; i < model_size; i++)
	{
		model[i] = 0.0f;
	}
}
void EMCRec::update_prob()
{
	std::vector<float> max_value(frame_num);
	std::vector<float> sum_value(frame_num, 0.0f);
	float lower_band = -100.0f;

	//reduce each column with frame_num size the same value
	//which is the sum of each pattern, then get the maximum
	//value of each row.
	for (auto i = 0; i < frame_num; i++)
	{
		for (auto j = 0; j < pattern_num; j++)
		{
			prob[i*pattern_num+j] -= pattern_sum[j];
			if (j == 0 || (prob[i*pattern_num+j] > max_value[i]))
			{
				max_value[i] = prob[i*pattern_num+j];
			}
		}
	}
	//each row reduced by their maximum value, and calculate the sum. 
	for (auto i = 0; i < frame_num; i++)
	{
		for (auto j = 0; j < pattern_num; j++)
		{
			prob[i*pattern_num+j] -= max_value[i];
			if (prob[i*pattern_num+j] < lower_band)
			{
				prob[i*pattern_num+j] = lower_band;
			}
			prob[i*pattern_num+j] = std::exp(prob[i*pattern_num+j]);
			sum_value[i] += prob[i*pattern_num+j];
		}
	}
	//make the normalization
	for (auto i = 0; i < frame_num; i++)
	{
		for (auto j = 0; j < pattern_num; j++)
		{
			prob[i*pattern_num+j] /= sum_value[i];
			if (prob[i*pattern_num+j]<0.001f)
			{
				prob[i*pattern_num+j] = 0.0f;
			}
		}
	}
	sum_value.clear();
	max_value.clear();
}
void EMCRec::runEMC(std::vector<float> init_model, bool mode)
{
	//initialize all the parameters.
	initialize();
	if (init_model.size() != 0)
	{
		model.assign(init_model.begin(), init_model.end());
	}
	else
	{
		//the model is start form a random guess.
		generateRandomArray(model);
	}
	iter = 0;
	if (mode)
	{
		frames.clear();
		std::cout << "sparse ratio: " << nonzero_ratio << std::endl;
		while (iter< iteration_num)
		{
			//std::cout << "iteration:" << iter << std::endl;
			expandSparse();
			maximizeSparse();
			compressSparse();
			model_error();
		}
	}
	else
	{
		while (iter< iteration_num)
		{
			//std::cout << "iteration:" << iter << std::endl;
			expand();
			maximize();
			compress();
			model_error();
		}
	}
}
void EMCRec::gene_solution(const std::vector<float> &prob_mat)
{
	prob.assign(prob_mat.begin(), prob_mat.end());
	initialize();
	update_weight();
	//the dimension of prob is frame_num * pattern_num, row_oriented. 
	//the dimension of frames is frame_num * pattern_size, row_oriented. 
	//the dimension of patterns is pattern_num * pattern_size, row_oriented. 
	matrixMul(prob, false, frames, true, patterns, pattern_num, pattern_size, frame_num);
	compress();
}
void EMCRec::exportModel(std::vector<float> &model, int &row, int &col)
{
	row = this->row;
	col = this->col;
	model.resize(this->model.size());
	model.assign(this->model.begin(), this->model.end());
}
void EMCRec::exportProb(std::vector<float> &prob_mat, int &row, int &col)
{
	row = this->frame_num;
	col = this->pattern_num;
	prob_mat.resize(this->prob.size());
	prob_mat.assign(this->prob.begin(), this->prob.end());
}
void EMCRec::generateRandomArray(std::vector<float> &randomArray)
{
	size_t length = randomArray.size();
	srand(time(NULL));
	for (int i = 0; i < length; i++)
	{
		int ran_num = std::rand();
		if (ran_num != 0)
		{
			randomArray[i] = ((float)ran_num) / RAND_MAX;
		}
		else
		{
			randomArray[i] = 0.01f / RAND_MAX;
		}
	}
}
void EMCRec::importModel(std::vector<float> model, int row, int col)
{
	this->model.resize(model.size());
	this->model.assign(model.begin(), model.end());
	this->row = row;
	this->col = col;
}
void EMCRec::importMask(std::vector<float> mask_i)
{
	mask.assign(mask_i.begin(), mask_i.end());
	Fourier_filter = true;
}
void EMCRec::generateFrames(std::vector<float> &frames, int frame_num, int row_drift, int col_drift)
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
void EMCRec::model_error()
{
	double error_sum=0.0f;
	for (auto i = 0; i < model_size;i++)
	{
		error_sum += abs(model[i] - old_model[i]);
	}
	//error_sum /= model_size;
	std::cout << "error: " << error_sum << std::endl;
	if (error_sum<0.01)
	{
		iter++;
		if (iter<iteration_num)
		{
			if (Fourier_filter)
			{
				Filter::apply_Fourier_Filter(model, mask, row, col);
			}
			else
			{
				Filter::applyBWLimit(model, row, col, 70);
			}
			std::vector<float> random_model(model_size);
			generateRandomArray(random_model);
			for (auto i = 0; i < model_size; i++)
			{
				model[i] = (model[i] + 8.0*random_model[i]) /9.0f;
			}
		}
	}
}