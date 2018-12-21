#include "stdlib.h"
#include <random>
#include <vector>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
namespace EMC
{
	class GPUEMCRec
	{
	public:
		GPUEMCRec();
		GPUEMCRec(std::vector<float> frames, int frame_num, int row_frame, int col_frame, int row_drift,
			int col_drift, int iteration_num, bool sparse_mode = false, bool fast_mode = true);
		~GPUEMCRec();
		void importFrame(std::vector<float> frames, int frame_num, int row_frame, int col_frame, int row_drift,
			int col_drift, int iteration_num, bool sparse_mode = false, bool fast_mode = true);
		void runEMC(std::vector<float> init_model);
		void exportModel(std::vector<float> &model, int &row, int &col);
	private:
		float* model_device;
		float* frames_device;
		float* weight_device;
		float* log_prob_device;
		float* values_device;
		float* patterns_device;
		int* rows_device;
		int* cols_device;
		int* csrrow_device;
		std::vector<float> model;
		std::vector<float> frames;
		std::vector<float> sparse_values;
		std::vector<int> row_index;
		std::vector<int> col_index;
		int row;
		int col;
		int nonzero_len;
		bool sparse_mode;
		bool fast_mode;
		int row_pattern;
		int col_pattern;
		int row_range;
		int col_range;
		int row_drift;
		int col_drift;
		int pattern_size;
		int model_size;
		int pattern_num;
		int frame_num;
		int iteration_num;
		int block_size;
		size_t pattern_volume;
		dim3 Blk_DIM;
		dim3 Grd_DIM;
		dim3  Blk_size1D;
		dim3  Grd_size1D;
		float nonzero_ratio;
		unsigned int mem_size;
		void initialize();
		void expansionDense();
		void expansionStream();
		void expansionSparse();
		void expansionSparseFast();
		void expansionDenseFast();
		void maximization();
		void compressionDense();
		void compressionStream();
		void compressionSparse();
		void compressionSparseFast();
		void compressionDenseFast();
		void generateRandomArray(std::vector<float> &randomArray);
		void get_log_patterns();
		void update_model();
		void update_weight();
		void COO2CSR();
	};
}
