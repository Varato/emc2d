#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUEMCRec.h"
#include "thrust_functions.cu"
#include "kernels.cuh"
#include "host_functions.h"
#include "cusparse.h"
#define  MAX_STREAM 64
using namespace EMC;
GPUEMCRec::GPUEMCRec()
{
	row = 0;
	col = 0;
	row_drift = 0;
	col_drift = 0;
	row_pattern = 0;
	col_pattern = 0;
	sparse_mode = false;
}
GPUEMCRec::~GPUEMCRec()
{
	frames.clear();
	model.clear();
	col_index.clear();
	sparse_values.clear();
	row_index.clear();
	cudaFree(model_device);
	cudaFree(weight_device);
	cudaFree(log_prob_device);
	cudaFree(patterns_device);
	if (sparse_mode)
	{
		cudaFree(values_device);
		cudaFree(cols_device);
		cudaFree(csrrow_device);
		cudaFree(rows_device);
	}
	else
	{
		cudaFree(frames_device);
	}
}
GPUEMCRec::GPUEMCRec(std::vector<float> frames, int frame_num, int row_frame, int col_frame, int row_drift, int col_drift, int iteration_num, bool sparse_mode, bool fast_mode)
{
	importFrame(frames, frame_num, row_frame, col_frame, row_drift, col_drift, iteration_num, sparse_mode, fast_mode);
}
void GPUEMCRec::importFrame(std::vector<float> frames, int frame_num, int row_frame, int col_frame, int row_drift, int col_drift, int iteration_num, bool sparse_mode, bool fast_mode)
{
	this->row_pattern = row_frame;
	this->col_pattern = col_frame;
	this->row_drift = row_drift;
	this->col_drift = col_drift;
	this->frame_num = frame_num;
	this->iteration_num = iteration_num;
	//copy the data to the frames contained by the EMC class.
	this->frames.resize(frames.size());
	this->frames.assign(frames.begin(), frames.end());
	//initialize all the remaining data
	this->sparse_mode = sparse_mode;
	this->fast_mode = fast_mode;
	initialize();
}
void GPUEMCRec::initialize()
{
	//allocate data for the vectors
	row = 2 * row_drift + row_pattern;
	col = 2 * col_drift + col_pattern;
	row_range = 2 * row_drift+1;
	col_range = 2 * col_drift+1;
	pattern_num = row_range*col_range;
	pattern_size = row_pattern*col_pattern;
	model_size = row*col;
	pattern_volume = pattern_size*pattern_num;
	//check the validity of the frame size;
	if ((frames.size() / frame_num) != pattern_size)
	{
		std::cout << "data size not match" << std::endl;
		return;
	}
	mem_size = sizeof(float) * frame_num*pattern_num; cudaMalloc((void **)&log_prob_device, mem_size);
	mem_size = sizeof(float) * model_size;            cudaMalloc((void **)&model_device, mem_size);
	cudaMalloc((void **)&weight_device, mem_size);
	if (sparse_mode)
	{
		/************************************************************************/
		sparse_values.resize(frames.size());
		row_index.resize(frames.size()); 
		col_index.resize(frames.size());
		generate_sparse_index(frames, frame_num, pattern_size, sparse_values, row_index, col_index);
		/************************************************************************/
		nonzero_len = sparse_values.size();
		nonzero_ratio = (float)nonzero_len / frames.size();
		/************************************************************************/
		mem_size = sizeof(float) * nonzero_len;
		cudaMalloc((void **)&values_device, mem_size);
		cudaMemcpy(values_device, sparse_values.data(), mem_size, cudaMemcpyHostToDevice);

		mem_size = sizeof(int) * nonzero_len; cudaMalloc((void **)&rows_device, mem_size);
		cudaMalloc((void **)&cols_device, mem_size);
		cudaMemcpy(rows_device, row_index.data(), mem_size, cudaMemcpyHostToDevice);
		cudaMemcpy(cols_device, col_index.data(), mem_size, cudaMemcpyHostToDevice);
		mem_size = sizeof(int) *(frame_num + 1); cudaMalloc((void **)&csrrow_device, mem_size);
		COO2CSR();
		/************************************************************************/
	}
	else
	{
		//directly copy the dense data into the program. 
		nonzero_ratio = 1.0f;
		mem_size = sizeof(float) * frames.size(); cudaMalloc((void **)&frames_device, mem_size);
		cudaMemcpy(frames_device, frames.data(), mem_size, cudaMemcpyHostToDevice);
	}
	if (fast_mode)
	{
		//fast mode would consume quite lot of resource. 
		mem_size = sizeof(float) * pattern_volume;
		cudaMalloc((void **)&patterns_device, mem_size);
	}
	else
	{
		mem_size = sizeof(float) * MAX_STREAM*pattern_size;
		cudaMalloc((void **)&patterns_device, mem_size);
	}
	//initialize the thread and block settings. 
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	// use a larger block size for Fermi and above
	block_size = (deviceProp.major < 2) ? 16 : 32;
	Blk_DIM = dim3(block_size, block_size);
	Grd_DIM = dim3((row + block_size - 1) / block_size, (col + block_size - 1) / block_size);
	Blk_size1D.x = block_size*block_size;
	Grd_size1D.x = row_pattern;
}
void GPUEMCRec::COO2CSR()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	cusparseXcoo2csr(handle, rows_device, nonzero_len, frame_num, csrrow_device, CUSPARSE_INDEX_BASE_ZERO);
}
void GPUEMCRec::expansionDense()
{
	thrust::device_vector<float> pattern_sum(pattern_num);
	float* pattern_sum_device = thrust::raw_pointer_cast(pattern_sum.data());
	device::generate_pattern_sum << <pattern_num, Blk_size1D >> > (model_device, pattern_sum_device, row, col, row_pattern, col_pattern, row_range, col_range);
	device::device_fill(log_prob_device, frame_num*pattern_num, 0.0f);
	thrust::device_ptr<float> log_ptr = thrust::device_pointer_cast(log_prob_device);
	device::log_model << <Grd_DIM, Blk_DIM >> > (model_device, row, col);
	
	//start expansion.
	const float alpha = 1.0;
	const float beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	thrust::device_vector<float> pattern_vector_thrust(pattern_size);
	for (auto i = 0; i < pattern_num; i++)
	{
		int row_start = i / (2 * col_drift + 1);
		int col_start = i % (2 * col_drift + 1);
		float *pattern_vector = thrust::raw_pointer_cast(pattern_vector_thrust.data());
		//generate the pattern value at temporary position from model.
		device::generate_pattern << <Grd_size1D, Blk_size1D >> > (model_device, pattern_vector, row_start, col_start, row, col, row_pattern, col_pattern);
		//get the address of the log_probe_device, which is stored in pattern_num continuous way. 
		float* log_probe_device_temp = thrust::raw_pointer_cast(&log_ptr[frame_num*i]);
		//using the CUBLAS. It's should be notice that the matrix is Column orientated. 
		//matrix frames_device is continuous at a length scale of pattern_size, hence the row (m) is equal to pattern_size, and col (n)
		//is equal to frame_num. So the matrix should be transformed. 
		//Then the size of the pattern vector should be equal to m, and the result should has n size.
		cublasSgemv(handle, CUBLAS_OP_T, pattern_size, frame_num, &alpha, frames_device, pattern_size, pattern_vector, 1, &beta, log_probe_device_temp, 1);
		//get an vector with length frame_num, calculate the sum of pattern vector,  then find the maximum value, and
		device::device_minus_const(log_probe_device_temp, frame_num, handle, pattern_sum[i]);
	}
	pattern_vector_thrust.clear();
}
void GPUEMCRec::expansionSparse()
{
	thrust::device_vector<float> pattern_sum(pattern_num);
	float* pattern_sum_device = thrust::raw_pointer_cast(pattern_sum.data());
	device::generate_pattern_sum << <pattern_num, Blk_size1D >> > (model_device, pattern_sum_device, row, col, row_pattern, col_pattern, row_range, col_range);
	device::device_fill(log_prob_device, frame_num*pattern_num, 0.0f);
	thrust::device_ptr<float> log_ptr = thrust::device_pointer_cast(log_prob_device);
	device::log_model << <Grd_DIM, Blk_DIM >> > (model_device, row, col);
	
	//start to expansion.
	const float alpha = 1.0;
	const float beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusparseHandle_t sparse_handle;
	cusparseCreate(&sparse_handle);
	cusparseMatDescr_t frame_description;
	cusparseCreateMatDescr(&frame_description);
	cusparseSetMatIndexBase(frame_description, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(frame_description, CUSPARSE_MATRIX_TYPE_GENERAL);
	thrust::device_vector<float> pattern_vector_thrust(pattern_size);
	for (auto i = 0; i < pattern_num; i++)
	{
		int row_start = i / (2 * col_drift + 1);
		int col_start = i % (2 * col_drift + 1);
		float *pattern_vector = thrust::raw_pointer_cast(pattern_vector_thrust.data());
		//generate the pattern value at temporary position from model.
		device::generate_pattern << <Grd_size1D, Blk_size1D >> > (model_device, pattern_vector, row_start, col_start, row, col, row_pattern, col_pattern);
		//get the address of the log_probe_device, which is stored in pattern_num continuous way. 
		float* log_probe_device_temp = thrust::raw_pointer_cast(&log_ptr[frame_num*i]);
		cusparseScsrmv(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, frame_num, pattern_size, nonzero_len, &alpha, frame_description,
			values_device, csrrow_device, cols_device, pattern_vector, &beta, log_probe_device_temp);
		//get an vector with length frame_num, calculate the sum of pattern vector,  then find the maximum value, and
		device::device_minus_const(log_probe_device_temp, frame_num, handle, pattern_sum[i]);
	}
	pattern_vector_thrust.clear();
}
void GPUEMCRec::expansionStream()
{
	thrust::device_vector<float> pattern_sum(pattern_num);
	float* pattern_sum_device = thrust::raw_pointer_cast(pattern_sum.data());
	device::generate_pattern_sum << <pattern_num, Blk_size1D >> > (model_device, pattern_sum_device, row, col, row_pattern, col_pattern, row_range, col_range);
	
	//apply logarithm.
	device::device_fill(log_prob_device, frame_num*pattern_num, 0.0f);
	thrust::device_ptr<float> log_ptr = thrust::device_pointer_cast(log_prob_device);
	device::log_model << <Grd_DIM, Blk_DIM >> > (model_device, row, col);
	//start to expansion.
	cublasHandle_t handle;
	cublasCreate(&handle);
	const float alpha = 1.0;
	const float beta = 0.0;
	int start_frame = 0;
	int end_frame;
	int batch_size;
	while (start_frame < pattern_num)
	{
		end_frame = start_frame + MAX_STREAM;
		if (end_frame > pattern_num)
		{
			end_frame = pattern_num;
		}
		batch_size = end_frame - start_frame;
		float* log_probe_device_temp = thrust::raw_pointer_cast(&log_ptr[frame_num*start_frame]);
		pattern_sum_device = thrust::raw_pointer_cast(&pattern_sum[start_frame]);
		device::generate_partial_pattern << <Grd_DIM, Blk_DIM >> > (model_device, patterns_device, start_frame, end_frame, row, col, row_pattern,
			col_pattern, 2 * row_drift + 1, 2 * col_drift + 1, pattern_size);
		device::matrix_sum_row << <batch_size, 256 >> > (patterns_device, pattern_sum_device, batch_size, pattern_size);
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, frame_num, batch_size, pattern_size, &alpha, frames_device, pattern_size, patterns_device, pattern_size, &beta, log_probe_device_temp, frame_num);
		device::matrix_minus_row << <batch_size, 256 >> > (log_probe_device_temp, pattern_sum_device, batch_size, frame_num);
		start_frame = end_frame;
	}
	pattern_sum.clear();
}

void GPUEMCRec::expansionSparseFast()
{	
	thrust::device_vector<float> pattern_sum(pattern_num);
	float* pattern_sum_device = thrust::raw_pointer_cast(pattern_sum.data());
	device::generate_pattern_sum << <pattern_num, Blk_size1D >> > (model_device, pattern_sum_device, row, col, row_pattern, col_pattern, row_range, col_range);

	get_log_patterns();
	//matrix multiplication.
	const float alpha = 1.0;
	const float beta = 0.0;
	device::device_fill(log_prob_device, frame_num*pattern_num, 0.0f);
	//direct matrix multiplication. 
	//using the CUBLAS. It's should be notice that the matrix is Column orientated. 
	//matrix frames_device is continuous at a length scale of pattern_size, hence the row (m) is equal to pattern_size, and col (n)
	//is equal to frame_num. So the matrix should be transformed. 
	//Then the size of the pattern vector should be equal to m, and the result should has n size.
	cusparseHandle_t sparse_handle;
	cusparseCreate(&sparse_handle);
	cusparseMatDescr_t frame_description;
	cusparseCreateMatDescr(&frame_description);
	cusparseSetMatIndexBase(frame_description, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(frame_description, CUSPARSE_MATRIX_TYPE_GENERAL);
	//for sparse matrix multiplication, the B and C matrix is still column based, or the Fortran type. 
	//matrix A row: frame_num.  column: pattern_size.
	cusparseScsrmm(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, frame_num, pattern_num, pattern_size, nonzero_len, &alpha, frame_description,
		values_device, csrrow_device, cols_device, patterns_device, pattern_size, &beta, log_prob_device, frame_num);
	device::matrix_minus_row << <pattern_num, 256 >> > (log_prob_device, pattern_sum_device, pattern_num, frame_num);
	pattern_sum.clear();
}
void GPUEMCRec::expansionDenseFast()
{
	thrust::device_vector<float> pattern_sum(pattern_num);
	float* pattern_sum_device = thrust::raw_pointer_cast(pattern_sum.data());
	device::generate_pattern_sum << <pattern_num, Blk_size1D >> > (model_device, pattern_sum_device, row, col, row_pattern, col_pattern, row_range, col_range);

	get_log_patterns();
	//matrix multiplication.
	const float alpha = 1.0;
	const float beta = 0.0;
	device::device_fill(log_prob_device, frame_num*pattern_num, 0.0f);
	//direct matrix multiplication. 
	//using the CUBLAS. It's should be notice that the matrix is Column orientated. 
	//matrix frames_device is continuous at a length scale of pattern_size, hence the row (m) is equal to pattern_size, and col (n)
	//is equal to frame_num. So the matrix should be transformed. 
	//Then the size of the pattern vector should be equal to m, and the result should has n size.
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, frame_num, pattern_num, pattern_size, &alpha, frames_device, pattern_size, patterns_device, pattern_size, &beta, log_prob_device, frame_num);
	device::matrix_minus_row << <pattern_num, 256 >> > (log_prob_device, pattern_sum_device, pattern_num, frame_num);
	pattern_sum.clear();
}
void GPUEMCRec::maximization()
{
	//a non transpose version. 
	float lower_band = -100.0f;
	thrust::device_vector<float> patterns_Assist(frame_num);
	float* assist_vector_device = thrust::raw_pointer_cast(patterns_Assist.data());
	//log prob array: row: pattern_num, col: frame_num. 
	device::matrix_max_col << <frame_num, 256 >> > (log_prob_device, assist_vector_device, pattern_num, frame_num);
	device::matrix_minus_col << <frame_num, 256 >> > (log_prob_device, assist_vector_device, pattern_num, frame_num);
	device::apply_band_limit << <4, Blk_size1D >> > (log_prob_device, pattern_num*frame_num, lower_band);
	device::exp_array << <4, Blk_size1D >> > (log_prob_device, pattern_num*frame_num);
	device::matrix_sum_col << <frame_num, 256 >> > (log_prob_device, assist_vector_device, pattern_num, frame_num);
	device::matrix_sub_col << <frame_num, 256 >> > (log_prob_device, assist_vector_device, pattern_num, frame_num);
	device::apply_threshold << <4, Blk_size1D >> > (log_prob_device, pattern_num*frame_num, 0.001f);
	patterns_Assist.clear();
}
void GPUEMCRec::compressionDense()
{
	//update the weight.
	update_weight();
	//update the model.
	const float alpha = 1.0;
	const float beta = 0.0;
	cublasHandle_t handle;
	device::device_fill(model_device, model_size, 0.0f);
	thrust::device_ptr<float> log_ptr = thrust::device_pointer_cast(log_prob_device);
	thrust::device_vector<float> pattern_vector_thrust(pattern_size, 0.0f);
	//after adjust the content of log_probe array, apply the matrix multiplication with each column with the length of row_num
	thrust::device_vector<float> weight_array_thrust(model_size, 0.0f);
	float * weight_device = thrust::raw_pointer_cast(weight_array_thrust.data());
	cublasCreate(&handle);
	for (auto i = 0; i < pattern_num; i++)
	{
		int row_start = i / (2 * col_drift + 1);
		int col_start = i % (2 * col_drift + 1);
		//get the weight of each column with the size of frame_num;
		float* log_probe_device_temp = thrust::raw_pointer_cast(&log_ptr[frame_num*i]);
		//reset the value of pattern_vector to zero. 
		thrust::fill(pattern_vector_thrust.begin(), pattern_vector_thrust.end(), 0.0f);
		float *pattern_vector = thrust::raw_pointer_cast(pattern_vector_thrust.data());
		//multiply the frames, and store into the pattern_vector. The matrix should be not transformed. 
		//Then the size of the pattern vector should be equal to n, and the result should has m size.
		cublasSgemv(handle, CUBLAS_OP_N, pattern_size, frame_num, &alpha, frames_device, pattern_size, log_probe_device_temp, 1, &beta, pattern_vector, 1);
		//update the model based on the weight values of the pattern. 
		device::update_model << <Grd_size1D, Blk_size1D >> > (model_device, pattern_vector, row_start, col_start, row, col, row_pattern, col_pattern);
	}
	//get the normalized model. 
	device::array_sub << <Grd_DIM, Blk_DIM >> > (model_device, weight_device, row, col);
	pattern_vector_thrust.clear();
}
void GPUEMCRec::compressionSparse()
{
	//update the weight.
	update_weight();
	//update the model.
	const float alpha = 1.0;
	const float beta = 0.0;
	device::device_fill(model_device, model_size, 0.0f);
	thrust::device_ptr<float> log_ptr = thrust::device_pointer_cast(log_prob_device);
	thrust::device_vector<float> pattern_vector_thrust(pattern_size, 0.0f);
	//after adjust the content of log_probe array, apply the matrix multiplication with each column with the length of row_num
	//set the handles for sparse matrix.
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusparseHandle_t sparse_handle;
	cusparseCreate(&sparse_handle);
	cusparseMatDescr_t frame_description;
	cusparseCreateMatDescr(&frame_description);
	cusparseSetMatIndexBase(frame_description, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(frame_description, CUSPARSE_MATRIX_TYPE_GENERAL);
	for (auto i = 0; i < pattern_num; i++)
	{
		int row_start = i / (2 * col_drift + 1);
		int col_start = i % (2 * col_drift + 1);
		//get the weight of each column with the size of frame_num;
		float* log_probe_device_temp = thrust::raw_pointer_cast(&log_ptr[frame_num*i]);\
		//reset the value of pattern_vector to zero. 
		thrust::fill(pattern_vector_thrust.begin(), pattern_vector_thrust.end(), 0.0f);
		float *pattern_vector = thrust::raw_pointer_cast(pattern_vector_thrust.data());
		//multiply the frames, and store into the pattern_vector. The matrix should be not transformed. 
		//Then the size of the pattern vector should be equal to n, and the result should has m size.
		cusparseScsrmv(sparse_handle, CUSPARSE_OPERATION_TRANSPOSE, frame_num, pattern_size, nonzero_len, &alpha, frame_description,
			values_device, csrrow_device, cols_device, log_probe_device_temp, &beta, pattern_vector);
		//update the model based on the weight values of the pattern. 
		device::update_model << <Grd_size1D, Blk_size1D >> > (model_device, pattern_vector, row_start, col_start, row, col, row_pattern, col_pattern);
	}
	//get the normalized model. 
	device::array_sub << <Grd_DIM, Blk_DIM >> > (model_device, weight_device, row, col);
	pattern_vector_thrust.clear();
}
void GPUEMCRec::compressionStream()
{
	//update the weight value.
	update_weight();
	//begin update the model. 
	cublasHandle_t handle;
	cublasCreate(&handle);
	device::device_fill(model_device, model_size, 0.0f);
	thrust::device_ptr<float> log_ptr = thrust::device_pointer_cast(log_prob_device);
	const float alpha = 1.0;
	const float beta = 0.0;
	int start_frame = 0;
	int end_frame;
	int batch_size;
	while (start_frame < pattern_num)
	{
		end_frame = start_frame + MAX_STREAM;
		if (end_frame > pattern_num)
		{
			end_frame = pattern_num;
		}
		batch_size = end_frame - start_frame;
		device::device_fill(patterns_device, MAX_STREAM*pattern_size, 0.0f);
		float* log_probe_device_temp = thrust::raw_pointer_cast(&log_ptr[frame_num*start_frame]);
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, pattern_size, batch_size, frame_num, &alpha,
			frames_device, pattern_size, log_probe_device_temp, frame_num, &beta, patterns_device, pattern_size);
		device::update_partial_model << <Grd_DIM, Blk_DIM >> > (model_device, patterns_device, start_frame, end_frame,
			row, col, row_pattern, col_pattern, 2 * row_drift + 1, 2 * col_drift + 1, pattern_size);
		start_frame = end_frame;
	}
	//get the normalized model. 
	device::array_sub << <Grd_DIM, Blk_DIM >> > (model_device, weight_device, row, col);
}
void GPUEMCRec::compressionSparseFast()
{
	update_weight();
	const float alpha = 1.0;
	const float beta = 0.0;
	device::device_fill(patterns_device, pattern_volume, 0.0f);
	cusparseHandle_t sparse_handle;
	cusparseCreate(&sparse_handle);
	cusparseMatDescr_t frame_description;
	cusparseCreateMatDescr(&frame_description);
	cusparseSetMatIndexBase(frame_description, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(frame_description, CUSPARSE_MATRIX_TYPE_GENERAL);
	//for sparse matrix multiplication, the B and C matrix is still column based, or the Fortran type. 
	//matrix A row: frame_num.  column: pattern_size.
	cusparseScsrmm(sparse_handle, CUSPARSE_OPERATION_TRANSPOSE, frame_num, pattern_num, pattern_size, nonzero_len, &alpha, frame_description,
		values_device, csrrow_device, cols_device, log_prob_device, frame_num, &beta, patterns_device, pattern_size);
	update_model();
}
void GPUEMCRec::compressionDenseFast()
{
	update_weight();
	const float alpha = 1.0;
	const float beta = 0.0;
	device::device_fill(patterns_device, pattern_volume, 0.0f);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, pattern_size, pattern_num, frame_num, &alpha, frames_device, pattern_size, log_prob_device, frame_num, &beta, patterns_device, pattern_size);
	update_model();
}
void GPUEMCRec::runEMC(std::vector<float> init_model)
{
	model.resize(row*col);
	if (init_model.size() != 0)
	{
		model.assign(init_model.begin(), init_model.end());
	}
	else
	{
		generateRandomArray(model);
	}
	//copy the model from host to device. 
	mem_size = model_size * sizeof(float);
	cudaMemcpy(model_device, model.data(), mem_size, cudaMemcpyHostToDevice);
	//now start to iterate the data.
	if (sparse_mode)
	{
		std::cout << "non zero ratio: " << nonzero_ratio << std::endl;
		if (fast_mode)
		{
			for (auto iter = 0; iter < iteration_num; iter++)
			{
				//std::cout << "iteration: " << iter << std::endl;
				expansionSparseFast();
				maximization();
				compressionSparseFast();
			}
		}
		else
		{
			for (auto iter = 0; iter < iteration_num; iter++)
			{
				//std::cout << "iteration: " << iter << std::endl;
				expansionSparse();
				maximization();
				compressionSparse();
			}
		}
		
	}
	else
	{
		if (fast_mode)
		{
			for (auto iter = 0; iter < iteration_num; iter++)
			{
				std::cout << "iteration: " << iter << std::endl;
				expansionDenseFast();
				maximization();
				compressionDenseFast();
			}
		}
		else
		{
			for (auto iter = 0; iter < iteration_num; iter++)
			{
				std::cout << "iteration: " << iter << std::endl;
				expansionStream();
				maximization();
				compressionStream();
			}
		}
	}
	//When the iteration ends, the data would be copied to the host model matrix. 
	cudaMemcpy(model.data(), model_device, mem_size, cudaMemcpyDeviceToHost);
}
void GPUEMCRec::exportModel(std::vector<float> &model, int &row, int &col)
{
	row = this->row;
	col = this->col;
	model.resize(this->model.size());
	model.assign(this->model.begin(), this->model.end());
}
void GPUEMCRec::get_log_patterns()
{
	device::log_model << <Grd_DIM, Blk_DIM >> > (model_device, row, col);
	device::generate_whole_pattern << <Grd_DIM, Blk_DIM >> > (model_device, patterns_device, row, col, row_pattern,
		col_pattern, 2 * row_drift + 1, 2 * col_drift + 1, pattern_size);
}
void GPUEMCRec::update_model()
{
	//begin update the model and weight. 
	device::update_whole_model << <Grd_DIM, Blk_DIM >> > (model_device, patterns_device, row, col, row_pattern,
		col_pattern, 2 * row_drift + 1, 2 * col_drift + 1, pattern_size);
	device::array_sub << <Grd_DIM, Blk_DIM >> > (model_device, weight_device, row, col);
}
void GPUEMCRec::update_weight()
{
	thrust::device_vector<float> weights_sum(pattern_num);
	float* weight_sum_device = thrust::raw_pointer_cast(weights_sum.data());
	device::matrix_sum_row << < pattern_num, 256 >> > (log_prob_device, weight_sum_device, pattern_num, frame_num);
	device::update_whole_weight << <Grd_DIM, Blk_DIM >> > (weight_device, weight_sum_device, row, col, row_pattern,
		col_pattern, 2 * row_drift + 1, 2 * col_drift + 1, pattern_size);
	weights_sum.clear();
}
void GPUEMCRec::generateRandomArray(std::vector<float> &randomArray)
{
	size_t length = randomArray.size();
	for (int i = 0; i < length; i++)
	{
		int ran_num = std::rand();
		if (ran_num != 0)
		{
			randomArray[i] = ((float)ran_num) / RAND_MAX;
		}
		else
		{
			randomArray[i] = 0.1f / RAND_MAX;
		}
	}
} 
