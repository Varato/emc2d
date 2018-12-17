#ifndef THRUST_FUNCTION_H
#define THRUST_FUNCTION_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "stream.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
namespace device
{
	void inline device_fill(float * device_vector, int length, float value)
	{
		thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(device_vector);
		thrust::fill(dev_ptr, dev_ptr + length, value);
	}
	void inline device_minus_const(float * device_vector, int length, cublasHandle_t handle, float const_value)
	{
		// wrap raw pointer with a device_ptr 
		const float alpha = 1.0;
		thrust::device_vector<float> device_const_array(length, -const_value);
		float* assist_device = thrust::raw_pointer_cast(&device_const_array[0]);
		cublasSaxpy(handle, length, &alpha, assist_device, 1, device_vector, 1);
		device_const_array.clear();
	}
	float inline device_get_sum(float * device_vector, int length, cublasHandle_t handle)
	{
		thrust::device_vector<float> assist_vector(length, 1.0f);
		float* assist_device = thrust::raw_pointer_cast(assist_vector.data());
		float sum_result;
		//using the BLAS to calculate the value. 
		cublasSdot(handle, length, device_vector, 1, assist_device, 1, &sum_result);
		assist_vector.clear();
		return sum_result;
	}
	float inline device_get_max(float * device_vector, int length)
	{
		thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(device_vector);
		return *(thrust::max_element(dev_ptr, dev_ptr + length));
	}
	void inline device_get_row_sum(float * device_vector, int row, int col, std::vector<float> &max_values, int stream_num)
	{
		int ipos = 0;
		Stream cudaStreams(stream_num);
		cublasHandle_t handle;
		cublasCreate(&handle);
		//std::vector<cublasHandle_t> handles(stream_num);
		thrust::device_ptr<float> patterns_ptr = thrust::device_pointer_cast(device_vector);
		while (ipos < row)
		{
			cudaStreams.set_n_act_stream(row - ipos);
			for (auto istream = 0; istream < cudaStreams.n_act_stream; istream++)
			{
				thrust::device_vector<float> assist_vector(col, 1.0f);
				float* assist_device = thrust::raw_pointer_cast(assist_vector.data());
				float *pattern_vector_pointer = thrust::raw_pointer_cast(&patterns_ptr[(ipos + istream)*col]);
				//cublasCreate(&handles[istream]);
				cublasSetStream(handle, cudaStreams[istream]);
				cublasSdot(handle, col, pattern_vector_pointer, 1, assist_device, 1, &max_values[ipos + istream]);
				assist_vector.clear();
			}
			ipos += cudaStreams.n_act_stream;
		}
		cudaStreams.synchronize();
	}
	void inline device_minus_const_row(float * device_vector, int row, int col, std::vector<float> max_values, int stream_num)
	{
		int ipos = 0;
		Stream cudaStreams(stream_num);
		cublasHandle_t handle;
		cublasCreate(&handle);
		//std::vector<cublasHandle_t> handles(stream_num);
		thrust::device_ptr<float> patterns_ptr = thrust::device_pointer_cast(device_vector);
		while (ipos < row)
		{
			cudaStreams.set_n_act_stream(row - ipos);
			for (auto istream = 0; istream < cudaStreams.n_act_stream; istream++)
			{
				const float alpha = 1.0;
				thrust::device_vector<float> assist_vector(col, -max_values[ipos + istream]);
				float* assist_device = thrust::raw_pointer_cast(assist_vector.data());
				float *pattern_vector_pointer = thrust::raw_pointer_cast(&patterns_ptr[(ipos + istream)*col]);
				//cublasCreate(&handles[istream]);
				cublasSetStream(handle, cudaStreams[istream]);
				cublasSaxpy(handle, col, &alpha, assist_device, 1, pattern_vector_pointer, 1);
				assist_vector.clear();
			}
			ipos += cudaStreams.n_act_stream;
		}
		cudaStreams.synchronize();
	}
	void inline device_get_row_max(float * device_vector, int row, int col, std::vector<float> &max_values, int stream_num)
	{
		//get the normal of the matrix using CUBLAS. 
		cublasHandle_t handle;
		cublasCreate(&handle);
	}
	void inline device_norm_row(float * device_vector, int row, int col, std::vector<float> &sum_values, int stream_num)
	{
		int ipos = 0;
		Stream cudaStreams(stream_num);
		cublasHandle_t handle;
		cublasCreate(&handle);
		//std::vector<cublasHandle_t> handles(stream_num);
		thrust::device_ptr<float> patterns_ptr = thrust::device_pointer_cast(device_vector);
		while (ipos < row)
		{
			cudaStreams.set_n_act_stream(row - ipos);
			for (auto istream = 0; istream < cudaStreams.n_act_stream; istream++)
			{
				float *pattern_vector_pointer = thrust::raw_pointer_cast(&patterns_ptr[(ipos + istream)*col]);
				//cublasCreate(&handles[istream]);
				cublasSetStream(handle, cudaStreams[istream]);
				cublasSscal(handle, col, &sum_values[ipos + istream], pattern_vector_pointer, 1);
			}
			ipos += cudaStreams.n_act_stream;
		}
		cudaStreams.synchronize();
	}
}

#endif // 