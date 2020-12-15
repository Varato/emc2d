#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <limits>
#include <vector>
#include <algorithm>    // std::transform
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>


#include "utils.h"


namespace py = pybind11;
typedef float float_type;

typedef py::array_t<float_type, py::array::c_style | py::array::forcecast> py_array_float_ctype;
typedef py::array_t<uint32_t , py::array::c_style | py::array::forcecast> py_array_uint32_ctype;

const float_type eps = 1e-13f;
const float_type LjkLowerBound = -600.0f;

// inline
// uint32_t getFramesPixel(uint32_t framesFlat[], size_t numPixels, size_t k, size_t i){
//     /*
//      * The reason for this function is that, in future, we want the frames_flat to be in some sparse format.
//      * Adding this abstraction helps with changing the code.
//      * */
//     return framesFlat[k * numPixels + i];
// }


float_type frameRowLikelihood(float_type FkRow[], float_type logWjRow[], float_type WjRow[], size_t w) {

    float_type reduced;
    size_t remainder_w = w % 8;
    size_t whole_w = w - remainder_w;

    __m256 reduced256 = _mm256_setzero_ps();
    __m256 mul256, ll;

    size_t i = 0;
    for (; i < whole_w; i += 8) {
        // FkRow[i] * logWjRow[i] - WjRow[i];
        mul256 = _mm256_mul_ps(_mm256_load_ps(FkRow + i), _mm256_load_ps(logWjRow + i));
        ll = _mm256_sub_ps(mul256, _mm256_load_ps(WjRow + i));
        reduced256 = _mm256_add_ps(reduced256, ll);        
    }
    _mm256_store_ps(&reduced, reduced256);

    if (remainder_w > 0) {
        for (; i < w; ++i) {
            reduced += FkRow[i] * logWjRow[i] - WjRow[i];
        }
    }
    return reduced;
}


void computeLogLikelihoodMap(float_type framesFlat[],
                             float_type model[],
                             size_t H, size_t W,  // model dims
                             size_t h, size_t w,  // frame dims
                             size_t numPixels,
                             size_t numFrames,
                             uint32_t driftsInUse[],
                             size_t numDriftsInUse,
                             size_t maxDriftY,
                             float_type output[]) {
    /* Logic dimensions of arrays:
     *     N = number of frames
     *     M = number of all drifts
     *     m = number of effective drifts
     *     npix = w * h
     *
     *     framesFlat: (N, npix)
     *     model: (H, W)
     *     driftsInUse: (m, )
     *
     *     output: (m, N)
     *
     * maxDriftX and maxDriftY define the drift space of dimensions (2*maxDriftX + 1, 2*maxDriftY + 1)
     * Assume the origin is at the corner, i.e. (x, y) = (0, 0) is the first drift.
     */

    size_t modelNumPixels = W*H;

    // pre-compute the slow log
    std::vector<float_type> logModel(modelNumPixels);
    std::transform(model, model + modelNumPixels, logModel.begin(), [](float_type v){return log(v + eps);});

    #pragma omp parallel for
    for (int k = 0; k < numFrames; ++k) {
        for (size_t j = 0; j < numDriftsInUse; ++j) {
            size_t t = driftsInUse[j];
            size_t x = t / (2*maxDriftY + 1);
            size_t y = t % (2*maxDriftY + 1);

            float_type Ljk = 0;  //cumulator
            for (int row = 0; row < h; ++row) {
                float_type *modelRowPtr = model + (x + row) * W + y;
                float_type *logModelRowPtr = logModel.data() + (x + row) * W + y;
                float_type *frameRowPtr = framesFlat + k * numPixels + row * w;
                Ljk += frameRowLikelihood(frameRowPtr, logModelRowPtr, modelRowPtr, w);
            }
            output[j * numFrames + k] = Ljk;
        }
    }
}


void mergeFramesIntoModel(uint32_t framesFlat[],
                          size_t h, size_t w,
                          size_t H, size_t W,
                          float_type membershipProbability[],
                          size_t numFrames,
                          uint32_t driftsInUse[],
                          size_t maxDriftY,
                          size_t numDriftsInUse,
                          float_type output[]) {

    size_t x, y, t;
    size_t Mi, Mj;
    float_type Wji, Pjk;
    float_type norm;

    size_t numPixels = w*h;

    std::vector<unsigned> visitingTimes(W*H, 0);

    for (size_t j = 0; j < numDriftsInUse; ++j) {
        t = driftsInUse[j];
        x = t / (2*maxDriftY + 1);
        y = t % (2*maxDriftY + 1);

        for (size_t i = 0; i < numPixels; ++i) {
            Mi = (i / w) + x;
            Mj = (i % w) + y;
            visitingTimes[Mi * W + Mj] += 1;

            Wji = 0;
            norm = 0;
            for (size_t k = 0; k < numFrames; ++k) {
                Pjk = membershipProbability[j*numFrames + k];
                Wji += framesFlat[k*numPixels + i] * Pjk;
                norm += Pjk;
            }
            output[Mi * W + Mj] = Wji / (norm + eps);
        }
    }

    for (size_t i = 0; i < W*H; ++i) {
        if (visitingTimes[i] > 0)
            output[i] /= (float_type)visitingTimes[i];
    }
}

py::array computeLogLikelihoodMapWrapper(py_array_float_ctype framesFlat,
                                         py_array_float_ctype model,
                                         unsigned w, unsigned h,
                                         unsigned maxDriftY,
                                         py_array_uint32_ctype driftsInUse) {

    py::buffer_info framesFlatInfo = framesFlat.request();
    py::buffer_info modelInfo = model.request();
    py::buffer_info driftsInUseInfo = driftsInUse.request();

    size_t numFrames = framesFlatInfo.shape[0];
    size_t numPixels = framesFlatInfo.shape[1];
    size_t numDriftsInUse = driftsInUseInfo.shape[0];
    size_t H = modelInfo.shape[0];
    size_t W = modelInfo.shape[1];
    py::array output = make2dArray<float_type>(numDriftsInUse, numFrames);

    float_type* framesFlatPtr = (float_type *)(framesFlatInfo.ptr);
    float_type* modelPtr = (float_type *)(model.request().ptr);
    uint32_t* driftInUserPtr = (uint32_t *)(driftsInUseInfo.ptr);
    float_type* outPtr = (float_type *)(output.request().ptr);

    computeLogLikelihoodMap(framesFlatPtr,
                            modelPtr,
                            H, W,
                            h, w,
                            numPixels,
                            numFrames,
                            driftInUserPtr,
                            numDriftsInUse,
                            maxDriftY,
                            outPtr);
    return output;
}


py::array mergeFramesIntoModelWrapper(py_array_uint32_ctype framesFlat,
                                      size_t h, size_t w,
                                      size_t H, size_t W,
                                      size_t maxDriftY,
                                      py_array_float_ctype membershipProbability,
                                      py_array_uint32_ctype driftsInUse) {

    py::array output = make2dArray<float_type>(H, W);
    py::buffer_info framesFlatInfo = framesFlat.request();
    py::buffer_info memProbInfo = membershipProbability.request();
    py::buffer_info driftsInUseInfo = driftsInUse.request();


    size_t numFrames = framesFlatInfo.shape[0];
    size_t numDriftsInUse = memProbInfo.shape[0];

    uint32_t* framesFlatPtr = (uint32_t *)(framesFlatInfo.ptr);
    float_type* memProbPtr = (float_type *) (memProbInfo.ptr);
    uint32_t* driftInUserPtr = (uint32_t *)(driftsInUseInfo.ptr);
    float_type* outPtr = (float_type *)(output.request().ptr);

    mergeFramesIntoModel(framesFlatPtr,
                         h, w,
                         H, W,
                         memProbPtr,
                         numFrames,
                         driftInUserPtr,
                         maxDriftY,
                         numDriftsInUse,
                         outPtr);
    return output;
}


PYBIND11_MODULE(emc_kernel, m) {
    m.def("compute_log_likelihood_map",
          &computeLogLikelihoodMapWrapper,
          py::arg("frames_flat"),
          py::arg("model"),
          py::arg("h"), py::arg("w"),
          py::arg("max_drift_y"),
          py::arg("drifts_in_use"));

    m.def("merge_frames_into_model",
          &mergeFramesIntoModelWrapper,
          py::arg("frames_flat"),
          py::arg("h"), py::arg("w"),
          py::arg("H"), py::arg("W"),
          py::arg("max_drift_y"),
          py::arg("membership_probability"),
          py::arg("drifts_in_use")
    );
}