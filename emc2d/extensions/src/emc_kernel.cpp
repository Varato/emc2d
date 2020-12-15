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

typedef py::array_t<double, py::array::c_style | py::array::forcecast> py_array_double_ctype;
typedef py::array_t<uint32_t , py::array::c_style | py::array::forcecast> py_array_uint32_ctype;

const double eps = 1e-13;
const double LjkLowerBound = -600;

// inline
// uint32_t getFramesPixel(uint32_t framesFlat[], size_t numPixels, size_t k, size_t i){
//     /*
//      * The reason for this function is that, in future, we want the frames_flat to be in some sparse format.
//      * Adding this abstraction helps with changing the code.
//      * */
//     return framesFlat[k * numPixels + i];
// }


double frameRowLikelihood(double FkRow[], double logWjRow[], double WjRow[], size_t w) {

    const unsigned UNROLL = 4;
    double reduced;
    size_t whole_w = w / (4 * UNROLL);

    __m256d reduced256 = _mm256_set_pd(0.0F, 0.0F, 0.0F, 0.0F);
    __m256d mul256, ll;
    for (size_t i = 0; i < whole_w; i += 4) {
        // FkRow[i] * logWjRow[i] - WjRow[i];
        mul256 = _mm256_mul_pd(_mm256_load_pd(FkRow + i), _mm256_load_pd(logWjRow + i));
        ll = _mm256_sub_pd(mul256, _mm256_load_pd(WjRow + i));
        reduced256 = _mm256_add_pd(reduced256, ll);        
    }

    _mm256_store_pd(&reduced, reduced256);
    return reduced;
}


void computeLogLikelihoodMap(double framesFlat[],
                             double model[],
                             size_t H, size_t W,  // model dims
                             size_t h, size_t w,  // frame dims
                             size_t numPixels,
                             size_t numFrames,
                             uint32_t driftsInUse[],
                             size_t numDriftsInUse,
                             size_t maxDriftY,
                             double output[]) {
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
    std::vector<double> logModel(modelNumPixels);
    std::transform(model, model + modelNumPixels, logModel.begin(), [](double v){return log(v + eps);});

    #pragma omp parallel for
    for (int k = 0; k < numFrames; ++k) {
        for (size_t j = 0; j < numDriftsInUse; ++j) {
            int t = driftsInUse[j];
            int x = t / (2*maxDriftY + 1);
            int y = t % (2*maxDriftY + 1);

            double Ljk = 0;  //cumulator
            for (int row = 0; row < h; ++row) {
                double *modelRowPtr = model + (x + row) * W + y;
                double *logModelRowPtr = logModel.data() + (x + row) * W + y;
                double *frameRowPtr = framesFlat + k * numPixels + row * w;
                Ljk += frameRowLikelihood(frameRowPtr, logModelRowPtr, modelRowPtr, w);
            }
            output[j * numFrames + k] = Ljk;
        }
    }
}


void mergeFramesIntoModel(uint32_t framesFlat[],
                          size_t h, size_t w,
                          size_t H, size_t W,
                          double membershipProbability[],
                          size_t numFrames,
                          uint32_t driftsInUse[],
                          size_t maxDriftY,
                          size_t numDriftsInUse,
                          double output[]) {

    size_t x, y, t;
    size_t Mi, Mj;
    double Wji, Pjk;
    double norm;

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
            output[i] /= (double)visitingTimes[i];
    }
}

py::array computeLogLikelihoodMapWrapper(py_array_double_ctype framesFlat,
                                         py_array_double_ctype model,
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
    py::array output = make2dArray<double>(numDriftsInUse, numFrames);

    double* framesFlatPtr = (double *)(framesFlatInfo.ptr);
    double* modelPtr = (double *)(model.request().ptr);
    uint32_t* driftInUserPtr = (uint32_t *)(driftsInUseInfo.ptr);
    double* outPtr = (double *)(output.request().ptr);

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
                                      py_array_double_ctype membershipProbability,
                                      py_array_uint32_ctype driftsInUse) {

    py::array output = make2dArray<double>(H, W);
    py::buffer_info framesFlatInfo = framesFlat.request();
    py::buffer_info memProbInfo = membershipProbability.request();
    py::buffer_info driftsInUseInfo = driftsInUse.request();


    size_t numFrames = framesFlatInfo.shape[0];
    size_t numDriftsInUse = memProbInfo.shape[0];

    uint32_t* framesFlatPtr = (uint32_t *)(framesFlatInfo.ptr);
    double* memProbPtr = (double *) (memProbInfo.ptr);
    uint32_t* driftInUserPtr = (uint32_t *)(driftsInUseInfo.ptr);
    double* outPtr = (double *)(output.request().ptr);

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