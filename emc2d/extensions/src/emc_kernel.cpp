#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <cmath>


#include "emcCore.h"
#include "utils.h"


namespace py = pybind11;

typedef py::array_t<float_type, py::array::c_style | py::array::forcecast> py_array_float_ctype;
typedef py::array_t<uint32_t , py::array::c_style | py::array::forcecast> py_array_uint32_ctype;
typedef py::array_t<int , py::array::c_style | py::array::forcecast> py_array_int_ctype;



/* ----- pybind11 wrappers ----- */
py::array computeLogLikelihoodMapWrapper(const py_array_float_ctype& framesFlat,
                                         const py_array_float_ctype& model,
                                         unsigned h, unsigned w,
                                         unsigned driftRadiusY,
                                         const py_array_uint32_ctype& driftsInUse) {

    py::buffer_info framesFlatInfo = framesFlat.request();
    py::buffer_info modelInfo = model.request();
    py::buffer_info driftsInUseInfo = driftsInUse.request();

    size_t numFrames = framesFlatInfo.shape[0];
    size_t numPixels = framesFlatInfo.shape[1];
    size_t numDriftsInUse = driftsInUseInfo.shape[0];
    size_t H = modelInfo.shape[0];
    size_t W = modelInfo.shape[1];
    py::array output = make2dArray<float_type>(numDriftsInUse, numFrames);

    auto* framesFlatPtr = (float_type *)(framesFlatInfo.ptr);
    float_type* modelPtr = (float_type *)(model.request().ptr);
    auto* driftInUserPtr = (uint32_t *)(driftsInUseInfo.ptr);
    float_type* outPtr = (float_type *)(output.request().ptr);

    computeLogLikelihoodMap(framesFlatPtr,
                            modelPtr,
                            H, W,
                            h, w,
                            numPixels,
                            numFrames,
                            driftInUserPtr,
                            numDriftsInUse,
                            driftRadiusY,
                            outPtr);
    return output;
}

py::array computeLogLikelihoodMapFrameSparseWrapper(const py_array_float_ctype& framesFlat,
                                                    const py_array_float_ctype& model,
                                                    unsigned h, unsigned w,
                                                    unsigned driftRadiusX,
                                                    unsigned driftRadiusY,
                                                    const py_array_int_ctype& frameDriftsInUse) {

    py::buffer_info framesFlatInfo = framesFlat.request();
    py::buffer_info modelInfo = model.request();
    py::buffer_info frameDriftsInUseInfo = frameDriftsInUse.request();

    size_t numDrifts = (2*driftRadiusX + 1) * (2*driftRadiusY + 1);

    size_t numFrames = framesFlatInfo.shape[0];
    size_t numPixels = framesFlatInfo.shape[1];
    size_t H = modelInfo.shape[0];
    size_t W = modelInfo.shape[1];
    size_t maxNumFrameDrifts = frameDriftsInUseInfo.shape[1];
    py::array output = make2dArray<float_type>(numDrifts, numFrames);

    auto* framesFlatPtr = (float_type *)(framesFlatInfo.ptr);
    float_type* modelPtr = (float_type *)(model.request().ptr);
    int * frameDriftsInUsePtr = (int *)(frameDriftsInUseInfo.ptr);
    float_type* outPtr = (float_type *)(output.request().ptr);

    computeLogLikelihoodMapFrameSparse(framesFlatPtr,
                                       modelPtr,
                                       H, W,
                                       h, w,
                                       numPixels,
                                       numFrames,
                                       frameDriftsInUsePtr,
                                       maxNumFrameDrifts,
                                       driftRadiusX,
                                       driftRadiusY,
                                       outPtr);
    return output;
}

py::array mergeFramesSoftWrapper(const py_array_float_ctype& framesFlat,
                                 size_t h, size_t w,
                                 size_t H, size_t W,
                                 size_t driftRadiusY,
                                 const py_array_float_ctype& mergeWeights,
                                 const py_array_uint32_ctype& driftsInUse) {

    py::array output = make2dArray<float_type>(H, W);
    py::buffer_info framesFlatInfo = framesFlat.request();
    py::buffer_info mergeWeightsInfo = mergeWeights.request();
    py::buffer_info driftsInUseInfo = driftsInUse.request();

    size_t numFrames = framesFlatInfo.shape[0];
    size_t numDriftsInUse = mergeWeightsInfo.shape[0];

    auto* framesFlatPtr = (float_type *)(framesFlatInfo.ptr);
    auto* mergeWeightsPtr = (float_type *) (mergeWeightsInfo.ptr);
    auto* driftInUserPtr = (uint32_t *)(driftsInUseInfo.ptr);
    float_type* outPtr = (float_type *)(output.request().ptr);

    mergeFramesSoft(framesFlatPtr,
                    h, w,
                    H, W,
                    mergeWeightsPtr,
                    numFrames,
                    driftInUserPtr,
                    driftRadiusY,
                    numDriftsInUse,
                    outPtr);
    return output;
}


/* ----- pybind11 module def -----*/
PYBIND11_MODULE(emc_kernel, m) {
    m.def("compute_log_likelihood_map",
          &computeLogLikelihoodMapWrapper,
          py::arg("frames_flat"),
          py::arg("model"),
          py::arg("h"), py::arg("w"),
          py::arg("drift_radius_y"),
          py::arg("drifts_in_use"));

    m.def("merge_frames_soft",
          &mergeFramesSoftWrapper,
          py::arg("frames_flat"),
          py::arg("h"), py::arg("w"),
          py::arg("H"), py::arg("W"),
          py::arg("drift_radius_y"),
          py::arg("merge_weights"),
          py::arg("drifts_in_use")
    );

    m.def("compute_log_likelihood_map_frame_sparse",
          &computeLogLikelihoodMapFrameSparseWrapper,
          py::arg("frames_flat"),
          py::arg("model"),
          py::arg("h"), py::arg("w"),
          py::arg("drift_radius_x"),
          py::arg("drift_radius_y"),
          py::arg("frame_drifts_in_use"));
}
