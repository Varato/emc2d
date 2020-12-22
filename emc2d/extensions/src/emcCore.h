//
// Created by Chen on 21/12/2020.
//

#ifndef EMC2D_EMCCORE_H
#define EMC2D_EMCCORE_H

typedef float float_type;

void computeLogLikelihoodMap(float_type framesFlat[],
                             float_type model[],
                             size_t H, size_t W,  // model dims
                             size_t h, size_t w,  // frame dims
                             size_t numPixels,
                             size_t numFrames,
                             uint32_t const driftsInUse[],
                             size_t numDriftsInUse,
                             size_t driftRadiusY,
                             float_type output[]);

void mergeFramesSoft(float_type framesFlat[],
                     size_t h, size_t w,
                     size_t H, size_t W,
                     float_type const mergeWeights[],
                     size_t numFrames,
                     uint32_t const driftsInUse[],
                     size_t driftRadiusY,
                     size_t numDriftsInUse,
                     float_type output[]);

void computeLogLikelihoodMapFrameSparse(float_type framesFlat[],
                                        float_type model[],
                                        size_t H, size_t W,  // model dims
                                        size_t h, size_t w,  // frame dims
                                        size_t numPixels,
                                        size_t numFrames,
                                        int32_t const frameDriftsInUse[],
                                        size_t maxNumFrameDrifts,
                                        size_t driftRadiusX,
                                        size_t driftRadiusY,
                                        float_type output[]);

#endif //EMC2D_EMCCORE_H
