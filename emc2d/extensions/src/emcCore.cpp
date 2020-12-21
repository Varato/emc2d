//
// Created by Chen on 21/12/2020.
//

#include <tuple>
#include <vector>
#include <algorithm>    // std::transform
#include <cmath>
#include <cstdio>
#include <immintrin.h>  // avx intrinsics

#include "emcCore.h"

const float_type eps = 1e-13f;

// inline
// uint32_t getFramesPixel(uint32_t framesFlat[], size_t numPixels, size_t k, size_t i){
//     /*
//      * The reason for this function is that, in future, we want the frames_flat to be in some sparse format.
//      * Adding this abstraction helps with changing the code.
//      * */
//     return framesFlat[k * numPixels + i];
// }


// code from https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
// to reduce-sum a __m256 vector of 8 floats
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
static float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


/*
 * Notes for AVX
 * _mm256_load_ps load 32-Byte-ALIGNED data, meaning the data must be located at 32n, n=0,1,2... address in memory
 * _mm256_store_ps store a __m256 to 32-Byte-ALIGNED memory, too.
 *
 * Because here the arrays are from numpy single float, which obviously only guarantees 4-Byte alignment. This is why using
 * _mm256_load_ps and _mm256_store_ps results in segmentation fault.
 *
 * _mm256_loadu_ps and _mm256_storeu_ps do not require the 32-Byte alignment. So use them.
 */

inline
float_type frameRowLikelihood(float_type *FkRow, float_type logMjRow[], float_type MjRow[], size_t w) {

    float_type reduced;
    size_t remainder_w = w % 8;
    size_t whole_w = w - remainder_w;

    __m256 reduced256 = _mm256_setzero_ps();
    __m256 mul256, ll;
    size_t i = 0;
    for (; i < whole_w; i += 8) {
        // FkRow[i] * logMjRow[i] - MjRow[i];
        mul256 = _mm256_mul_ps(_mm256_loadu_ps(FkRow + i), _mm256_loadu_ps(logMjRow + i));
        ll = _mm256_sub_ps(mul256, _mm256_loadu_ps(MjRow + i));
        reduced256 = _mm256_add_ps(reduced256, ll);
    }
    reduced = sum8(reduced256);

    if (remainder_w > 0) {
        for (; i < w; ++i) {
            reduced += FkRow[i] * logMjRow[i] - MjRow[i];
        }
    }
    return reduced;
}


//inline
//void mergeOneRow0(float_type MjRow[], float_type const FkRow[], float_type wjk, size_t w) {
//
//    for (size_t i = 0; i < w; ++i){
//        MjRow[i] += wjk * FkRow[i];
//    }
//}


inline
void mergeOneRow(float_type MjRow[], float_type FkRow[], float_type wjk, size_t w) {
    // only marginal speed gain using avx here ...

    size_t remainder_w = w % 8;
    size_t whole_w = w - remainder_w;

    __m256 MjRow256, FkRow256;
    __m256 wjk256 = _mm256_set1_ps(wjk);

    size_t i = 0;
    // MjRow[i] += wjk * FkRow[i];
    for (; i < whole_w; i+=8) {
        MjRow256 = _mm256_loadu_ps(MjRow + i);
        FkRow256 = _mm256_loadu_ps(FkRow + i);
        MjRow256 = _mm256_add_ps(MjRow256, _mm256_mul_ps(wjk256, FkRow256));
        _mm256_storeu_ps(MjRow + i, MjRow256);
    }
    if (remainder_w > 0) {
        for(; i < w; ++i) {
            MjRow[i] += wjk * FkRow[i];
        }
    }
}


void computeLogLikelihoodMap(float_type framesFlat[],
                             float_type model[],
                             size_t H, size_t W,  // model dims
                             size_t h, size_t w,  // frame dims
                             size_t numPixels,
                             size_t numFrames,
                             uint32_t const driftsInUse[],
                             size_t numDriftsInUse,
                             size_t driftRadiusY,
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
     * maxDriftX and driftRadiusY define the drift space of dimensions (2*maxDriftX + 1, 2*driftRadiusY + 1)
     * Assume the origin is at the corner, i.e. (x, y) = (0, 0) is the first drift.
     */

    size_t modelNumPixels = W * H;

    // pre-compute the slow log
    std::vector<float_type> logModel(modelNumPixels);
    std::transform(model, model + modelNumPixels, logModel.begin(), [](float_type v){return log(v + eps);});

#pragma omp parallel for
    for (int k = 0; k < numFrames; ++k) {
        for (size_t j = 0; j < numDriftsInUse; ++j) {
            size_t t = driftsInUse[j];
            size_t x = t / (2*driftRadiusY + 1);
            size_t y = t % (2*driftRadiusY + 1);

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


void mergeFramesSoft(float_type framesFlat[],
                     size_t h, size_t w,
                     size_t H, size_t W,
                     float_type const mergeWeights[],
                     size_t numFrames,
                     uint32_t const driftsInUse[],
                     size_t driftRadiusY,
                     size_t numDriftsInUse,
                     float_type output[]) {

    size_t x, y, t;
    size_t numPixels = w*h;

    float_type wjk, wj0;
    float_type *modelRowPtr;
    float_type *vTimeRowPtr;
    float_type *frameRowPtr;
    std::vector<float_type> visitingTimes(H*W, 0);
    std::fill(output, output + H*W, 0.0f);

    // deal with k=0 for visitingTimes.
    for (size_t j = 0; j < numDriftsInUse; ++j) {
        t = driftsInUse[j];
        x = t / (2*driftRadiusY + 1);
        y = t % (2*driftRadiusY + 1);
        wj0 = mergeWeights[j*numFrames];
        for (int row = 0; row < h; ++row) {
            modelRowPtr = output + (x + row) * W + y;
            vTimeRowPtr = visitingTimes.data() + (x + row) * W + y;
            frameRowPtr = framesFlat + row * w;
            mergeOneRow(modelRowPtr, frameRowPtr, wj0, w);
            // this for loop needs only one run when k=0
            for (size_t i=0; i < w; ++i) {
                vTimeRowPtr[i] += 1;
            }
        }
    }
    for (int k = 1; k < numFrames; ++k) {
        for (size_t j = 0; j < numDriftsInUse; ++j) {
            t = driftsInUse[j];
            x = t / (2*driftRadiusY + 1);
            y = t % (2*driftRadiusY + 1);
            wjk = mergeWeights[j*numFrames + k];
            for (int row = 0; row < h; ++row) {
                modelRowPtr = output + (x + row) * W + y;
                frameRowPtr = framesFlat + k * numPixels + row * w;
                mergeOneRow(modelRowPtr, frameRowPtr, wjk, w);
            }
        }
    }

    for (size_t i = 0; i < W*H; ++i) {
        if (visitingTimes[i] > 0)
            output[i] /= visitingTimes[i];
    }
}