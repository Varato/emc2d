# Is memory saving possible in EMC

## Notations

- N: number of frames
- M: number of translations (or orientations) under consideration
- n: number of pixels in each frame
- h, w: height and width of each frame
- H, W: height and width of the model
- R: drift range

in motion correction, we have
H = h + 2R,
W = w + 2R.

## Problem

Although usually we do not care about memory but only time complexity, saving memory may be useful for motion-correction EMC because M grows quadratically with the given drift rage. As Deepan has observed, when the drift range is larger than 125 pixels, the computation becomes difficult.

During EMC iterations, the model is expanded into M patches, and then the M by N probability matrix is computed.
But essentially, we only need the probability matrix for subsequent computations. Therefore, as long as we can pick up pixels correctly in the computations of the probability matrix, we do not have to save the expanded M patches in the memory. The same idea holds when merging frames into the model. Specifically, if we know where each frame pixel will be compared with (to compute likelihood) or merged into (for the compress step) a pixel in the model, we can directly do that.

The problem, however, might be a low-level efficiency trade-off. By having the expanded model as a (M by n) matrix in memory, the probability matrix computation becomes a neat matrix multiplication ((M, n) @ (n, N) -> (M, N)), which can be efficiently handled by optimized linear algebra code (like numpy). On the contrary, if we pick up pixels "randomly" to virtually have the (M by n) matrix, cache-missing (and other possible issues) can happen, resulting in low hardware efficiency.

So far, by using AVX intrinsics (a CPU SIMD instruction set which vectorize computation of 256 bits at a time) and OpenMP (very cheating to numpy), I can make the memery-saving computation of the probability matrix comprably fast as my old numpy implementation. But for the merging step, my best trial is still a multiple (2 to 5 times) slower than the numpy implementation.
The follwoing two figures show a prob-matrix-only comparison and a full-one-step-EMC comparison, respectively.

![bm_prob](/benchmark/benchmarkProbMatrix.png)
![bm_one_step](/benchmark/benchmarkOneStepEMC.png)

## Snippets

The full code is at 

### Code for computing the probability matrix

```cpp
// code from https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
// to reduce-sum a __m256 vector of 8 floats
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum8(__m256 x) {
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


inline
float_type frameRowLikelihood(float_type FkRow[], float_type logMjRow[], float_type MjRow[], size_t w) {

    float_type reduced;
    size_t remainder_w = w % 8;
    size_t whole_w = w - remainder_w;

    __m256 reduced256 = _mm256_setzero_ps();
    __m256 mul256, ll;

    size_t i = 0;
    for (; i < whole_w; i += 8) {
        // FkRow[i] * logMjRow[i] - MjRow[i];
        mul256 = _mm256_mul_ps(_mm256_load_ps(FkRow + i), _mm256_load_ps(logMjRow + i));
        ll = _mm256_sub_ps(mul256, _mm256_load_ps(MjRow + i));
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

    size_t modelNumPixels = W * H;

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
```

### Code for merging frames in to one (H, W)-sized model

```cpp
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
        MjRow256 = _mm256_load_ps(MjRow + i);
        FkRow256 = _mm256_load_ps(FkRow + i);
        MjRow256 = _mm256_add_ps(MjRow256, _mm256_mul_ps(wjk256, FkRow256));
        _mm256_store_ps(MjRow + i, MjRow256);
    }
    if (remainder_w > 0) {
        for(; i < w; ++i) {
            MjRow[i] += wjk * FkRow[i];
        }
    }
}


void mergeFramesSoft(float_type framesFlat[],
                     size_t h, size_t w,
                     size_t H, size_t W,
                     float_type mergeWeights[],
                     size_t numFrames,
                     uint32_t driftsInUse[],
                     size_t maxDriftY,
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
        x = t / (2*maxDriftY + 1);
        y = t % (2*maxDriftY + 1);
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
            x = t / (2*maxDriftY + 1);
            y = t % (2*maxDriftY + 1);
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
```
