/*plane cuda kernels*/

#define MAX_BLOCK_SIZE 1024


__global__
void genPattern(float *model, int modelWidth, int modelHeight, 
                float *pattern, int patternWidth, int patternHeight,
                int rowStart, int colStart)
/* generate one pattern from model so that the left top corner of the pattern 
 * is at (rowStart, colStart) of the model 
 */
{
    int patternCol = threadIdx.x;  // map to column index of the pattern
    int patternRow = blockIdx.x;   // map to row index of the pattern

    // if the blockDim.x >= patternWidth, this while loop will be executed 
    // once only (i.e. each thread takes care one pixel).
    while (patternCol < patternWidth) {
        int modelIndex = (patternRow + rowStart) * modelWidth + patternCol + colStart;
        int patternIndex = patternRow * patternWidth + patternCol;
        pattern[patternIndex] = model[modelIndex]
        patternCol += blockDim.x;
    }
}

__global__ 
void genAllPatterns(float *model, int modelWidth, int modelHeight,
                    float *allPatterns, int patternWidth, int patternHeight)
{
    int modelCol = threadIdx.x + blockIdx.x * blockDim.x;
    int modelRow = threadIdx.y + blockIdx.y + blockDim.y;
    int modelIndex = modelCol + modelRow * modelWidth;
    float threadModelValue = model[modelIndex];  // the model value owned by one thread.
    int patternSize = patternWidth * patternHeight;
    //int driftLimitHorizantal[2];
    //int driftLimitVertical[2];


    for (int rowStart = 0; rowStart < driftRangeVert; ++rowStart) 
        for (int colStart = 0; colStart < driftRangeHori; ++colStart) {
            int patternRow = modelRow - rowStart;
            int patternCol = modelCol - colStart; 
            // TODO: Avoid this if statement
            if (patternRow >= 0 && patternRow < patternHeight && patternCol >= 0 && patternCol < patternWidth) {
                long patternIndex = patternCol + patternRow * patternWidth 
                                    + (colStart + rowStart * driftRangeHori) * patternSize;
                allPatterns[patternIndex] = threadModelValue;
            }
        }
}

__global__
void genPatrialPatterns(float *model, int modelWidth, int modelHeight,
                        float *partialPatterns, int patternWidth, int patternHeight,
                        int driftRangeVert, int driftRangeHori,
                        int driftIndexStart, int driftIndexEnd)
{
    int modelCol = threadIdx.x + blockIdx.x * blockDim.x;
    int modelRow = threadIdx.y + blockIdx.y + blockDim.y;
    int modelIndex = modelCol + modelRow * modelWidth;
    float threadModelValue = model[modelIndex];  // the model value owned by one thread.

    int patternSize = patternWidth * patternHeight;

    for (int driftIndex = driftIndexStart; driftIndex < driftIndexEnd; ++driftIndex) {
        int rowStart = driftIndex / driftRangeHori;
        int colStart = driftIndex % driftRangeHori;
        int patternRow = modelRow - rowStart;
        int patternCol = modelCol - colStart;
        if (patternRow >= 0 && patternRow < patternHeight && patternCol >= 0 && patternCol < patternWidth){
            long patternIndex = patternCol + patternRow * patternWidth 
                                + (driftIndex - driftIndexStart) * patternSize;
            partialPatterns[patternIndex] = threadModelValue;
        }
    }
}


__global__ 
void genSummedPatterns(float *model, int modelWidth, int modelHeight,
                       float *summedPatterns, int patternWidth, int patternHeight,
                       int driftRangeVert, int driftRangeHori)
{
    // each block takes care one pattern position (one drift)
    int driftIndex = blockIdx.x;
    int rowStart = driftIndex / driftRangeHori;
    int colStart = driftIndex % driftRangeHori;

    int patternRow = threadIdx.x;

    int cacheIndex = threadIdx.x;
    // Notice the blockDim.x could be smaller than MAX_BLOCK_SIZE.
    // Only [0, blockDim.x) elements in cache is used.
    __shared__ float cache[MAX_BLOCK_SIZE];

    float rowSumOfPattern = 0.0f;
    while (patternRow < patternHeight) {
        int modelColStartIndex = (patternRow + rowStart) * modelWidth + colStart;
        for (int col = 0; col < patternWidth; ++col) {
            rowSumOfPattern += model[col + modelColStartIndex];
        }
        patternRow += blockDim.x;
    }
    cache[cacheIndex] = rowSumOfPattern;
    __syncthreads();

    /* 
     * Reduction within block to get the sum of each whole pattern.
     * Notice this reduction will miss the last element (blockDim.x-1) 
     * if blockDim.x is an odd number. 
     */
    for (unsigned step = blockDim.x/2; step > 0; step >>= 1) {
        if (cacheIndex < step)
            cache[cacheIndex] += cache[cacheIndex + step];
        __syncthreads();
    }
    
    if (cacheIndex == 0)
        summedPatterns[driftIndex] = cache[0];
        if (blockDim.x % 2 != 0)
            summedPatterns[driftIndex] += cache[blockDim.x - 1];

}

__global__
void mergePatterns(float *modelCanvas, int modelWidth, int modelHeight, 
                    float *patterns, int patternWidth, int patternHeight,
                    int driftRangeVert, int driftRangeHori)
{
    int modelCol = threadIdx.x + blockIdx.x * blockDim.x;
    int modelRow = threadIdx.y + blockIdx.y * blockDim.y; 
    int modelIndex = modelCol + modelRow * modelWidth;
    int patternSize = patternWidth * patternHeight;

    float sum = 0.0f;
    for (int rowStart = 0; rowStart < driftRangeVert; ++rowStart)
        for (int colStart = 0; colStart < driftRangeHori; ++colStart) {
            int patternCol = modelCol - colStart;
            int patternRow = modelRow - rowStart;
            if (patternRow >= 0 && patternRow < patternHeight && patternCol >= 0 && patternCol < patternWidth){
                long patternIndex = patternCol + patternRow * patternWidth
                                    + (colStart * rowStart * driftRangeHori) * patternSize;
                sum += patterns[patternIndex];
            }
        }
    modelCanvas[modelIndex] = sum;
}



















