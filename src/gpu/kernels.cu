/*plane cuda kernels*/

__global__
void pattern_gen(float *model, int row, int col, 
                 float* pattern, int row_patter, int col_pattern, 
                 int row_start, int col_start
                ) {
    int idx = threadIdx.x;  // map to column index of the pattern
    int idy = blockIdx.x;   // map to row index of the pattern

    // if the blockDim.x >= col_pattern, this while loop will be executed 
    // once only (i.e. each thread takes care one pixel).
    while (idx < col_pattern) {
        int indx_M = (idy + row_start)*col + idx + col_start;
        int indx_P = idy*col_frame + idx;
        pattern[indx_P] = model[indx_M];
        idx += blockDim.x;    
    }
}
