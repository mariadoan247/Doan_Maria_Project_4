
#define BLUR_SIZE 2

/* Convolution Kernel --------------------------------------
*       Blurs an image
*/
__global__
void kernel(int* inputMatrix, int* outputMatrix, int* filterMatrix, int n_row, int n_col) {

    // Initialize row (i) and col (j) index
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Performing convolution
    if (j < n_col && i < n_row) {
        int sum_val = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
        {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
            {
                int curRow = i + blurRow;
                int curCol = j + blurCol;

                int i_row = blurRow + BLUR_SIZE;
                int i_col = blurCol + BLUR_SIZE;

                if( curRow > -1 && curRow < n_row && curCol > -1 && curCol < n_col)
                {
                    sum_val += inputMatrix[curRow*n_col + curCol]*filterMatrix[i_row*5 + i_col]; 
                }
            }
        }

        outputMatrix[i*n_col+j] = sum_val;
    }

} // End Hash Kernel //
