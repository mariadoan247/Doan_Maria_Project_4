#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cuda.h>

#include "kernel.cu"

// block size is 32 because we are working with a matrix and 
// 32*32=1024 which is the max block size possible
#define BLOCK_SIZE 32

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

// functions used
void err_check(cudaError_t ret, char* msg, int exit_code);


/* Main ------------------ //
*   This is the main program.
*/
int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_CUDA n_row n_col mat_input.csv mat_output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[3]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* filterMatrix_h = (int*)malloc(5 * 5 * sizeof(int));
    int* inputMatrix_h  = (int*)malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*)malloc(n_row * n_col * sizeof(int));

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }


    // Filling filter
	// 1 0 0 0 1 
	// 0 1 0 1 0 
	// 0 0 1 0 0 
	// 0 1 0 1 0 
	// 1 0 0 0 1 
    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            filterMatrix_h[i*5+j]=0;

    filterMatrix_h[0*5+0] = 1;
    filterMatrix_h[1*5+1] = 1;
    filterMatrix_h[2*5+2] = 1;
    filterMatrix_h[3*5+3] = 1;
    filterMatrix_h[4*5+4] = 1;
    
    filterMatrix_h[4*5+0] = 1;
    filterMatrix_h[3*5+1] = 1;
    filterMatrix_h[1*5+3] = 1;
    filterMatrix_h[0*5+4] = 1;

    fclose(inputFile1); 

    // Set cuda error return
    cudaError_t cuda_ret;

    // To use with kernels
    dim3 dimGrid(ceil(n_col/(float)BLOCK_SIZE), ceil(n_row/(float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // --- Step 1&2: Transfer the input image (the A matrix) to the device memory -------- //
    // ------------- Transfer the convolution filter (the K matrix) to the device memory---//
    struct timespec start, end;    
    clock_gettime(CLOCK_REALTIME, &start);

    // Allocate the input image device memory
    int* inputMatrix_d;
    cuda_ret = cudaMalloc((void**)&inputMatrix_d, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate input image to device memory!", 1);
    cuda_ret = cudaMemcpy(inputMatrix_d, inputMatrix_h, n_row * n_col * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to copy input image from host memory to device memory!", 3);

    // Allocate the input image device memory
    int* outputMatrix_d;
    cuda_ret = cudaMalloc((void**)&outputMatrix_d, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate output matrix to device memory!", 1);

    // Allocate the convolution filter device memory
    int* filterMatrix_d;
    cuda_ret = cudaMalloc((void**)&filterMatrix_d, 5 * 5 * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate convolution filter to device memory!", 1);
    cuda_ret = cudaMemcpy(filterMatrix_d, filterMatrix_h, 5 * 5 * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to copy convolution filter from host memory to device memory!", 3);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent1 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    // --- Step 3: Launch the convolution kernel to compute the filter map --------------- //
    // ----------- (the B matrix) by applying the convolution to every ------------------- //
    // ----------- pixel in the input image ---------------------------------------------- //
    clock_gettime(CLOCK_REALTIME, &start);

	// Performing convolution
	// Take a look at slides about the blurring example
    // Launch the kernel
    kernel <<< dimGrid, dimBlock >>> (
        inputMatrix_d,  // input image matrix
        outputMatrix_d, // put the resulting image here
        filterMatrix_d, // convolution filter matrix
        n_row,          // number of rows
        n_col          // number of columns
        );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch kernel!", 2);

    // Free memory
    cudaFree(inputMatrix_d);
    cudaFree(filterMatrix_d);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent2 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    // --- Step 4: Transfer the filter map (the B matrix) from the device ---------------- //
    // ----------- memory to the system memory. ------------------------------------------ //  
    clock_gettime(CLOCK_REALTIME, &start);

    // Transfer the filter map from the device memory to the system memory
    cuda_ret = cudaMemcpy(outputMatrix_h, outputMatrix_d, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to transfer filter map from device to host memory!", 3);

    // Free memory
    cudaFree(outputMatrix_d);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent3 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

	// Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time
    fprintf(timeFile, "%.20f\n%.20f\n%.20f", time_spent1, time_spent2, time_spent3);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    free(inputMatrix_h);
    free(outputMatrix_h);
    free(filterMatrix_h);

    return 0;
}


/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //