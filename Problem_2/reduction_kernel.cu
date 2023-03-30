
#define BLOCK_SIZE 1024

/* Reduction Kernel --------------------------------------
*       Finds the local nonce values with the minimum hash values
*/
__global__
void reduction_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* min_hash_array, unsigned int* min_nonce_array, unsigned int MAX) {

    // Calculate thread index
    unsigned int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int hash_reduction[BLOCK_SIZE];
    __shared__ unsigned int nonce_reduction[BLOCK_SIZE];
    // Find local mins
    if (index < array_size) {
        hash_reduction[threadIdx.x] = hash_array[index];
        nonce_reduction[threadIdx.x] = nonce_array[index];
    } else {
        hash_reduction[threadIdx.x] = MAX;
        nonce_reduction[threadIdx.x] = MAX;
    }
    if ((index + BLOCK_SIZE) < array_size && hash_reduction[threadIdx.x] > hash_array[index + BLOCK_SIZE]) {
        hash_reduction[threadIdx.x] = hash_array[index + BLOCK_SIZE];
        nonce_reduction[threadIdx.x] = nonce_array[index + BLOCK_SIZE];
    }
    for (int stride = BLOCK_SIZE/2; stride >= 1; stride = stride/2) {
        __syncthreads();
        if (threadIdx.x < stride && hash_reduction[threadIdx.x] > hash_reduction[threadIdx.x + stride]) {
            hash_reduction[threadIdx.x] = hash_reduction[threadIdx.x + stride];
            nonce_reduction[threadIdx.x] = nonce_reduction[threadIdx.x + stride];
        }
    }

    if(threadIdx.x == 0) {
        min_hash_array[blockIdx.x] = hash_reduction[0];
        min_nonce_array[blockIdx.x] = nonce_reduction[0];
    }

} // End Reduction Kernel //
