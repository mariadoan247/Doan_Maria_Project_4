
// To generate hash value
__device__
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions, unsigned int MAX);


/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
__global__
void hash_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Generate hash values
    if (index < array_size) {
        hash_array[index] = generate_hash(nonce_array[index], index, transactions, n_transactions, mod);
    }

} // End Hash Kernel //



/* Generate Hash ----------------------------------------- //
*   Generates a hash value from a nonce and transaction list.
*/
__device__
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions, unsigned int MAX) {

    unsigned int hash = (nonce + transactions[0] * (index + 1)) % MAX;
    for (int j = 1; j < n_transactions; j++) {
        hash = (hash + transactions[j] * (index + 1)) % MAX;
    }
    return hash;

} // End Generate Hash ---------- //