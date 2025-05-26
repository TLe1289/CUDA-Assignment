#include <assert.h>
#include <stdio.h>
#include <math.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}


void output_display(int* h_prefixSum, int* h_histogram, int n_partitions) {
    printf("Partition Information:\n");
    for (int i = 0; i < n_partitions; ++i) {
        printf("partition %d: offset %d, number of keys %d\n", i, h_prefixSum[i], h_histogram[i]);
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

//define the histogram kernel here
__global__ void histogram(int* d_r, int *d_histogram, int* d_bucket, int r_size, int n_partitions, int start, int nbits)
{
    extern __shared__ int shared_histogram[]; // Shared memory for local histogram
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory histogram
    for (int i = threadIdx.x; i < n_partitions; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    __syncthreads();

    // Populate shared histogram
    for (int idx = threadId; idx < r_size; idx += stride) {
        uint bucket = bfe(d_r[idx], start, nbits);
        if (bucket < n_partitions) {
            atomicAdd(&shared_histogram[bucket], 1); // Local shared memory update
            d_bucket[idx] = bucket;                 // Store bucket mapping
        }
    }
    __syncthreads();

    // Merge shared histogram into global histogram
    for (int i = threadIdx.x; i < n_partitions; i += blockDim.x) {
        atomicAdd(&d_histogram[i], shared_histogram[i]);
    }


}

//define the prefix scan kernel here
__global__ void prefixScan(int*d_histogram, int*d_prefixSum, int n_partitions)
{   

    extern __shared__ int shared_data[]; // Shared memory for scan
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Load data into shared memory
    if (globalIdx < n_partitions) {
        shared_data[localIdx] = d_histogram[globalIdx];
    } else {
        shared_data[localIdx] = 0; // Avoid out-of-bounds issues
    }
    __syncthreads();

    // Upsweep phase: Perform inclusive scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (localIdx >= stride) {
            temp = shared_data[localIdx - stride];
        }
        __syncthreads();
        shared_data[localIdx] += temp;
        __syncthreads();
    }

    // Convert to exclusive scan
    if (localIdx < n_partitions) {
        int temp = shared_data[localIdx];
        if (localIdx == 0) {
            shared_data[localIdx] = 0; // First element for exclusive scan
        } else {
            shared_data[localIdx] = shared_data[localIdx - 1];
        }
        d_prefixSum[globalIdx] = shared_data[localIdx]; // Write result to global memory
    }
    __syncthreads();

}

//define the reorder kernel here
__global__ void Reorder(int* d_r, int *d_prefixSum, int* d_output, int* d_bucket,  int n_partitions, int r_size, int start, int nbits)
{
   /* int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < r_size) {
        uint h = bfe(d_r[threadId], start, nbits);
        int offset = atomicAdd(&d_prefixSum[h], 1);
        d_output[offset] = d_r[threadId];
    } */

	extern __shared__ int shared_prefixSum[];
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	for (int i = threadIdx.x; i < n_partitions; i += blockDim.x) {
        shared_prefixSum[i] = 0;
    	}
    	__syncthreads();

	if (i >= r_size) return; 			//

	if (i < r_size){
		uint bucket = d_bucket[i];
		//int index =  d_prefixSum[bucket];
		int index = d_prefixSum[bucket] + shared_prefixSum[bucket];
		d_output[index] = d_r[i];
		atomicAdd(&shared_prefixSum[bucket],1);
	}
	__syncthreads();


	for (int stride = n_partitions/2; stride > 0; stride /=2) {
		if(threadIdx.x < stride && (threadIdx.x + stride) < n_partitions){
			shared_prefixSum[threadIdx.x] = shared_prefixSum[threadIdx.x + stride];
		}
    	}
	__syncthreads();


	/*High Thread Divergence -  ensure only one thread is active*/
	//if (threadIdx.x == 0) {
    	for (int i = threadIdx.x; i < threadIdx.x; i += blockDim.x) {
        	atomicAdd(&d_prefixSum[i], shared_prefixSum[i]);
    	}
	//}


}

int main(int argc, char const *argv[])
{
    if (argc < 3) {
        printf("Incorrect number of inputs");
        return -1;
    }
    int rSize = atoi(argv[1]);
    int npartitions = atoi(argv[2]);
    int* r_h;
    int* d_r;
    int* d_histogram;
    int* d_prefixSum;
    int* d_output;
    int* d_bucket;

    cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory 
    cudaMalloc((void**)&d_r, sizeof(int)*rSize);
    cudaMalloc((void **)&d_histogram, sizeof(int) * npartitions);
    cudaMalloc((void **)&d_prefixSum, sizeof(int) * npartitions);
    cudaMalloc((void **)&d_output, sizeof(int) * rSize);
    cudaMalloc((void **)&d_bucket, sizeof(int) *rSize);

    dataGenerator(r_h, rSize, 0, 1);

    cudaMemcpy(d_r, r_h, sizeof(int) * rSize,cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, sizeof(int) * npartitions);
    cudaMemset(d_prefixSum, 0, sizeof(int) * npartitions);
    cudaMemset(d_output, 0, sizeof(int) * rSize);
    cudaMemset(d_bucket, 0, sizeof(int) * rSize);

    int blocksize = npartitions;
    int gridsize = (rSize + blocksize - 1) / blocksize;
    //int sharedmemsize = blocksize * sizeof(int);
    int sharedmemsize = npartitions * sizeof(int); 
    int starting = 0;
    int nbits  = (int)floor(log((double)npartitions) / log(2.0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    /* your code */

   // Kernel 1
    histogram<<<gridsize, blocksize,sharedmemsize>>>(d_r, d_histogram, d_bucket, rSize, npartitions,  starting, nbits);
    cudaDeviceSynchronize(); // Ensure kernel 1 completes

// Kernel 2
    prefixScan<<<gridsize, blocksize,sharedmemsize>>>(d_histogram, d_prefixSum, npartitions);
    cudaDeviceSynchronize(); // Ensure kernel 2 completes

// Kernel 3
    Reorder<<<gridsize, blocksize, sharedmemsize>>> (d_r, d_prefixSum , d_output, d_bucket, npartitions, rSize, starting, nbits);
    cudaDeviceSynchronize(); // Ensure kernel 3 completes

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    int* h_prefixSum = (int*)malloc(sizeof(int) * npartitions);
    int* h_histogram = (int*)malloc(sizeof(int) * npartitions);

    cudaMemcpy(h_prefixSum, d_prefixSum, sizeof(int) * npartitions, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histogram, d_histogram, sizeof(int) * npartitions, cudaMemcpyDeviceToHost);

    output_display(h_prefixSum, h_histogram, npartitions);

    free(h_prefixSum);
    free(h_histogram);


    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total time to run 3 kernels: %0.5f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaFreeHost(r_h);
    cudaFree(d_r);
    cudaFree(d_histogram);
    cudaFree(d_prefixSum);
    cudaFree(d_output);
    cudaFree(d_bucket);
    return 0;
}
