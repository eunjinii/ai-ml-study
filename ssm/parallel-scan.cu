#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>

#define SBLOCK_DIM 256
#define BLOCK_DIM 128 // SBLOCK_DIM/2

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code == cudaSuccess) return;
    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

double timeStamp() {
    struct timeval tv; 
    gettimeofday(&tv, NULL);
    return tv.tv_usec / 1000.0 + tv.tv_sec;
}

void sumCPU(float *h_X, float *h_Y, int M) {
    for (int i = 0; i < M; ++i) {
        h_Y[i] = h_X[i] + h_Y[i];
    }
}

void displayResults(float *h_X, float *h_Y, float *h_Y_CPU, int M) {
    for (int i = 0; i < M; ++i) {
        printf("[Index %d] ORG - %.4f / HOST - %.4f / DEVICE - %.4f\n", i, h_X[i], h_Y_CPU[i], h_Y[i]);
    }
}

void validateResults(float *h_Y, float *h_Y_CPU, int M) {
    int incorrectCount = 0;
    for (int i = 0; i < M; ++i) {
        if (abs(h_Y_CPU[i] - h_Y[i]) > 1e-5) {
            incorrectCount++;
        }
    }
    
    if (incorrectCount == 0) {
        printf("Validation Passed!\n");
    } else {
        printf("Validation Failed: %d elements incorrect.\n", incorrectCount);
    }
}

__global__ void brentKungScan(float* d_X, float* d_Y, int N) {
    
    __shared__ float sBlock[SBLOCK_DIM]; // BLOCK_DIM * 2

    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x; // Block reads two elements per thread

    if (i < N) {
        sBlock[threadIdx.x] = d_X[i];
    }
    if (i + blockDim.x < N) {
        sBlock[threadIdx.x + blockDim.x] = d_X[i + blockDim.x];
    }

    for (unsigned int offset = 1; offset <= BLOCK_DIM; offset *= 2) {
        int curr = offset * 2 * (threadIdx.x + 1) - 1;
        if (curr < SBLOCK_DIM) {
            sBlock[curr] += sBlock[curr - offset];
        }
        __syncthreads();
    }

    for (unsigned int offset = BLOCK_DIM/2; offset >= 1; offset /= 2) {
        int curr = offset * 2 * (threadIdx.x + 1) - 1;
        if (curr + offset < SBLOCK_DIM) {
            sBlock[curr + offset] += sBlock[curr];
        }
        __syncthreads();
    }

    __syncthreads();

    if (i < N) {
        d_Y[i] = sBlock[threadIdx.x];
    }
    if (i + blockDim.x < N) {
        d_Y[i + blockDim.x] = sBlock[threadIdx.x + blockDim.x];
    }
}

int main(int argc, char *argv[]) {
    // Set matrix size
    int N = atoi(argv[1]);
    if (N <= 0) return 0;
    size_t bytes = N * sizeof(float);

	float *h_X, *h_Y;
	float *d_X, *d_Y;
    float *h_Y_CPU = (float *)malloc(N * sizeof(float)); // for validation

	// Allocate host memory
    gpuErrchk(cudaHostAlloc((void **)&h_X, bytes, cudaHostAllocMapped));
    gpuErrchk(cudaHostAlloc((void **)&h_Y, bytes, cudaHostAllocMapped));
    
	// initialize data
	for (int i = 0; i < N; ++i) {
        h_X[i] = (float)(rand() % 10 + 1);
	}

    // allocate device memory
    gpuErrchk(cudaMalloc(&d_X, bytes));
    gpuErrchk(cudaMalloc(&d_Y, bytes));
    

	// copy host data to device
	gpuErrchk(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

	dim3 blockDim(BLOCK_DIM, 1);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, 1);
    
    printf("blockDim: %d x %d\n", blockDim.x, blockDim.y);
    printf("gridDim: %d x %d\n", gridDim.x, gridDim.y);
    
    double start = timeStamp();
	brentKungScan<<<gridDim, blockDim>>>(d_X, d_Y, N);
    double finish = timeStamp();
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

	// copy result back to host
	gpuErrchk(cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost));

    printf("Throughput: %.4f GB/s, Time: %.4f ms\n\n", ((double)bytes / (finish-start))*1.0e-9, (finish - start)*1000);

    sumCPU(h_Y, h_Y_CPU, N);
    // displayResults(h_X, h_Y, h_Y_CPU, N);
    validateResults(h_Y, h_Y_CPU, N);

	// clean up data
    gpuErrchk(cudaFreeHost(h_X));
    gpuErrchk(cudaFreeHost(h_Y));
    gpuErrchk(cudaFree(d_X)); 
    gpuErrchk(cudaFree(d_Y));
    gpuErrchk(cudaDeviceReset());
    free(h_Y_CPU);

	return 0;
}