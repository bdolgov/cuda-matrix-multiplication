#include <cstdlib>
#include <cstdio>
#include <cassert>

typedef float float_t;

#define el(M, I, J) (*((float_t*)((char*)((M).ptr) + (I) * (M).pitch) + (J)))
#define w xsize
#define h ysize

#define eps 1e-4
#ifndef BLOCK_H
#define BLOCK_H 32
#endif
#ifndef BLOCK_W
#define BLOCK_W 32
#endif

cudaPitchedPtr allocHostMatrix(size_t h, size_t w)
{
#ifdef HAVE_SHARED
	h = BLOCK_W * ((h + BLOCK_W - 1) / BLOCK_W);
	w = BLOCK_W * ((w + BLOCK_W - 1) / BLOCK_W);
#endif
	void *mem = calloc(sizeof(float_t), w * h);
	if (!mem)
	{
		fprintf(stderr, "Out of memory on host!\n");
		exit(1);
	}
	return make_cudaPitchedPtr(mem, sizeof(float_t) * w, w, h);
}

cudaPitchedPtr allocDevMatrix(size_t h, size_t w)
{
#ifdef HAVE_SHARED
	h = BLOCK_W * ((h + BLOCK_W - 1) / BLOCK_W);
	w = BLOCK_W * ((w + BLOCK_W - 1) / BLOCK_W);
#endif
	void *mem;
	int rc; size_t pitch;
#ifndef HAVE_PITCH
	rc = cudaMalloc(&mem, sizeof(float_t) * w * h);
	pitch = sizeof(float_t) * w;
#else
	rc = cudaMallocPitch(&mem, &pitch, sizeof(float_t) * w, h);
#endif
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "Out of memory on device!");
		exit(1);
	}
	return make_cudaPitchedPtr(mem, pitch, w, h);
}

void matrixCpy(cudaPitchedPtr dst, cudaPitchedPtr src, cudaMemcpyKind kind)
{
	assert(src.h == dst.h && src.w == dst.w);
	int rc;
#ifndef HAVE_PITCH
	rc = cudaMemcpy(dst.ptr, src.ptr, src.h * src.w * sizeof(float_t), kind);
#else
	rc = cudaMemcpy2D(dst.ptr, dst.pitch, src.ptr, src.pitch, dst.w * sizeof(float_t), dst.h, kind);
#endif
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "matrixCpy error %d!", rc);
		exit(1);
	}
}

cudaPitchedPtr generateMatrix(size_t h, size_t w)
{
	cudaPitchedPtr ret = allocHostMatrix(h, w);
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			el(ret, i, j) = (float_t)rand() / RAND_MAX;
		}
	}
	return ret;
}

void hostMpy(cudaPitchedPtr C, cudaPitchedPtr A, cudaPitchedPtr B)
{
	assert(C.h == A.h && A.w == B.h && B.w == C.w);

	for (int i = 0; i < C.h; ++i)
	{
		for (int j = 0; j < C.w; ++j)
		{
			for (int k = 0; k < A.w; ++k)
			{
				el(C, i, j) += el(A, i, k) * el(B, k, j);
			}
		}
	}
}

__global__ void doDevMpy1(cudaPitchedPtr C, cudaPitchedPtr A, cudaPitchedPtr B)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= C.h || j >= C.w) return;
	float_t result = 0;
	for (int k = 0; k < A.w; ++k)
	{
		result += el(A, i, k) * el(B, k, j);
	}
	el(C, i, j) = result;
}

__global__ void doDevMpy2(cudaPitchedPtr C, cudaPitchedPtr A, cudaPitchedPtr B)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int thrI = threadIdx.y;
	int thrJ = threadIdx.x;
	__shared__ float_t Ac[BLOCK_W][BLOCK_W], Bc[BLOCK_W][BLOCK_W];
	float_t result = 0;
	for (int step = 0; step < A.w / BLOCK_W; ++step)
	{
		Ac[thrI][thrJ] = el(A, i, thrJ + BLOCK_W * step);
		Bc[thrI][thrJ] = el(B, thrI + BLOCK_W * step, j);
		__syncthreads();
		for (int k = 0; k < BLOCK_W; ++k)
		{
			result += Ac[thrI][k] * Bc[k][thrJ];
		}
		__syncthreads();
	}
	el(C, i, j) = result;
}

__global__ void doDevMpy3(cudaPitchedPtr C, cudaPitchedPtr A, cudaPitchedPtr B)
{
	int i = BLOCK_W * blockIdx.y + threadIdx.y;
	int j = BLOCK_W * blockIdx.x + threadIdx.x;
	int thrI = threadIdx.y;
	int thrJ = threadIdx.x;
	__shared__ float_t Ac[BLOCK_W][BLOCK_W], Bc[BLOCK_W][BLOCK_W];
	float_t result1 = 0, result2 = 0;
	for (int step = 0; step < A.w / BLOCK_W; ++step)
	{
		Ac[thrI][thrJ] = el(A, i, thrJ + BLOCK_W * step);
		Bc[thrI][thrJ] = el(B, thrI + BLOCK_W * step, j);
		Ac[thrI + BLOCK_H][thrJ] = el(A, i + BLOCK_H, thrJ + BLOCK_W * step);
		Bc[thrI + BLOCK_H][thrJ] = el(B, thrI + BLOCK_W * step + BLOCK_H, j);
		__syncthreads();
		for (int k = 0; k < BLOCK_W; ++k)
		{
			result1 += Ac[thrI][k] * Bc[k][thrJ];
			result2 += Ac[thrI + BLOCK_H][k] * Bc[k][thrJ];
		}
		__syncthreads();
	}
	el(C, i, j) = result1;
	el(C, i + BLOCK_H, j) = result2;
}

void devMpy(cudaPitchedPtr C, cudaPitchedPtr A, cudaPitchedPtr B)
{
	assert(C.h == A.h && A.w == B.h && B.w == C.w);
	dim3 threads(BLOCK_W, BLOCK_H);
	dim3 grid((C.w + BLOCK_W - 1) / BLOCK_W, (C.h + BLOCK_H - 1) / BLOCK_H);
#if defined(HAVE_SHARED) && BLOCK_H == BLOCK_W
	assert(BLOCK_W == BLOCK_H && C.h % BLOCK_W == 0 && C.w % BLOCK_W == 0 && A.w % BLOCK_W == 0);
	doDevMpy2<<<grid, threads>>>(C, A, B);
#elif defined(HAVE_SHARED) && BLOCK_H * 2 == BLOCK_W
	grid.y /= 2;
	assert(BLOCK_W == 2 * BLOCK_H && C.h % BLOCK_W == 0 && C.w % BLOCK_W == 0 && A.w % BLOCK_W == 0);
	doDevMpy3<<<grid, threads>>>(C, A, B);
#else
	doDevMpy1<<<grid, threads>>>(C, A, B);
#endif
}

void matrixPrint(const char* name, cudaPitchedPtr A)
{
	printf("Matrix %s:\n", name);
	int ilimit = A.h, jlimit = A.w;
	if (ilimit > 30) ilimit = 30;
	if (jlimit > 15) jlimit = 15;
	int i, j;
	for (i = 0; i < ilimit; ++i)
	{
		for (j = 0; j < jlimit; ++j)
		{
			printf("%f ", el(A, i, j));
		}
		if (j != A.w)
		{
			printf("...");
		}
		printf("\n");
	}
	if (i != A.h)
	{
		printf("...\n");
	}
	printf("-----\n");
}

bool hostEquals(cudaPitchedPtr A, cudaPitchedPtr B)
{
	if (A.h != B.h || A.w != B.w) return false;
	for (int i = 0; i < A.h; ++i)
	{
		for (int j = 0; j < A.w; ++j)
		{
			if (fabs(el(A, i, j) - el(B, i, j)) > eps)
			{
				return false;
			}
		}
	}
	return true;
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "Usage: %s n m k [--check]\n", argv[0]);
		return 1;
	}

	int n = atoi(argv[1]), m = atoi(argv[2]), k = atoi(argv[3]);
	bool check = argc >= 5 && !strcmp(argv[4], "--check");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float time;

	cudaPitchedPtr A = generateMatrix(n, m), B = generateMatrix(m, k), C = allocHostMatrix(n, k);
	
	cudaPitchedPtr Ad = allocDevMatrix(n, m), Bd = allocDevMatrix(m, k), Cd = allocDevMatrix(n, k);
	matrixCpy(Ad, A, cudaMemcpyHostToDevice);
	matrixCpy(Bd, B, cudaMemcpyHostToDevice);

	
	cudaEventRecord(start, 0);
	
	devMpy(Cd, Ad, Bd);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Elapsed time: %f\n", time);

	matrixCpy(C, Cd, cudaMemcpyDeviceToHost);

	cudaFree(Ad.ptr);
	cudaFree(Bd.ptr);
	cudaFree(Cd.ptr);

	if (check)
	{
		cudaPitchedPtr C1 = allocHostMatrix(n, k);
		hostMpy(C1, A, B);
		if (getenv("DUMP_RESULTS"))
		{
			matrixPrint("A", A);
			matrixPrint("B", B);
			matrixPrint("C", C);
			matrixPrint("C1", C1);
		}
		printf("Check %s!\n", hostEquals(C, C1) ? "OK" : "Not OK");
		free(C1.ptr); 
	}

	free(A.ptr);
	free(B.ptr);
	free(C.ptr);
}
