
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vgl.h>

#include <stdio.h>

#include <cuda_gl_interop.h>

const int DIM = 512;

struct cuComplex{
	float r;
	float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}

	__device__ float magnitude2(void)
	{
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex & a)
	{
		return cuComplex((r * a.r - i * a.i), (i * a.r + r * a.i));
	}

	__device__ cuComplex operator+ (const cuComplex & a)
	{
		return cuComplex((r + a.r), (i + a.i));
	}
};

__device__ int julia(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void gpuKernel(uchar4 * ptr)
{
	int threadx = threadIdx.x + blockIdx.x * blockDim.x;
	int thready = threadIdx.y + blockIdx.y * blockDim.y;

	int thread = threadx + thready * blockDim.x * gridDim.x;

	int juliaValue = julia(threadx, thready);
	ptr[thread].x = 255 * juliaValue;
	ptr[thread].y = 0;
	ptr[thread].z = 0;
	ptr[thread].w = 255;
}

cudaError_t JuliaWithCuda();

enum Buffer_IDs{PIXEL_UNPACK_BUFFER, NUMBUFFERS};

GLuint Buffers[NUMBUFFERS];
cudaGraphicsResource * resource;

void DisplayFunc()
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

void gl_init(int argc, char ** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Julia_Sample");

	if (glewInit())
	{
		printf("GLEW INITLIZATION FAILED! EXIT...\n");
		exit(0);
	}

	glGenBuffers(NUMBUFFERS, Buffers);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffers[PIXEL_UNPACK_BUFFER]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	glutDisplayFunc(DisplayFunc);
}

int main(int argc, char ** argv)
{
	gl_init(argc, argv);

	cudaError_t cudaStatus = JuliaWithCuda();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "JuliaWithCuda failed!\n");
		return 1;
	}

	glutMainLoop();
	
	return 0;
}

cudaError_t JuliaWithCuda()
{
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaError_t cudaStatus = cudaChooseDevice(&dev, &prop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaChooseDevice failed! Do you have a CUDA-capable GPU installed?\n");
		return cudaStatus;
	}

	cudaStatus = cudaSetDevice(dev);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!\n");
		return cudaStatus;
	}

	cudaStatus = cudaGraphicsGLRegisterBuffer(&resource, Buffers[PIXEL_UNPACK_BUFFER], cudaGraphicsMapFlagsNone);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!\n");
		return cudaStatus;
	}

	uchar4 * devPtr;
	size_t size;
	cudaStatus = cudaGraphicsMapResources(1, &resource, NULL);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsMapResources failed!\n");
		return cudaStatus;
	}

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, resource);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!\n");
		return cudaStatus;
	}

	dim3 grids(DIM / 16, DIM / 16);
	dim3 blocks(16, 16);
	gpuKernel << <grids, blocks >> >(devPtr);

	cudaStatus = cudaGraphicsUnmapResources(1, &resource, NULL);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsUnmapResources failed!\n");
		return cudaStatus;
	}
}