
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <memory>
#include <time.h>

#include <glew.h>
#include <glut.h>

#include <cuda_gl_interop.h>

const int DIM = 512;

GLuint bufferObj;
cudaGraphicsResource * resource;

__global__ void kernel(uchar4 * ptr, int ticks)
{
	int threadx = threadIdx.x + blockIdx.x * blockDim.x;
	int thready = threadIdx.y + blockIdx.y * blockDim.y;

	int thread = threadx + thready * blockDim.x * gridDim.x;

	float fx = threadx - DIM / 2;
	float fy = thready - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.f - ticks / 7.0f) / (d / 10.f + 1.0f));

	ptr[thread].x = grey;
	ptr[thread].y = grey;
	ptr[thread].z = grey;
	ptr[thread].w = 255;
}

cudaError_t choose_Device()
{
	cudaDeviceProp prop;
	int devID;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaError_t cudaStatus = cudaChooseDevice(&devID, &prop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaChooseDevice failed!\n");
		return cudaStatus;
	}

	cudaStatus = cudaSetDevice(devID);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!\n");
		return cudaStatus;
	}
	return cudaStatus;
}

void displayFunc()
{
	uchar4 * devPtr;
	size_t size;
	cudaError_t cudaStatus = cudaGraphicsMapResources(1, &resource, NULL);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsMapResources failed!\n");
	}

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, resource);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!\n");
	}

	clock_t ticks = clock();
	dim3 grid(DIM / 16, DIM / 16);
	dim3 block(16, 16);
	kernel << <grid, block >> >(devPtr, ticks);

	cudaGraphicsUnmapResources(1, &resource, NULL);

	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glutSwapBuffers();
	glutPostRedisplay();
}

void init_OpenGL(int argc, char ** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("grey_animation");

	if (glewInit())
	{
		fprintf(stderr, "Glew Initlization failed! Exit...\n");
		exit(0);
	}

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	glutDisplayFunc(displayFunc);
}

int main(int argc, char ** argv)
{
	cudaError_t cudaStatus = choose_Device();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "choose_Device failed! Exit...\n");
		exit(0);
	}

	init_OpenGL(argc, argv);

	cudaStatus = cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!\n");
		exit(0);
	}

	glutMainLoop();

	return 0;
}
