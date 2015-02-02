/*
 Copyright (c) 2015, Burak Sarac, burak@linux.com
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
 following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 following disclaimer in the documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NeuralNetwork.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
using namespace std;
#define SUCCESS 0
#define FAILURE 1
#define E (2.7182818284590452353602874713526624977572470937L )

int NeuralNetwork::convertToString(const char *filename, std::string& s) {
	size_t size;
	char* str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open()) {
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t) f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str) {
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return FAILURE;
}
NeuralNetwork::NeuralNetwork(double* aList, double* bList, int lCount, int* nCounts, int nOfLabels, int yWeight, int xCSize) {
	xColumnSize = xCSize;
	xList = aList;
	yList = bList;
	layerCount = lCount;
	neuronCounts = nCounts;
	numberOfLabels = nOfLabels;
	ySize = yWeight;
	ySizeDouble = ySize;
	dDim1 = layerCount - 1;
	dMatrixDimensions = (int*) malloc(sizeof(int) * dDim1);
	dLayerCache = new int[layerCount];
	nLayerCache = new int[layerCount + 1];
	eLayerCache = new int[layerCount];
	dMatrixSize = 0;
	xDim2 = neuronCounts[0];
	yDim2 = numberOfLabels;
	nLayerCache[0] = 0;
	eLayerCache[0] = 0;
	dLayerCache[0] = 0;
	neuronSize = 0;
	errorSize = 0;
	deltaSize = 0;
	costs = (double*) malloc(sizeof(double) * ySize);
	for (int i = 0; i < layerCount; ++i) {

		neuronSize += i == layerCount - 1 ? neuronCounts[i] : neuronCounts[i] + 1;
		nLayerCache[i + 1] = neuronSize;

		if (i < layerCount - 1) {

			errorSize += i == layerCount - 2 ? neuronCounts[i + 1] : neuronCounts[i + 1] + 1;
			eLayerCache[i + 1] = errorSize;
			dMatrixDimensions[i] = neuronCounts[i] + 1;

			deltaSize += (neuronCounts[i + 1] * dMatrixDimensions[i]);
			dLayerCache[i + 1] = deltaSize;
		}
	}

	mNeuronSize = sizeof(double) * neuronSize;
	mErrorSize = sizeof(double) * errorSize;
	mDeltaSize = sizeof(double) * deltaSize;
	deltas = (double *) malloc(mDeltaSize);
	tSums = (double *) malloc(mDeltaSize);

	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS) {
		cout << "Error: Getting platforms!" << endl;
	}

	/*For clarity, choose the first available platform. */
	if (numPlatforms > 0) {
		cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}

	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	cl_uint numDevices = 0;
	cl_device_id *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (numDevices == 0)	//no GPU available.
			{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	} else {
		cout << "GPU device available." << endl;
		devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}

	/*Step 3: Create context.*/
	context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

	/*Step 4: Creating command queue associate with the context.*/
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	/*Step 5: Create program object */
	const char *filename = "neuralNetwork.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	//printf("%s", sourceStr.c_str());
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	p_neuralNetworkByGpu = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

	/*Step 5: Create program object */
	const char *filename1 = "reduceDeltas.cl";
	string sourceStr1;
	status = convertToString(filename1, sourceStr1);
	//printf("%s", sourceStr.c_str());
	const char *source1 = sourceStr1.c_str();
	size_t sourceSize1[] = { strlen(source1) };
	p_reduceDeltasByGpu = clCreateProgramWithSource(context, 1, &source1, sourceSize1, NULL);

	/*Step 6: Build program. */
	status = clBuildProgram(p_neuralNetworkByGpu, 1, devices, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		cout << "Error: couldnt build program!" << status << endl;
	}
	/*Step 6: Build program. */
	status = clBuildProgram(p_reduceDeltasByGpu, 1, devices, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		cout << "Error: couldnt build program!" << status << endl;
	}
	k_neuralNetworkByGpu = clCreateKernel(p_neuralNetworkByGpu, "calculate", NULL);
	k_reduceDeltasByGpu = clCreateKernel(p_reduceDeltasByGpu, "reduce", NULL);

	d = (double*) malloc(sizeof(double) * deltaSize * ySize);
	d_xList = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * ySize * xColumnSize, aList, NULL);
	d_yList = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * ySize * numberOfLabels, yList, NULL);
	d_costs = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * ySize, costs, NULL);

	d_deltas = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * deltaSize * ySize, d, NULL);
	d_neuronCounts = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * layerCount, neuronCounts, NULL);
	d_dlayerCache = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * layerCount, dLayerCache, NULL);
	d_elayerCache = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * layerCount, eLayerCache, NULL);
	d_nlayerCache = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * layerCount, nLayerCache, NULL);
	d_matrixInfo = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * layerCount, dMatrixDimensions, NULL);
	d_delta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * deltaSize, deltas, NULL);
	d_tSums = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * deltaSize, tSums, NULL);
	double e = E;
	clSetKernelArg(k_neuralNetworkByGpu, 0, sizeof(cl_mem), (void *) &d_deltas);
	clSetKernelArg(k_neuralNetworkByGpu, 1, sizeof(cl_mem), (void *) &d_xList);
	clSetKernelArg(k_neuralNetworkByGpu, 2, sizeof(cl_int), (void *) &ySize);
	clSetKernelArg(k_neuralNetworkByGpu, 3, sizeof(cl_mem), (void *) &d_yList);
	clSetKernelArg(k_neuralNetworkByGpu, 4, sizeof(cl_int), (void *) &layerCount);
	clSetKernelArg(k_neuralNetworkByGpu, 5, sizeof(cl_mem), (void *) &d_neuronCounts);

	clSetKernelArg(k_neuralNetworkByGpu, 8, sizeof(cl_int), (void *) &neuronSize);
	clSetKernelArg(k_neuralNetworkByGpu, 9, sizeof(cl_int), (void *) &errorSize);
	clSetKernelArg(k_neuralNetworkByGpu, 10, sizeof(cl_int), (void *) &deltaSize);
	clSetKernelArg(k_neuralNetworkByGpu, 11, sizeof(cl_int), (void *) &xColumnSize);
	clSetKernelArg(k_neuralNetworkByGpu, 12, sizeof(cl_mem), (void *) &d_dlayerCache);
	clSetKernelArg(k_neuralNetworkByGpu, 13, sizeof(cl_mem), (void *) &d_matrixInfo);
	clSetKernelArg(k_neuralNetworkByGpu, 14, sizeof(cl_mem), (void *) &d_nlayerCache);
	clSetKernelArg(k_neuralNetworkByGpu, 15, sizeof(cl_mem), (void *) &d_elayerCache);
	clSetKernelArg(k_neuralNetworkByGpu, 16, sizeof(cl_int), (void *) &numberOfLabels);
	clSetKernelArg(k_neuralNetworkByGpu, 17, sizeof(cl_mem), (void *) &d_costs);
	clSetKernelArg(k_neuralNetworkByGpu, 18, sizeof(cl_double), (void *) &e);
	clSetKernelArg(k_reduceDeltasByGpu, 0, sizeof(cl_mem), (void *) &d_deltas);
	clSetKernelArg(k_reduceDeltasByGpu, 1, sizeof(cl_int), (void *) &ySize);
	clSetKernelArg(k_reduceDeltasByGpu, 2, sizeof(cl_mem), (void *) &d_delta);
	clSetKernelArg(k_reduceDeltasByGpu, 3, sizeof(cl_mem), (void *) &d_neuronCounts);
	clSetKernelArg(k_reduceDeltasByGpu, 4, sizeof(cl_mem), (void *) &d_dlayerCache);
	clSetKernelArg(k_reduceDeltasByGpu, 5, sizeof(cl_mem), (void *) &d_matrixInfo);
	clSetKernelArg(k_reduceDeltasByGpu, 6, sizeof(cl_int), (void *) &layerCount);
	clSetKernelArg(k_reduceDeltasByGpu, 9, sizeof(cl_mem), (void *) &d_tSums);

	double yMultiplier = 1 / ySizeDouble;

	clSetKernelArg(k_reduceDeltasByGpu, 10, sizeof(cl_int), (void *) &deltaSize);
	clSetKernelArg(k_reduceDeltasByGpu, 11, sizeof(cl_double), (void *) &yMultiplier);

}
void NeuralNetwork::destroy() {
	clFinish(commandQueue);
	clFlush(commandQueue);
	clReleaseMemObject(d_xList);
	clReleaseMemObject(d_yList);
	clReleaseMemObject(d_costs);
	clReleaseMemObject(d_delta);
	clReleaseMemObject(d_tSums);
	clReleaseMemObject(d_deltas);
	clReleaseMemObject(d_neuronCounts);
	clReleaseMemObject(d_dlayerCache);
	clReleaseMemObject(d_elayerCache);
	clReleaseMemObject(d_nlayerCache);
	clReleaseMemObject(d_matrixInfo);
	clReleaseKernel(k_neuralNetworkByGpu);				//Release kernel.
	clReleaseProgram(p_neuralNetworkByGpu);
	clReleaseCommandQueue(commandQueue);	//Release  Command queue.
	clReleaseContext(context);
	free(costs);
	free(tSums);
	free(d);
	free(dMatrixDimensions);
	delete[] dLayerCache;
	delete[] nLayerCache;
	delete[] eLayerCache;

}
GradientParameter* NeuralNetwork::calculateBackCostWithThetas(double lambda, double* thetas) {
//	printf("\n------------------------------------------------------------------------------------------------------------------------");
	double thetaSum = 0.0;
	double cost = 0;
	//struct timeval start, end;
	//gettimeofday(&start, NULL);
	d_tList = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * deltaSize, thetas, NULL);
	clSetKernelArg(k_neuralNetworkByGpu, 6, sizeof(cl_double), (void *) &lambda);
	clSetKernelArg(k_neuralNetworkByGpu, 7, sizeof(cl_mem), (void *) &d_tList);

	size_t global_work_size = ySize;

	cl_int status = clEnqueueNDRangeKernel(commandQueue, k_neuralNetworkByGpu, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	if (status != CL_SUCCESS) {
		cout << "Error: executing kernel!" << status << endl;
	}
	double lambdaMultiplier = lambda / ySizeDouble;

	clSetKernelArg(k_reduceDeltasByGpu, 7, sizeof(cl_double), (void *) &lambda);
	clSetKernelArg(k_reduceDeltasByGpu, 8, sizeof(cl_mem), (void *) &d_tList);
	clSetKernelArg(k_reduceDeltasByGpu, 12, sizeof(cl_double), (void *) &lambdaMultiplier);
	size_t global_work_size2 = deltaSize;
	status = clEnqueueNDRangeKernel(commandQueue, k_reduceDeltasByGpu, 1, NULL, &global_work_size2, NULL, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		cout << "Error: executing kernel!" << status << endl;
	}
	clEnqueueReadBuffer(commandQueue, d_costs, CL_TRUE, 0, ySize * sizeof(cl_double), costs, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_delta, CL_TRUE, 0, deltaSize * sizeof(cl_double), deltas, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueue, d_tSums, CL_TRUE, 0, deltaSize * sizeof(cl_double), tSums, 0, NULL, NULL);

	for (int i = 0; i < ySize; ++i) {
		cost += costs[i];
	}
	for (int i = 0; i < deltaSize; ++i) {
		thetaSum += tSums[i];
	}
	thetaSum = ((lambda / (2 * ySizeDouble)) * thetaSum);
	cost += thetaSum;
	//gettimeofday(&end, NULL);
	//uint diff = 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000;

	clReleaseMemObject(d_tList);
	//printf("\ntook: %llu ms,Cost is: %0.50f", diff, cost);

	return new GradientParameter(deltas, cost);

}

void NeuralNetwork::predict(double* tList, double* yTemp) {

	int totalCorrect = 0;
	int totalWrong = 0;

	for (int i = 0; i < ySize; ++i) {

		double* neurons = forwardPropogate(i, tList);
		double closer = RAND_MAX;
		double val = 0;
		for (int j = 0; j < numberOfLabels; j++) {

			if (fabs(1 - closer) > fabs((1 - neurons[nLayerCache[layerCount - 1] + j]))) {
				val = j + 1;
				closer = neurons[nLayerCache[layerCount - 1] + j];
			}
		}

		if (yTemp[i] == val) {
			totalCorrect++;
		} else {
			totalWrong++;
		}

		free(neurons);

	}

	printf("\nPrediction complete. Total %i correct and %i wrong prediction\n", totalCorrect, totalWrong);
	double successRate = totalCorrect * 100 / ySize;
	printf("\n Success rate is: %%%0.0f", successRate);
}
double* NeuralNetwork::forwardPropogate(int aListIndex, double* tList) {

	int mNeuronSize = sizeof(double) * neuronSize;
	double* neurons = (double *) malloc(mNeuronSize);
	for (int l = 0; l < layerCount; l++) {
		int previousLayer = nLayerCache[l];
		int xCache = xColumnSize * aListIndex;
		double* x = &(xList[xCache]);
		int dCache = dLayerCache[l - 1];
		int neuronSize = l == layerCount - 1 ? neuronCounts[l] : neuronCounts[l] + 1;
		for (int j = 0; j < neuronSize; j++) {
			int row = previousLayer + j;
			neurons[row] = 0;

			if (j == 0 && l != layerCount - 1) {
				neurons[row] = 1;
			} else if (l == 0) {
				neurons[row] = x[(j - 1)];
			} else {
				int pNCache = nLayerCache[l - 1];
				int index = l == layerCount - 1 ? j : j - 1;
				int dRowCache = (dMatrixDimensions[l - 1] * index) + dCache;
				int nCounts = neuronCounts[l - 1] + 1;
				double* t = &(tList[dRowCache]);
				double* n = &(neurons[pNCache]);
				for (int k = 0; k < nCounts; k++) {
					neurons[row] += t[k] * n[k];
				}

				neurons[row] = 1 / (1 + pow(E, -1 * neurons[row]));

			}
		}
	}

	return neurons;
}

