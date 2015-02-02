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

#ifndef SRC_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_H_
#include "GradientParameter.h"
#include <string.h>
#include <CL/cl.h>
#include <iostream>
using namespace std;
class NeuralNetwork {
private:
	int layerCount;
	int* neuronCounts;
	int numberOfLabels;
	int ySize;
	double ySizeDouble;
	int dDim1;
	int* dMatrixDimensions;
	int* dLayerCache;
	int* nLayerCache;
	int* eLayerCache;
	int dMatrixSize;
	int xDim2;
	int yDim2;
	int neuronSize;
	int errorSize;
	int deltaSize;
	int mNeuronSize;
	int mErrorSize;
	int mDeltaSize;
	double* deltas;
	double* tSums;
	cl_context context;
	cl_command_queue commandQueue;
	cl_program p_neuralNetworkByGpu;
	cl_kernel k_neuralNetworkByGpu;
	cl_program p_reduceDeltasByGpu;
	cl_kernel k_reduceDeltasByGpu;
	cl_mem d_xList;
	cl_mem d_deltas;
	cl_mem d_yList;
	cl_mem d_tList;
	cl_mem d_delta;
	cl_mem d_tSums;
	cl_mem d_costs;
	cl_mem d_neuronCounts;
	cl_mem d_dlayerCache;
	cl_mem d_elayerCache;
	cl_mem d_nlayerCache;
	cl_mem d_matrixInfo;
	double* costs;
	double* xList;
	double* yList;
	double* d;
	int xColumnSize;
public:
	NeuralNetwork(double* xList, double* yList,int layerCount, int* neuronCounts, int numberOfLabels, int ySize, int xColumnSize);
	int convertToString(const char *filename, string& s);
	void destroy();
	GradientParameter* calculateBackCostWithThetas(double lambda, double* thetas);
	double* forwardPropogate(int aListIndex, double* tList);
	void predict(double* tList, double* yTemp);

};
#endif /* SRC_NEURALNETWORK_H_ */
