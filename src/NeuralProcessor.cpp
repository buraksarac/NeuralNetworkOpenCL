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
#include "IOUtils.h"
#include "Fmincg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
clock_t start, end;
void printHelp() {
	printf("\nUSAGE:\n");
	printf("\n--help\tThis help info\n");
	printf("\n-x\tX(input) file path\n");
	printf("\n-y\tY(expected result) file path\n");
	printf("\n-r\tRowcount of X or Y file (should be equal)\n");
	printf("\n-c\tColumn count of X file (each row should have same count)\n");
	printf("\n-n\tNumber of labels in Y file (how many expected result)\n");
	printf("\n-t\tTotal layer count for neural network(including X)\n");
	printf("\n-h\tHidden layer size (excluding bias unit)\n");
	printf("\n-i\tNumber of iteration for training\n");
	printf("\n-l\tLambda value\n");
	printf("\n-p\tDo prediction for each input after training complete (0 for disable 1 for enable default 1)\n");
	printf("\n-tp\tTheta path. If you have previously saved a prediction result you can continue"
			"\n\tfrom this result by loading from file path. (-lt value should be 1)\n");
	printf("\n-lt\tLoad previously saved thetas (prediction result)"
			"\n\t(0 for disable 1 for enable default 0) (-tp needs to be set)\n");
	printf("\n-st\tSave thetas (prediction result)(0 for disable 1 for enable default 1)\n");
	printf("\n");
	printf("\nPlease see http://www.u-db.org for more details\n");
}
int main(int argc, char **argv) {

	string aPath; // = "x.dat";
	string bPath; // = "y.dat";
	string tPath;
	int rowCount;
	int colCount;
	int numberOfLabels;
	int totalLayerCount;
	int hiddenLayerCount;
	int maxIteration;
	double lambda = 1;
	int predict = 1;
	int loadThetas = 0;
	int saveThetas = 1;
	if ((argc % 1) != 0) {
		printf("Invalid parameter size");
	}
	for (int i = 1; i < argc; i = i + 2) {
		if (!strcmp(argv[i], "--help")) {
			printHelp();
			return 1;
		} else if (!strcmp(argv[i], "-x")) {
			aPath = argv[i + 1];
		} else if (!strcmp(argv[i], "-y")) {
			bPath = argv[i + 1];
		} else if (!strcmp(argv[i], "-r")) {
			rowCount = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-c")) {
			colCount = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-n")) {
			numberOfLabels = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-t")) {
			totalLayerCount = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-h")) {
			hiddenLayerCount = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-i")) {
			maxIteration = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-l")) {
			lambda = atof(argv[i + 1]);
		} else if (!strcmp(argv[i], "-p")) {
			predict = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-tp")) {
			tPath = argv[i + 1];
		} else if (!strcmp(argv[i], "-lt")) {
			loadThetas = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-st")) {
			saveThetas = atoi(argv[i + 1]);
		} else {
			printf("Couldnt recognize user input %s", argv[i]);
			return 0;
		}

	}
	printf("Start!\n");

	double* xlist = IOUtils::getArray(aPath, rowCount, colCount);
	double* yTemp = IOUtils::getArray(bPath, rowCount, 1);

	double* ylist = new double[rowCount * numberOfLabels];

	for (int r = 0; r < rowCount; r++) {
		for (int c = 0; c < numberOfLabels; c++) {
			ylist[(r * numberOfLabels) + c] = ((c + 1) == abs(yTemp[r])) ? 1 : 0;
		}
	}
	int itemCount = colCount + 1;
	int* neuronCount = new int[totalLayerCount];
	neuronCount[0] = colCount;
	for (int j = 1; j < totalLayerCount - 1; ++j) {
		neuronCount[j] = hiddenLayerCount;
		itemCount += hiddenLayerCount + 1;
	}
	neuronCount[totalLayerCount - 1] = numberOfLabels;
	itemCount += numberOfLabels;

	if(itemCount > 1000){
		printf("Sorry openCL version doesnt support currently more than 1000 item\n Try reducing column or hiddenlayer count.");
		return 0;
	}


	double* tList;
	double thetaRowCount = 0;
	for (int i = 0; i < totalLayerCount - 1; i++) {
		for (int j = 0; j < neuronCount[i + 1]; j++) {
			for (int k = 0; k < neuronCount[i] + 1; k++) {
				thetaRowCount++;
			}
		}
	}

	start = clock();

	GradientParameter* gd;
	if (loadThetas) {
		tList = IOUtils::getArray(tPath, thetaRowCount, 1);
		gd = Fmincg::calculate(thetaRowCount, numberOfLabels, maxIteration, xlist, rowCount, colCount, ylist, totalLayerCount, neuronCount, lambda, tList);
	} else {
		gd = Fmincg::calculate(numberOfLabels, maxIteration, xlist, rowCount, colCount, ylist, totalLayerCount, neuronCount, lambda);
	}
	end = clock();
	double diff = (((double) end - (double) start) / 1000000.0F) * 1000;

	printf("\n\nProcess took: %0.3f millisecond \n", diff);

	if (saveThetas) {
		IOUtils::saveThetas(gd->getThetas(), thetaRowCount);
	}
	printf("\nPrediction will start. Gd: %0.14f\n", gd->getCost());
	NeuralNetwork* nn = Fmincg::getNN();
	if (predict) {

		nn->predict(gd->getThetas(), yTemp);

	}
	nn->destroy();
	delete nn;
	delete[] ylist;
	free(xlist);
	free(yTemp);
	delete[] neuronCount;
	gd->destroy();
	delete gd;
	printf("\nFinish!");
}

