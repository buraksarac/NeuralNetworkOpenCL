/*
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
%
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
%
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.

 Changes mage:
  burak sarac : c/c++ implementation
 */

#include "Fmincg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GradientParameter.h"
#include "NeuralNetwork.h"
#include <limits>
#include <sys/time.h>

using namespace std;
static const double RHO = 0.01; // a bunch of constants for line searches
static const double SIG = 0.5; // RHO and SIG are the constants in the Wolfe-Powell conditions
static const double INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket
static const double EXT = 3.0; // extrapolate maximum 3 times the current bracket
static const double MAX = 20; // max 20 function evaluations per line search
static const double RATIO = 100; // maximum allowed slope ratio
#define E (2.7182818284590452353602874713526624977572470937L )

static NeuralNetwork* neuralNetwork;
NeuralNetwork* Fmincg::getNN() {
	return neuralNetwork;
}
GradientParameter* Fmincg::calculate(int numberOfLabels, int maxIterations, double* aList, int ySize, int xColumnSize, double* yList, int totalLayerCount, int* neuronCounts,
		double lambda) {

	double thetaRowCount = 0;
	for (int i = 0; i < totalLayerCount - 1; i++) {
		for (int j = 0; j < neuronCounts[i + 1]; j++) {
			for (int k = 0; k < neuronCounts[i] + 1; k++) {
				thetaRowCount++;
			}
		}
	}

	double* thetas = (double *) malloc(sizeof(double) * thetaRowCount);
	for (int i = 0; i < totalLayerCount - 1; i++) {
		for (int j = 0; j < neuronCounts[i + 1]; j++) {
			for (int k = 0; k < neuronCounts[i] + 1; k++) {
				thetas[i] = ((double) rand() / (RAND_MAX)) * (2 * numeric_limits<double>::epsilon()) - numeric_limits<double>::epsilon();
			}
		}
	}

	return Fmincg::calculate(thetaRowCount, numberOfLabels, maxIterations, aList, ySize, xColumnSize, yList, totalLayerCount, neuronCounts, lambda, thetas);

}

GradientParameter* Fmincg::calculate(int thetaRowCount, int numberOfLabels, int maxIterations, double* aList, int ySize, int xColumnSize, double* yList, int layerCount,
		int* neuronCounts, double lambda, double* tList) {

	double* x = tList;

	neuralNetwork = new NeuralNetwork(aList, yList, layerCount, neuronCounts, numberOfLabels, ySize, xColumnSize);
	int i = 0;
	int ls_failed = 0;   // no previous line search has failed
	int n = 0;
//gd instance will change during the iteration
	GradientParameter* gd = neuralNetwork->calculateBackCostWithThetas(lambda, x);
	n++;
	double d1 = 0.0; //search direction is steepest and calculate slope
	double f1 = gd->getCost();
	double* df1 = new double[thetaRowCount];
	double* s = new double[thetaRowCount];
	double* results = new double[maxIterations];

	for (int r = 0; r < thetaRowCount; r++) {
		df1[r] = gd->getThetas()[r];
		s[r] = -1 * df1[r];
		d1 += -1 * s[r] * s[r];
	}
	delete gd;
	double z1 = 1 / (1 - d1);

	double* x0 = new double[thetaRowCount];
	double* df0 = new double[thetaRowCount];
	double* df2 = new double[thetaRowCount];
	double d2 = 0.0;
	double f3 = 0.0;

	double A = 0.0;
	double B = 0.0;
	while (i < abs(maxIterations)) {
		i++;
		//lets start

		//X0 = X; f0 = f1; df0 = df1; make a copy of current values
		double f0 = f1;

		for (int r = 0; r < thetaRowCount; r++) {
			x0[r] = x[r]; //copy x value into x0
			df0[r] = df1[r]; //copy df1 value into df0
			x[r] += z1 * s[r]; //update x as X = X + z1*s;
		}

		//request new gradient after we update X -- octave -->[f2 df2] = eval(argstr);
		GradientParameter* gd2 = neuralNetwork->calculateBackCostWithThetas(lambda, x);
		n++;
		double f2 = gd2->getCost();

		d2 = 0.0;
		for (int r = 0; r < thetaRowCount; r++) {
			df2[r] = gd2->getThetas()[r];
			d2 += s[r] * df2[r]; // d2 = df2'*s;
		}

		//f3 = f1; d3 = d1; z3 = -z1;        initialize point 3 equal to point 1
		f3 = f1;
		double d3 = d1;
		double z3 = -1 * z1;
		double M = MAX;
		int success = 0;
		double limit = -1;
		delete gd2;
		while (1) {
			double z2 = 0.0;
			while (((f2 > f1 + (z1 * RHO * d1)) | (d2 > (-1 * SIG * d1))) & (M > 0)) {
				limit = z1;

				if (f2 > f1) {
					z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
				} else {
					A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
					B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
					z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A;
				}

				if (isnan(z2) | isinf(z2)) {
					z2 = z3 / 2;
				}

				double z3m1 = INT * z3;
				double z3m2 = (1 - INT) * z3;
				double min = z2 < z3m1 ? z2 : z3m1;
				z2 = min > z3m2 ? min : z3m2;
				z1 = z1 + z2;

				for (int r = 0; r < thetaRowCount; ++r) {
					x[r] += z2 * s[r]; //update x as X = X + z2*s;
				}

				GradientParameter* gd3 = neuralNetwork->calculateBackCostWithThetas(lambda, x);
				n++;
				M = M - 1;
				f2 = gd3->getCost();

				d2 = 0.0;
				for (int r = 0; r < thetaRowCount; ++r) {
					df2[r] = gd3->getThetas()[r];
					d2 += s[r] * df2[r]; // d2 = df2'*s;
				}
				delete gd3;
				z3 = z3 - z2;                    // z3 is now relative to the location of z2
			}

			if ((f2 > f1 + (z1 * RHO * d1)) | (d2 > -1 * SIG * d1)) {
				break;
			} else if (d2 > SIG * d1) {
				success = 1;
				break;
			} else if (M == 0) {
				break;
			}

			A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);                      // make cubic extrapolation
			B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
			z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3));        // num. error possible - ok!
			if (isnan(z2) | isinf(z2) | (z2 < 0)) {
				if (limit < -1 * 0.5) {
					z2 = z1 * (EXT - 1);
				} else {
					z2 = (limit - z1) / 2;
				}
			} else if ((limit > -0.5) & (z2 + z1 > limit)) {
				z2 = (limit - z1) / 2;
			} else if ((limit < -0.5) & (z2 + z1 > z1 * EXT)) {
				z2 = z1 * (EXT - 1.0);
			} else if (z2 < -z3 * INT) {
				z2 = -z3 * INT;
			} else if ((limit > -0.5) & (z2 < (limit - z1) * (1.0 - INT))) {
				z2 = (limit - z1) * (1.0 - INT);
			}

			f3 = f2;
			d3 = d2;
			z3 = -1 * z2;
			z1 = z1 + z2;

			for (int r = 0; r < thetaRowCount; r++) {
				x[r] += z2 * s[r]; //update x as X = X + z2*s;
			}

			GradientParameter* gd4 = neuralNetwork->calculateBackCostWithThetas(lambda, x);
			n++;
			M = M - 1;
			f2 = gd4->getCost();
			d2 = 0.0;
			for (int r = 0; r < thetaRowCount; r++) {
				df2[r] = gd4->getThetas()[r];
				d2 += df2[r] * s[r]; // d2 = df2'*s;
			}
			delete gd4;
		}

		if (success) {
			f1 = f2;
			results[i] = f1;
			printf("\n Next success cost: %0.50f total %i iteration and %i neural calculation complete", f1, i, n);
			// Polack-Ribiere direction
			double sum1 = 0.0;
			double sum2 = 0.0;
			double sum3 = 0.0;
			for (int r = 0; r < thetaRowCount; r++) {
				sum1 += df2[r] * df2[r];
				sum2 += df1[r] * df2[r];
				sum3 += df1[r] * df1[r];
			}

			double p = (sum1 - sum2) / sum3;
			d2 = 0.0;
			for (int r = 0; r < thetaRowCount; r++) {
				s[r] = p * s[r] - df2[r];
				double tmp = df1[r];
				df1[r] = df2[r];
				df2[r] = tmp;
				d2 += df1[r] * s[r]; // d2 = df1'*s;
			}

			if (d2 > 0) {
				d2 = 0.0;
				for (int r = 0; r < thetaRowCount; r++) {
					s[r] = -1 * df1[r]; // s = -df1;
					d2 += -1 * s[r] * s[r]; // d2 = -s'*s;
				}
			}

			double sum4 = d1 / (d2 - numeric_limits<double>::min());
			z1 = z1 * (RATIO < sum4 ? RATIO : sum4);
			d1 = d2;
			ls_failed = 0;

		} else {

			f1 = f0;
			for (int r = 0; r < thetaRowCount; r++) {
				x[r] = x0[r];
				df1[r] = df0[r];
			}

			if (ls_failed) {
				break;
			}

			d1 = 0.0;
			for (int r = 0; r < thetaRowCount; r++) {
				double tmp = df1[r];
				df1[r] = df2[r];
				df2[r] = tmp;
				s[r] = -1 * df1[r];
				d1 += -1 * s[r] * s[r];
			}

			z1 = 1 / (1 - d1);
			ls_failed = 1;

		}

	}

	delete[] df1;
	delete[] s;
	delete[] x0;
	delete[] df0;
	delete[] df2;

	return new GradientParameter(x, results);

}

