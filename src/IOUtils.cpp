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

#include "IOUtils.h"
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <sys/time.h>
struct timeval timeValue;
IOUtils::IOUtils() {
	// TODO Auto-generated constructor stub

}

void IOUtils::saveThetas(double* thetas, lint size) {
	gettimeofday(&timeValue, NULL);
	string fileName = "thetas_";
	std::stringstream sstm;
	sstm << fileName << timeValue.tv_sec << ".dat";

	ofstream f(sstm.str().c_str());
	copy(thetas, thetas + size, ostream_iterator<double>(f, "\n"));
	printf("Thetas (%s) has been saved into project folder.", sstm.str().c_str());
}

double* IOUtils::getArray(string path, lint rows, lint columns) {

	ifstream inputStream;

	lint currentRow = 0;
	std::string s;
	inputStream.open(path.c_str());

	lint size = columns * rows;
	lint mListSize = sizeof(double) * size;
	double* list = (double *) malloc(mListSize);

	while (!inputStream.eof()) {

		if (currentRow < size) {

			inputStream >> s;
			list[currentRow++] = strtod(s.c_str(), NULL);

		} else {
			break;
		}
	}

	inputStream.close();

	return list;
}
