__kernel void calculate(
__global double *deltas,
const __global double* xList, 
const int ySize, 
const __global double* yList, 
const int layerCount, 
const __global int* neuronCounts, 
const double lambda, 
const __global double* thetas,  
const int totalNeuronSize, 
const int errorSize, 
const int deltaSize, 
const int xListRows, 
const __global int* dlayerCache, 
const __global int* dMatrixInfo, 
const __global int* nLayerCache, 
const __global int* eLayerCache, 
const int numLabels,
__global double* costs,
const double eDef) {

	uint m = get_global_id(0);
	int yCache = m * numLabels;
	int xCache = xListRows * m;
	double neurons[1024];
	double errors[1024];
	
	for (int l = 0; l < layerCount; l++) {
		int previousLayer = nLayerCache[l];

		int neuronSize = l == layerCount - 1 ? neuronCounts[l] : neuronCounts[l] + 1;
		for (int j = 0; j < neuronSize; j++) {
			int row = previousLayer + j;
			neurons[row] = 0;

			if (j == 0 && l != layerCount - 1) {
				neurons[row] = 1;
			} else if (l == 0) {
				neurons[row] = xList[xCache + (j - 1)];
			} else {
				int dCache = dlayerCache[l - 1];
				int pNCache = nLayerCache[l - 1];
				int index = l == layerCount - 1 ? j : j - 1;
				int dRowCache = (dMatrixInfo[l - 1] * index) + dCache;
				int nCounts = neuronCounts[l - 1] + 1;
				for (int k = 0; k < nCounts; k++) {
					neurons[row] += thetas[dRowCache + k] * neurons[pNCache + k];
				}

				neurons[row] = 1 / (1 + pow(eDef, -1 * neurons[row]));
			}
		}
	}

	for (int i = layerCount - 2; i >= 0; i--) {

		int neuronSize = i == layerCount - 2 ? neuronCounts[i + 1] : neuronCounts[i + 1] + 1;
		int previousLayer = eLayerCache[i];
		int nCache = nLayerCache[i + 1];

		int dCache = dlayerCache[i + 1];
		int eCache = eLayerCache[i + 1];
		for (int j = neuronSize - 1; j >= 0; j--) {
			int row = previousLayer + j;

			errors[row] = 0; //reset
			double nVal = neurons[nCache + j];
			if (i == layerCount - 2) {
				errors[row] = nVal - yList[yCache + j];
			} else {
				int nCounts = neuronCounts[i + 2];
				int isLast = nCounts - 1;
				double sigmoid = (nVal * (1 - nVal));
				int dif = nCounts % 4;
				int siz = nCounts - dif;
				for (int k = 0; k < siz; k = k + 4) {
					int r = (dMatrixInfo[i + 1] * k);
					int r1 = (dMatrixInfo[i + 1] * (k + 1));
					int r2 = (dMatrixInfo[i + 1] * (k + 2));
					int r3 = (dMatrixInfo[i + 1] * (k + 3));
					errors[row] += thetas[dCache + j + r] * errors[eCache + k];
					errors[row] += thetas[dCache + j + r1] * errors[eCache + k + 1];
					errors[row] += thetas[dCache + j + r2] * errors[eCache + k + 2];
					errors[row] += thetas[dCache + j + r3] * errors[eCache + k + 3];

				}

				for (int a = 0; a < dif; a++) {
					int k = siz + a;
					int r = (dMatrixInfo[i + 1] * k);
					errors[row] += thetas[dCache + j + r] * errors[eCache + k];

					if (k == isLast) {
						errors[row] = errors[row] * sigmoid;
					}
				}
			}

		}
	}

	double sum = 0.0;
	for (int i = 0; i < layerCount - 1; i++) {
		int n1 = neuronCounts[i + 1];
		int n2 = neuronCounts[i] + 1;
		int nCache1 = nLayerCache[i + 1];
		int eCache = eLayerCache[i];
		int nCache = nLayerCache[i];
		int isLast = i == layerCount - 2;
		int dCache = (m * deltaSize) + dlayerCache[i];
		int dif = n1 % 4;
		int siz = n1 - dif;
		for (int j = 0; j < siz; j = j + 4) {
			if (isLast) {

				sum += ((-1 * yList[yCache + j]) * log(neurons[nCache1 + j])) - ((1 - yList[yCache + j]) * log(1 - neurons[nCache1 + j]));
				sum += ((-1 * yList[yCache + j + 1]) * log(neurons[nCache1 + j + 1])) - ((1 - yList[yCache + j + 1]) * log(1 - neurons[nCache1 + j + 1]));
				sum += ((-1 * yList[yCache + j + 2]) * log(neurons[nCache1 + j + 2])) - ((1 - yList[yCache + j + 2]) * log(1 - neurons[nCache1 + j + 2]));
				sum += ((-1 * yList[yCache + j + 3]) * log(neurons[nCache1 + j + 3])) - ((1 - yList[yCache + j + 3]) * log(1 - neurons[nCache1 + j + 3]));
			}
			int index = i == 0 ? j + 1 : j;
			int index2 = index + 1;
			int index3 = index + 2;
			int index4 = index + 3;
			int drcache = (dMatrixInfo[i] * j);
			int drcache2 = (dMatrixInfo[i] * (j + 1));
			int drcache3 = (dMatrixInfo[i] * (j + 2));
			int drcache4 = (dMatrixInfo[i] * (j + 3));
			double eVal = errors[eCache + index];
			double eVal2 = errors[eCache + index2];
			double eVal3 = errors[eCache + index3];
			double eVal4 = errors[eCache + index4];
			int d2 = dCache + drcache;
			int d22 = dCache + drcache2;
			int d23 = dCache + drcache3;
			int d24 = dCache + drcache4;
			int diff = n2 % 4;
			int size = n2 - diff;
			for (int k = 0; k < size; k = k + 4) {
				double nval = neurons[nCache + k];
				double nval2 = neurons[nCache + k + 1];
				double nval3 = neurons[nCache + k + 2];
				double nval4 = neurons[nCache + k + 3];
				deltas[d2 + k] = eVal * nval;
				deltas[d2 + k + 1] = eVal * nval2;
				deltas[d2 + k + 2] = eVal * nval3;
				deltas[d2 + k + 3] = eVal * nval4;

				deltas[d22 + k] = eVal2 * nval;
				deltas[d22 + k + 1] = eVal2 * nval2;
				deltas[d22 + k + 2] = eVal2 * nval3;
				deltas[d22 + k + 3] = eVal2 * nval4;

				deltas[d23 + k] = eVal3 * nval;
				deltas[d23 + k + 1] = eVal3 * nval2;
				deltas[d23 + k + 2] = eVal3 * nval3;
				deltas[d23 + k + 3] = eVal3 * nval4;

				deltas[d24 + k] = eVal4 * nval;
				deltas[d24 + k + 1] = eVal4 * nval2;
				deltas[d24 + k + 2] = eVal4 * nval3;
				deltas[d24 + k + 3] = eVal4 * nval4;
			}
			for (int d = 0; d < diff; d++) {
				double nVal = neurons[nCache + size + d];
				deltas[d2 + size + d] = eVal * nVal;
				deltas[d22 + size + d] = eVal2 * nVal;
				deltas[d23 + size + d] = eVal3 * nVal;
				deltas[d24 + size + d] = eVal4 * nVal;
			}
		}

		for (int a = 0; a < dif; a++) {
			int j = a + siz;
			if (isLast) {

				sum += ((-1 * yList[yCache + j]) * log(neurons[nCache1 + j])) - ((1 - yList[yCache + j]) * log(1 - neurons[nCache1 + j]));
			}
			int index = i == 0 ? j + 1 : j;
			int drcache = (dMatrixInfo[i] * j);
			double eVal = errors[eCache + index];
			int d2 = dCache + drcache;
			int diff = n2 % 4;
			int size = n2 - diff;
			for (int k = 0; k < size; k = k + 4) {
				deltas[d2 + k] = eVal * neurons[nCache + k];
				deltas[d2 + k + 1] = eVal * neurons[nCache + k + 1];
				deltas[d2 + k + 2] = eVal * neurons[nCache + k + 2];
				deltas[d2 + k + 3] = eVal * neurons[nCache + k + 3];
			}
			for (int d = 0; d < diff; d++) {
				double nVal = neurons[nCache + size + d];
				deltas[d2 + size + d] = eVal * nVal;
			}
		}

	}
	double ySizeDouble = ySize;
	costs[m] = (1 / ySizeDouble) * sum;

}