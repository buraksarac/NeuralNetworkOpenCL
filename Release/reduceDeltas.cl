__kernel void reduce(
const __global double *deltas,
const int ySize,
__global double* result,
const __global int* nCounts,
const __global int* dLayerCache,
const __global int* matrixInfo,
const int layerCount,
const double lambda,
const __global double *thetas,
__global double *tSums,
const int deltaSize,
const double yMultiplier,
const double lambdaMultiplier) {

	uint m = get_global_id(0);
	result[m] = 0;
	int layer = 0;
	for(int i = 0;i<layerCount - 1;i++){
	
		if(m < dLayerCache[i + 1]){
			layer = i;
			break;
		}
	
	}
	int col = (m - dLayerCache[layer]) % matrixInfo[layer];
	int row = (m - dLayerCache[layer]) / matrixInfo[layer];
	double sum = 0.0;
	double correction = 0.0;
	for(int i = 0;i<ySize;i++){
		double value = deltas[(i*deltaSize) + m];
		double y = value - correction;
		double t = sum + y;
		correction = (t - sum) - y;
		sum = t;
	}
	result[m] = yMultiplier * sum;
	
	if(col > 0){
	 result[m] += lambdaMultiplier * thetas[m];
	 
	 tSums[m] = thetas[m] * thetas[m];
		
	}
}