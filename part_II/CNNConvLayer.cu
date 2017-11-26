// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
using namespace std;

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea  = FMSIZE/3 * FMSIZE/3;
	int sum;
	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/3 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/3 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 3; y++){
					for(x = 0; x < 3; x++){
						ofmy = fmy*3 + y;
						ofmx = fmx*3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU()
{
}
/***	Implement your CUDA Kernel here	***/

int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end; 
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	initCoo();

	//Convolution by CPU                                                
	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = "  <<  ((float)convLayerCPUExecTime)/1000 << "ms" << endl;

  
	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	convLayerGPU<<<1,1>>>(); // Lunch the kernel
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;

	
	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	//release memory space
	ending();
	
	return 0;
}
