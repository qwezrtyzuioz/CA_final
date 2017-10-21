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
	int filtVol = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea = FMSIZE / 3 * FMSIZE / 3;
	int sum;
	// Convolution
	for (fn = 0; fn < FILTNUM; fn++){
		for (fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for (fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for (sli = 0; sli < FMDEPTH; sli++){
					for (y = 0; y < FILTSIZE; y++){
						for (x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if (ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if (sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	for (sli = 0; sli < FILTNUM; sli++){
		for (fmy = 0; fmy < FMSIZE / 3; fmy += 1){
			for (fmx = 0; fmx < FMSIZE / 3; fmx += 1){
				outNeuIdx = sli*fmArea + fmy * 3 * FMSIZE + fmx * 3;
				max = outNeu[outNeuIdx];
				for (y = 0; y < 3; y++){
					for (x = 0; x < 3; x++){
						ofmy = fmy * 3 + y;
						ofmx = fmx * 3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];
						if (tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE / 3 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int* ifm, int* ifilt, int* outNeu, int* outGPU)
{
	// ------------------------------------------------------------------------------
	//   Variables declaration
	// ------------------------------------------------------------------------------

	int depth = blockIdx.x,							// The depth this block deal with.
		filt_num = threadId.x,						// The filter this thread deal with.
		fm_area = FMSIZE * FMSIZE,					// Area of one feature map.
		pad_width = FILTSIZE / 2,					// Padding width
		pad_size = FMSIZE + pad_width * 2,			// Size of feature map after padding.
		filt_vol = FMDEPTH * FILTSIZE * FILTSIZE,	// Volume of 128 filters.(one depth)
		filt_area = FILTSIZE * FILTSIZE,			// Area of one filter.
		offset,										// Start point of iteration.
		filt_index,									// Index for filter.
		fm_index,									// Index for feature map.
		fm_ul,										// Upper left point of a sliding window
		sum,										// For inner product
		i, j,										// iterator
		fmx, fmy,									// iterator, on feature map.
		row, col,									// iterator, on filter.
		temp;										// Temporary storage.

	// ------------------------------------------------------------------------------
	//   Share memory Declaration
	// ------------------------------------------------------------------------------

	__share__ int fm[pad_size * pad_size];				// Feature map with specific depth,
	__share__ int filt[FILTNUM* FILTSIZE* FILTSIZE];	// 128 filter corresponding to the depth.

	// ------------------------------------------------------------------------------
	//   Share memory initialization
	// ------------------------------------------------------------------------------
	/*
	 * 1. Global input feature maps and filters store in "depth0, depth1.....depth n" 
	 *    order. We will copy on of depth into share memory called fm.
	 * 2. One depth of filters set including 128 filters. We will copy one depth ( 128
	 *    filters ) into share memory called filt.
	 * 3. Also do padding.
	 */

	// Fill the input feature map.
	// Using pad_size number of thread to fill the matrix.

	offset = filt_nun * pad_size;

	// Upper side zero padding.
	if (filt_num < pad_width){
		for (i = 0; i < pad_size; ++i){
			fm_index = offset + i;
			fm[fm_index] = 0;
		}
	}
	// Fill feature map from input argument. Also padding at the beginning and the ending.
	else if (filt_num < FMSIZE + pad_width){

		// Padding at the beginning.
		for (i = 0; i < pad_width; ++i){
			fm_index = offset + i;
			fm[fm_index] = 0;
		}
		// Fill the feature map from input argument.
		temp = (filt_num - pad_width) * FMSIZE;
		for (i = pad_width; i < FMSIZE + pad_width; ++i){
			fm_index = offset + i;
			fm[fm_index] = ifm[depth * fm_area + temp + i - pad_width];
		}
		// Padding at the ending.
		for (i = FMSIZE + pad_width; i < pad_size; ++i){
			fm_index = offset + i;
			fm[fm_index] = 0;
		}
	}
	// Lower side zero padding.
	else if (filt_num < pad_size){
		for (i = 0; i < pad_size; ++i){
			fm_index = offset + i;
			fm[fm_index] = 0;
		}
	}

	// Fill 128 corresponding filters.
	// Using all 128 thread.
	offset = filt_num * filt_vol + depth * filt_area;
	for (int i = 0, i < filt_area; ++i){
		filt_index = offset + i;
		filt[filt_index] = ifilt[depth * filt_vol + filt_index];
	}

	__syncthreads();	// End of share memory initialization.

	// ------------------------------------------------------------------------------
	//   Inner product
	// ------------------------------------------------------------------------------
	
	offset = filt_num * filt_area;
	for (fmy = pad_width; fmy < FMSIZE + pad_widht; ++fmy){
		for (fmx = pad_width; fmx < FMSIZE + pad_widht; ++fmx){

			fm_ul = (fmy - pad_width)* pad_size + fmx;
			sum = 0;
			for (row = 0; i < FILTSIZE; ++i){
				for (col = 0; j < FILTSIZE; ++j){
					fm_index = fm_ul + pad_width* row + col;
					filt_index = offset + row * FILTSIZE + col;
					sum += fm[fm_index] * filt[filt_index];
				}
			}
		}
	}

}
/***	Implement your CUDA Kernel here	***/

int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end;
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();


	//Convolution by CPU                                                
clock_gettime(CLOCK_REALTIME, &time_begin);

	convLayerCPU();

clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = " << ((float)convLayerCPUExecTime) / 1000 << "ms" << endl;


	//Convolution by GPU   
clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/

	convLayerGPU << <FMDEPTH, FILTNUM >> >(inNeu, filt, outNeu, GPUout); // Lunch the kernel
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()

	/***	Lunch your CUDA Kernel here	***/
clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " << ((float)convLayerGPUExecTime) / 1000 << "ms" << endl;


	//check the anser from CPU and from GPU
	if (checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	//release memory space
	ending();

	return 0;
}