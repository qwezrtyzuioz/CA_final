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
void convLayerGPU(int* ifm, int* ifilt, int* dev_outNeu, int* dev_outGPU)
{ 
	// ------------------------------------------------------------------------------
	//   Variables declaration
	// ------------------------------------------------------------------------------

	const int
		depth = blockIdx.x,							// The depth this block deal with.
		filt_num = threadIdx.x,						// The filter this thread deal with.
		fm_area = FMSIZE * FMSIZE,					// Area of one feature map.
		pad_width = FILTSIZE / 2,					// Padding width
		pad_size = FMSIZE + pad_width * 2,			// Size of feature map after padding.
		pad_area = pad_size * pad_size,				// Area of featrue map after padding.
		filt_vol = FMDEPTH * FILTSIZE * FILTSIZE,	// Volume of 128 filters.(one depth)
		filt_area = FILTSIZE * FILTSIZE,			// Area of one filter.
		out_area = (FMSIZE / 3) * (FMSIZE / 3);
	int
		offset,										// Start point of iteration.
		filt_index,									// Index for filter.
		fm_index,									// Index for feature map.
		fm_ul,										// Upper left point of a sliding window
		sum,										// For inner product
		i, j,										// iterator
		fmx, fmy,									// iterator, on feature map.
		row, col,									// iterator, on filter.
		temp;										// Temporary storage.

	int filt[FILTSIZE * FILTSIZE];	// 128 filter corresponding to the depth.


	// ------------------------------------------------------------------------------
	//   Share memory Declaration
	// ------------------------------------------------------------------------------

	__shared__ int fm[pad_area];
	__syncthreads();
	// ------------------------------------------------------------------------------
	//   Share memory initialization
	// ------------------------------------------------------------------------------
	/*
	 * 1. Global input feature maps store in "depth0, depth1.....depth n" order. 
	 *    We will copy on of depth into share memory called fm.   
	 * 2. Also do padding.
	 */

	// Fill the input feature map.
	// Using pad_size number of thread to fill the matrix.
    //
	offset = filt_num * pad_size;
    
	//// Upper side zero padding.
	if (filt_num < pad_width) {
		for (i = 0; i < pad_size; ++i) {
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

	//------------------------------------------------------------------------------
	//	Local memory initialization
	//------------------------------------------------------------------------------
	
	/*
	 * 1. Global input filters store in "depth0, depth1.....depth n" order. 
	 * 2. One kind of filter include 96 depth. Total 128 kind of filter.
	 */

	// Using all 128 thread.
	offset = filt_num * filt_vol + depth * filt_area;
	for (int i = 0; i < filt_area; ++i){
		filt[i] = ifilt[offset + i];
	}
    
	__syncthreads();	// End of share memory initialization.
	
	// ------------------------------------------------------------------------------
	//   Inner product
	// ------------------------------------------------------------------------------
	
	for (fmy = 0; fmy < FMSIZE ; ++fmy) {
		for (fmx = 0; fmx < FMSIZE ; ++fmx) {
    
			// Envalue the index of upper and left point of the ROI.
			fm_ul = fmy * pad_size + fmx;
			sum = 0;
    
			// For each point in ROI.
			for (row = 0; row < FILTSIZE; ++row) { 
				for (col = 0; col < FILTSIZE; ++col) {
    
					fm_index = fm_ul + pad_size* row + col;
					filt_index = row * FILTSIZE + col;
					sum += fm[fm_index] * filt[filt_index];
				}
			}
			atomicAdd(&dev_outNeu[threadIdx.x * fm_area + fmy*FMSIZE + fmx], sum);
		}
	}
	
	
}
/***	Implement your CUDA Kernel here	***/
__global__
void Activation_Pooling_GPU(int* dev_outNeu, int* dev_outGPU){
	const int
		depth = blockIdx.x,							// The depth this block deal with.
		filt_num = threadIdx.x,						// The filter this thread deal with.
		fm_area = FMSIZE * FMSIZE,					// Area of one feature map.
		pad_width = FILTSIZE / 2,					// Padding width
		pad_size = FMSIZE + pad_width * 2,			// Size of feature map after padding.
		pad_area = pad_size * pad_size,				// Area of featrue map after padding.
		filt_vol = FMDEPTH * FILTSIZE * FILTSIZE,	// Volume of 128 filters.(one depth)
		filt_area = FILTSIZE * FILTSIZE,			// Area of one filter.
		out_area = (FMSIZE / 3) * (FMSIZE / 3);
	int
		offset,										// Start point of iteration.
		filt_index,									// Index for filter.
		fm_index,									// Index for feature map.
		fm_ul,										// Upper left point of a sliding window
		sum,										// For inner product
		i, j,										// iterator
		fmx, fmy,									// iterator, on feature map.
		row, col,									// iterator, on filter.
		temp;										// Temporary storage.

	int filt[FILTSIZE * FILTSIZE];	// 128 filter corresponding to the depth.
	
	// Activation - ReLU
	if( blockIdx.x < 64) {
		if( threadIdx.x < FMSIZE) {
			for(int i = 0; i< 2; ++i) {
			offset = blockIdx.x * 2 * fm_area + i * fm_area + threadIdx.x * FMSIZE;
				for (int j = 0; j< FMSIZE; ++j) {
					if(dev_outNeu[ offset+ j]<0) {
						dev_outNeu[offset+ j] = 0;
					}	
				}
			}
		}
	}
	
	__syncthreads();
	
	// Max Pooling
	int max;
	if( blockIdx.x < 64) {
		if( threadIdx.x < 81) {
			
			int grid_row = threadIdx.x/ 9;
			int grid_col = threadIdx.x% 9;
			
			for(int i = 0 ; i < 2; ++i) {
				
				//lu point location
				offset = blockIdx.x * 2 * fm_area + i * fm_area + grid_row * 3 * FMSIZE + grid_col * 3;
				max = dev_outNeu[offset];
				
				for (int j = 0; j < 3; ++j) { 
					for (int k=0;k<3;++k) {
						if (dev_outNeu[ offset + j*FMSIZE + k ]>=max) {
							max = dev_outNeu[ offset + j*FMSIZE + k ];
						}
					}	
				}
				dev_outGPU[blockIdx.x*2* out_area+i* out_area + threadIdx.x] = max;
			}
		}
	}

}
int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end;
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	initGPU();

	//Convolution by CPU                                                
	clock_gettime(CLOCK_REALTIME, &time_begin);

	convLayerCPU();

	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = " << ((float)convLayerCPUExecTime) / 1000 << "ms" << endl;


	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	
	cout<< endl;
	dim3 numBlocks(FMDEPTH);
	dim3 threadsPerBlock(FILTNUM);
	convLayerGPU<<<numBlocks, threadsPerBlock>>>(dev_ifm, dev_ifilt, dev_outNeu, dev_outGPU); // Lunch the kernel
	cout<<"cudaDeviceSynchronize: "<< cudaGetErrorString(cudaDeviceSynchronize())<< endl; // Do synchronization before clock_gettime()
	
	Activation_Pooling_GPU<<<numBlocks, threadsPerBlock>>>(dev_outNeu, dev_outGPU);
	cout<<"cudaMemcpy:"<< cudaGetErrorString(cudaMemcpy(outGPU, dev_outGPU,
														sizeof(int)* FILTNUM * FMSIZE/3 * FMSIZE/3,
														cudaMemcpyDeviceToHost)) << endl;
	cout<<"cudaDeviceSynchronize: "<< cudaGetErrorString(cudaDeviceSynchronize())<< endl;
	for(int i = 0;i<100;++i){
		cout<<outGPU[i]<<" "; 
	}
	cout<<endl;
	
	
	
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
