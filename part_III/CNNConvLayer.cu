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
	int outArea  = (FMSIZE/3) * (FMSIZE/3);
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
				outIdx = sli*outArea + fmy*(FMSIZE/3) + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int* dev_inNeuCooNNZ, int* dev_inNeuCooData, 
					int* dev_filtCooNNZ, int* dev_filtCooData,
					int* dev_outGPU, int* dev_outNeu)
{
	
	// ------------------------------------------------------------------------------
	//   Variables declaration
	// ------------------------------------------------------------------------------
	
	const int
		bid= blockIdx.x,
		tid= threadIdx.x,
		PAD_WIDTH= FILTSIZE/2,
		PAD_SIZE= FMSIZE+ PAD_WIDTH* 2;
	int 
		offset,
		fm_offset,
		filt_nnz,
		fm_nnz,
		col_mask= 0x1F,
		row_mask= 0x3E0,
		tmp;
	char
		filt_data[9],
		fm_data;
	unsigned char 
		filt_row[9],
		filt_col[9],
		fm_row,
		fm_col;
		
	//------------------------------------------------------------------------------
	//	Share memory initialization
	//------------------------------------------------------------------------------
	/*
	 * Clear share memory to zero.
	 */

	__shared__ int out_neu_slice[PAD_SIZE* PAD_SIZE];
	
	if(tid< PAD_SIZE){
		for(int i= 0; i<PAD_SIZE; ++i){
			out_neu_slice[tid* PAD_SIZE+ i]= 0;
		}
	}
	
	__syncthreads();
	
	// ------------------------------------------------------------------------------
	//   Local memory initialization
	// ------------------------------------------------------------------------------
	/*
	 * 1. Extract number of non-zero terms of feature maps and filters.
	 * 2. Put all non-zero term of filters to local memory.
	 * 3. Filp the coordinate of filters.
	 */
	
	// Extract number of non-zero terms of feature maps and filters.
	filt_nnz= tex1Dfetch(filt_nnz_tex, bid* FMDEPTH+ tid+ 1)- tex1Dfetch(filt_nnz_tex, bid* FMDEPTH+ tid);
	fm_nnz= tex1Dfetch(neu_nnz_tex, tid+ 1)- tex1Dfetch(neu_nnz_tex, tid);
	
	// Put all non-zero term of filters to local memory.
	offset= dev_filtCooNNZ[bid* FMDEPTH+ tid];
	for(int i= 0; i< filt_nnz; ++i){
		tmp= tex1Dfetch(filtCooData_tex, i+ offset);
		// Filp the coordinate of filters.
		filt_row[i]= 2- ((tmp& row_mask)>> 5);
		filt_col[i]= 2- tmp& col_mask;
		filt_data[i]= (tmp>> 10)- 10000;
	}
	
	// ------------------------------------------------------------------------------
	//   Correlate feature map and filter.
	// ------------------------------------------------------------------------------
	
	fm_offset= dev_inNeuCooNNZ[tid];
	for(int i = 0; i< fm_nnz; ++i){
		tmp= tex1Dfetch(inNeuCooData_tex, fm_offset+ i);
		fm_row= ((tmp& row_mask)>> 5);
		fm_col= tmp& col_mask;
		fm_data= (tmp>> 10)- 10000;
		
		
		offset= fm_row* PAD_SIZE+ fm_col;
		
		for(int j = 0; j< filt_nnz; ++j){
			atomicAdd(&out_neu_slice[offset+ filt_row[j]* PAD_SIZE+ filt_col[j]], fm_data* filt_data[j]);
		}
	}
	
	__syncthreads();
	
	// ------------------------------------------------------------------------------
	//   Put the value in share memory back to global memory.
	// ------------------------------------------------------------------------------
	
	if(tid< 112){
		for(int i = 0; i< 7; ++i){
			dev_outNeu[bid* FMSIZE* FMSIZE+ i* 112+ tid]= out_neu_slice[(i* 4+ tid/ FMSIZE+ PAD_WIDTH)* PAD_SIZE+ (tid% FMSIZE)+ PAD_WIDTH];
		}
	}
    

}


__global__
void Activation_Pooling_GPU(int* dev_outNeu, int* dev_outGPU){
	const int
		bid = blockIdx.x,							// The depth this block deal with.
		tid = threadIdx.x,							// The filter this thread deal with.
		fm_area = FMSIZE * FMSIZE,					// Area of one feature map.
		out_area = (FMSIZE / 3) * (FMSIZE / 3);
	int
		offset,										// Start point of iteration.
		i, j,										// iterator
		max;										// Temporary storage, for max pooling.

	
	// ------------------------------------------------------------------------------
	//   Activation - ReLU
	// ------------------------------------------------------------------------------
	/* 
	 * Use 28 thread to do ReLU. Each thread scan through a row
	 */
	
	if( tid < FMSIZE) {	
		offset = bid * fm_area + tid * FMSIZE;
		for (i = 0; i< FMSIZE; ++i) {
			if(dev_outNeu[offset+ i]<0) {
				dev_outNeu[offset+ i] = 0;
			}	
		}
		
	}
	
	__syncthreads();
	
	// ------------------------------------------------------------------------------
	//   Max Pooling
	// ------------------------------------------------------------------------------

	int grid_row = tid/ 9;
	int grid_col = tid% 9;
	
	//lu point location
	offset = bid * fm_area + grid_row * 3 * FMSIZE + grid_col * 3;
	max = dev_outNeu[offset];
	
	for (j = 0 ; j < 3 ; ++j) { 
		for (int k = 0 ; k < 3 ; ++k) {
			if (dev_outNeu[offset + j* FMSIZE + k]>=max) {
				max = dev_outNeu[offset + j* FMSIZE + k];
			}
		}	
	}
	dev_outGPU[bid* out_area+ tid] = max;
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
	
	initGPU();
	dim3 numBlocks(FILTNUM);
	dim3 threadsPerBlock(FMDEPTH);
	
	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	convLayerGPU<<<numBlocks, threadsPerBlock>>>(dev_inNeuCooNNZ, dev_inNeuCooData, 
												dev_filtCooNNZ, dev_filtCooData,
												dev_outGPU, dev_outNeu);
    
	cudaDeviceSynchronize();
	
	Activation_Pooling_GPU<<<FILTNUM, (FMSIZE/3) * (FMSIZE/3)>>>(dev_outNeu, dev_outGPU);

	
	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;

	cout<< cudaGetErrorString(cudaMemcpy(outGPU, dev_outGPU,
								sizeof(int)* (FILTNUM * (FMSIZE/3) * (FMSIZE/3)),
								cudaMemcpyDeviceToHost))<<endl;
	
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
