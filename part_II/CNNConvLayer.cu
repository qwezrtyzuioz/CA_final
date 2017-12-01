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
void convLayerGPU(int* dev_inNeuCooNNZ, unsigned char* dev_inNeuCooRow, 
					unsigned char* dev_inNeuCooCol, int* dev_inNeuCooData, 
					int* dev_filtCooNNZ, unsigned char* dev_filtCooRow, 
					unsigned char* dev_filtCooCol, int* dev_filtCooData,
					int* dev_outGPU, int* dev_outNeu)
{
	
	// ------------------------------------------------------------------------------
	//   Variables declaration
	// ------------------------------------------------------------------------------
	
	const int
		bid= blockIdx.x,
		tid= threadIdx.x;
	int 
		index,
		offset,
		fm_offset,
		filt_nnz,
		fm_nnz,
		filt_data[18],
		fm_data;
	unsigned char 
		filt_row[18],
		filt_col[18],
		fm_row,
		fm_col;
		
	//------------------------------------------------------------------------------
	//	Local memory initialization
	//------------------------------------------------------------------------------
	
	__shared__ int out_neu_slice[31* 31];
	
	if(tid< 31){
		for(int i= 0; i<31; ++i){
			out_neu_slice[tid* 31+ i]= 0;
		}
	}
	
	__syncthreads();
	
	// ------------------------------------------------------------------------------
	//   Local memory initialization
	// ------------------------------------------------------------------------------
	
	filt_nnz= dev_filtCooNNZ[bid* FMDEPTH+ tid+ 1]- dev_filtCooNNZ[bid* FMDEPTH+ tid];
	fm_nnz= dev_inNeuCooNNZ[tid+ 1]- dev_inNeuCooNNZ[tid];
	
	
	offset= dev_filtCooNNZ[bid* FMDEPTH+ tid];
	for(int i= 0; i< filt_nnz; ++i){
		index= i+ offset;
		filt_row[i]= 4- int(dev_filtCooRow[index]);
		filt_col[i]= 4- int(dev_filtCooCol[index]);
		filt_data[i]= dev_filtCooData[index];
	}
	
	
	fm_offset= dev_inNeuCooNNZ[tid];
	for(int i = 0; i< fm_nnz; ++i){
		index= fm_offset+ i;
		fm_row= int(dev_inNeuCooRow[index]);
		fm_col= int(dev_inNeuCooCol[index]);
		fm_data= dev_inNeuCooData[index];
		
		offset= int(fm_row)* 31+ int(fm_col);
		
		for(int j = 0; j< filt_nnz; ++j){
			atomicAdd(&out_neu_slice[offset+ filt_row[j]* 31+ filt_col[j]], fm_data* filt_data[j]);
		}
	}
	
	__syncthreads();
	
	if(tid< FMSIZE){
		for(int i = 0; i< FMSIZE; ++i){
			dev_outNeu[bid* FMSIZE* FMSIZE+ tid* FMSIZE+ i]= out_neu_slice[(tid+ 2)* 31+ i+ 2];
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
	 * Use 27 thread to do ReLU. Each thread scan through a row
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
			if (dev_outNeu[ offset + j*FMSIZE + k ]>=max) {
				max = dev_outNeu[ offset + j * FMSIZE + k ];
			}
		}	
	}
	dev_outGPU[bid * out_area + tid] = max;
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

	//Check GPU connection
	cout<< endl;
	
	const int num=100;
    int* g;
    
    int a[num], b[num];
    for(int k=0; k<num; k++){
            a[k]=k;
            b[k]=0;
    }
	
    cout<< "cudaMalloc : "<< cudaGetErrorString(cudaMalloc((void**) &g, sizeof(int)*num))<< endl;
    cout<< "cudaMemcpy a => g : "<< cudaGetErrorString(cudaMemcpy(g, a, sizeof(int)*num, cudaMemcpyHostToDevice))<< endl;
    cout<< "cudaMemcpy g => b : "<< cudaGetErrorString(cudaMemcpy(b, g, sizeof(int)*num, cudaMemcpyDeviceToHost)) << endl;
    
    for(int k=0; k<num; k++){
            if(a[k]!=b[k]){
                    cout << "Fail to ";
                    break;
            }
    }
    cout<< "Connect"<< endl<< endl;

	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);
	/***	Lunch your CUDA Kernel here	***/
	
	initGPU();
	dim3 numBlocks(FILTNUM);
	dim3 threadsPerBlock(FMDEPTH);
	convLayerGPU<<<numBlocks, threadsPerBlock>>>(dev_inNeuCooNNZ, dev_inNeuCooRow, 
												dev_inNeuCooCol, dev_inNeuCooData, 
												dev_filtCooNNZ, dev_filtCooRow, 
												dev_filtCooCol, dev_filtCooData,
												dev_outGPU, dev_outNeu);
    
	cudaDeviceSynchronize();
	
	Activation_Pooling_GPU<<<FILTNUM, 81>>>(dev_outNeu, dev_outGPU);
	cudaMemcpy(outGPU, dev_outGPU,
			sizeof(int)* FILTNUM * FMSIZE/3 * FMSIZE/3,
			cudaMemcpyDeviceToHost);
			
	//int* test= new int[sizeof(int)* (FILTNUM * FMSIZE * FMSIZE)];
	//cout<< cudaGetErrorString(cudaMemcpy(test, dev_outNeu, sizeof(int)* (FILTNUM * FMSIZE/3 * FMSIZE/3), cudaMemcpyDeviceToHost))<<endl;
	
	//unsigned char* test= new unsigned char[inNeuCooNNZ[FMDEPTH]];
	//cout<< cudaGetErrorString(cudaMemcpy(test, dev_inNeuCooRow, sizeof(unsigned char)* inNeuCooNNZ[FMDEPTH], cudaMemcpyDeviceToHost))<<endl;
	//	
	//
	//cout<< endl;
	//for(int i= 0; i< 100; ++i){
	//	cout<< int(test[i])<< " ";
	//	cout<< int(inNeuCooRow[i]) << "  ";
	//}
	//cout<<endl;
	
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
