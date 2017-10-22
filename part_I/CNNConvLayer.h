#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define FMSIZE 27
#define FMDEPTH 96
#define FILTSIZE 5
#define FILTNUM 128
#define STRIDE 1
#define OUTAREA 81
#define POOLSIZE 9

int*inNeu;
int*inNeu2;
int*filt;
int*filt2;
int*outNeu;
int*outNeu2;
int*outCPU;
int*outGPU;

int* dev_ifm;
int* dev_ifilt;
int* dev_outNeu;
int* dev_outGPU;

void init()
{
	ifstream ifs;
	string str;
	
	int inNeuIdx, filtIdx;
	int tmp;
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;
	int outVol = FILTNUM * FMSIZE/3 * FMSIZE/3;
	
	inNeu = new int[FMSIZE*FMSIZE*FMDEPTH]();
	ifs.open("data/neuron.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the neurons input file\n";
	}
	
	for(int i = 0 ; i < FMDEPTH ; i++){
		ifs >> str; 
		for(int j = 0 ; j < FMSIZE ; j++){ 
			for(int k = 0 ; k < FMSIZE ; k++){ 
				ifs >> tmp;
				inNeuIdx = i*FMSIZE*FMSIZE + j*FMSIZE + k;
				inNeu[inNeuIdx] = tmp;
			}
		}
	}
	ifs.close();
				
		
	filt = new int[FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM]();
	ifs.open("data/filter.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the filters input file\n";
	}
	
	for(int i=0 ; i<FILTNUM ; i++){ 
		for(int j = 0 ; j < FMDEPTH ; j++){	
			ifs >> str >> str >> str; 
			for(int k=0 ; k<FILTSIZE ; k++){
				for(int l=0 ; l<FILTSIZE ; l++){
					ifs >> tmp;
					filtIdx = i*FMDEPTH*FILTSIZE*FILTSIZE + j*FILTSIZE*FILTSIZE	+ k*FILTSIZE + l;
					filt[filtIdx] = tmp;
				}
			}
		}	
	}
	ifs.close();

	outNeu = new int[outNeuVol]();
	outCPU = new int[outVol]();
	outGPU = new int[outVol]();

}
void initGPU(){
	 
	cout<<"cudaMalloc: "<< cudaGetErrorString(cudaMalloc(&dev_outNeu, sizeof(int)*FILTNUM * FMSIZE * FMSIZE))<< endl;
	cout<<"cudaMalloc: "<< cudaGetErrorString(cudaMalloc(&dev_ifm, sizeof(int)*FMSIZE*FMSIZE*FMDEPTH))<< endl;
	cout<<"cudaMalloc: "<< cudaGetErrorString(cudaMalloc(&dev_ifilt, sizeof(int)*FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM))<< endl;
	cout<<"cudaMalloc: "<< cudaGetErrorString(cudaMalloc(&dev_outGPU, sizeof(int)*FILTNUM * OUTAREA))<< endl;
	
	
	cout<<"cudaMemcpy: "<< cudaGetErrorString(cudaMemcpy(dev_ifm, inNeu, sizeof(int)*FMSIZE*FMSIZE*FMDEPTH, cudaMemcpyHostToDevice)) <<endl;
	cout<<"cudaMemcpy: "<< cudaGetErrorString(cudaMemcpy(dev_ifilt, filt, sizeof(int)*FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM, cudaMemcpyHostToDevice))<< endl;
	cout<<"cudaMemset: "<< cudaGetErrorString(cudaMemset(dev_outNeu,0,sizeof(int)*FILTNUM * FMSIZE * FMSIZE))<< endl;
	
	outNeu2 = new int[FILTNUM * FMSIZE * FMSIZE]();
	inNeu2 = new int[FMSIZE*FMSIZE*FMDEPTH]();
	filt2 = new int[FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM]();
	
	cout<< "End of GPU initialization" << endl<< endl;
}

void ending()
{
	delete [] filt;
	delete [] inNeu;
	delete [] outNeu;
	delete [] outCPU;
	delete [] outGPU;
}

bool checker(){
	int outVol = FILTNUM * FMSIZE/3 * FMSIZE/3;

	for(int i = 0; i < outVol; i++){ 
		if(  outCPU[i] != outGPU[i]   ){
			cout << "The element: " << i << " is wrong!\n";
			cout << "outCPU[" << i << "] = " << outCPU[i] << endl;
			cout << "outGPU[" << i << "] = " << outGPU[i] << endl;
			return false;
		}
	}
	return true;
}

int timespec_diff_us(timespec& t1, timespec& t2)
{                                                                                
  return (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;        
} 
