#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define FMSIZE 28
#define FMDEPTH 192
#define FILTSIZE 3
#define FILTNUM 256
#define STRIDE 1

int*inNeu;
int*filt;
int*outNeu;
int*outCPU;
int*outGPU;

int *filtCooNNZ;
int *filtCooData;
unsigned char *filtCooRow;
unsigned char *filtCooCol;
int *tmp_filtCooData;
unsigned char *tmp_filtCooRow;
unsigned char *tmp_filtCooCol;

int *inNeuCooNNZ;
int *inNeuCooData;
unsigned char *inNeuCooRow;
unsigned char *inNeuCooCol;
int *tmp_inNeuCooData;
unsigned char *tmp_inNeuCooRow;
unsigned char *tmp_inNeuCooCol;

int *dev_inNeuCooNNZ;
unsigned char *dev_inNeuCooRow;
unsigned char *dev_inNeuCooCol;
int *dev_inNeuCooData;
int *dev_filtCooNNZ;
int *dev_filtCooData;
unsigned char *dev_filtCooRow;
unsigned char *dev_filtCooCol;
int *dev_filt;
int *dev_outGPU; 	  
int *dev_outNeu;

string 
	filt_path= "data/filter.txt",
	neur_path= "data/neuron.txt",
	filt_COO_path= "data/filt_COO_irregular.txt",
	neur_COO_path= "data/neuron_COO_irregular.txt";

void init()
{
	ifstream ifs;
	string str;
	
	int inNeuIdx, filtIdx;
	int tmp;
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;
	int outVol = FILTNUM * FMSIZE/3 * FMSIZE/3;
	
	inNeu = new int[FMSIZE*FMSIZE*FMDEPTH]();
	ifs.open(neur_path.c_str(), ifstream::in);
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
	ifs.open(filt_path.c_str(), ifstream::in);
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

void initCoo()
{
	int i, j, k, idx;
	int tmp, nnz;
	string str;
	int current_nnz;
	fstream ifs;

	filtCooNNZ = new int [FILTNUM* FMDEPTH+ 1];

	ifs.open(filt_COO_path.c_str(), ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the filters input file\n";
		exit(-1);
	}
	current_nnz=0;
	
	for(i = 0; i < FILTNUM; i++){
		ifs >> str; 
		for(j = 0; j < FMDEPTH; j++){
			ifs >> str; 
			ifs >> str >> nnz; 
			idx = i*FMDEPTH + j;
			filtCooNNZ[idx] = current_nnz;
			
			if(i == 0 && j==0){
				filtCooData = new int [nnz];
				filtCooRow  = new unsigned char [nnz];
				filtCooCol  = new unsigned char [nnz];
			}
			else{
				tmp_filtCooData = new int [current_nnz];
				tmp_filtCooRow  = new unsigned char [current_nnz];
				tmp_filtCooCol  = new unsigned char [current_nnz];
			
				memcpy(tmp_filtCooData,filtCooData,sizeof(int)*current_nnz);
				memcpy(tmp_filtCooRow,filtCooRow,sizeof(unsigned char)*current_nnz);
				memcpy(tmp_filtCooCol,filtCooCol,sizeof(unsigned char)*current_nnz);
				
				delete [] filtCooData;
				delete [] filtCooRow;
				delete [] filtCooCol;
				
				filtCooData = new int [current_nnz+nnz];
				filtCooRow = new unsigned char [current_nnz+nnz];
				filtCooCol = new unsigned char [current_nnz+nnz];
				
				memcpy(filtCooData,tmp_filtCooData,sizeof(int)*current_nnz);
				memcpy(filtCooRow,tmp_filtCooRow,sizeof(unsigned char)*current_nnz);
				memcpy(filtCooCol,tmp_filtCooCol,sizeof(unsigned char)*current_nnz);
			}
			
			ifs >> str ;
			for(k = 0; k < nnz; k++){
				ifs >> tmp;
				idx = current_nnz + k;
				filtCooData[idx] = tmp;
				
			}
			ifs >> str ;
			for(k = 0; k < nnz; k++){
				ifs >> tmp;
				idx =current_nnz +k;
				filtCooRow[idx] = tmp;
			}
			ifs >> str ;
			for(k = 0; k < nnz; k++){
				ifs >>  tmp;
				idx = current_nnz+ k;
				filtCooCol[idx] = tmp;
			}
			
			for(k = 0; k < nnz; k++){
				idx = current_nnz + k;
				filtCooData[idx]+= 10000;
				filtCooData[idx]= filtCooData[idx]<< 5;
				filtCooData[idx]+= filtCooRow[idx];
				filtCooData[idx]= filtCooData[idx]<< 5;
				filtCooData[idx]+= filtCooCol[idx];
				
			}
	
			// cout << "filtCooNNZ[" << i*FMDEPTH + j << "] =" << filtCooNNZ[i*FMDEPTH + j] << endl;
			// for(k = 0; k < nnz; k++){
				// cout << "filtCooData[" << current_nnz+k << "] =" << filtCooData[current_nnz+k] << endl;
				// cout << "filtCooRow[" << current_nnz+k << "] =" << filtCooRow[current_nnz+k] << endl;
				// cout << "filtCooCol[" << current_nnz+k << "] ="<< filtCooCol[current_nnz+k] << endl;
			// }
			
			current_nnz=current_nnz+nnz;
		
		}
	}
	filtCooNNZ[FILTNUM* FMDEPTH] = current_nnz;
	ifs.close();
	
	
	

	current_nnz=0;
	inNeuCooNNZ = new int [FMDEPTH+ 1];

	ifs.open(neur_COO_path.c_str(), ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the neurons input file\n";
		exit(-1);
	}
	for(i = 0; i < FMDEPTH ; i++){
		ifs >> str; 
		ifs >> str >> nnz; 
		inNeuCooNNZ[i] = current_nnz;
		
		if(i == 0){
				inNeuCooData = new int [nnz];
				inNeuCooRow  = new unsigned char [nnz];
				inNeuCooCol  = new unsigned char [nnz];
		}
		else{
				tmp_inNeuCooData = new int [current_nnz];
				tmp_inNeuCooRow  = new unsigned char [current_nnz];
				tmp_inNeuCooCol  = new unsigned char [current_nnz];
			
				memcpy(tmp_inNeuCooData , inNeuCooData , sizeof(int)*current_nnz);
				memcpy(tmp_inNeuCooRow  , inNeuCooRow  , sizeof(unsigned char)*current_nnz);
				memcpy(tmp_inNeuCooCol  , inNeuCooCol  , sizeof(unsigned char)*current_nnz);
				
				delete [] inNeuCooData ;
				delete [] inNeuCooRow  ;
				delete [] inNeuCooCol  ;
				
				inNeuCooData = new int [current_nnz+nnz];
				inNeuCooRow  = new unsigned char [current_nnz+nnz];
				inNeuCooCol  = new unsigned char [current_nnz+nnz];
				
				memcpy(inNeuCooData , tmp_inNeuCooData ,sizeof(int)*current_nnz);
				memcpy(inNeuCooRow  , tmp_inNeuCooRow  ,sizeof(unsigned char)*current_nnz);
				memcpy(inNeuCooCol  , tmp_inNeuCooCol  ,sizeof(unsigned char)*current_nnz);
		}

		ifs >> str;
		for(j = 0; j < nnz; j++){
			ifs >> tmp;
			idx = current_nnz + j;
			inNeuCooData[idx] = tmp;
		}
		ifs >> str;
		for(j = 0; j < nnz; j++){
			ifs >> tmp;
			idx = current_nnz + j;
			inNeuCooRow[idx] = tmp;
		}
		ifs >> str;
		for(j = 0; j < nnz; j++){
			ifs >> tmp;
			idx = current_nnz + j;
			inNeuCooCol[idx] = tmp;
		}
		
		for(j = 0; j < nnz; j++){
			idx = current_nnz + j;
			inNeuCooData[idx]+= 10000;
			inNeuCooData[idx]= inNeuCooData[idx]<< 5;
			inNeuCooData[idx]+= inNeuCooRow[idx];
			inNeuCooData[idx]= inNeuCooData[idx]<< 5;
			inNeuCooData[idx]+= inNeuCooCol[idx];
		}
		
		// cout << "inNeuCooNNZ[" << i << "] =" << inNeuCooNNZ[i] << endl;
		// for(k = 0; k < nnz; k++){
			// cout << "inNeuCooData[" << current_nnz+k << "] =" << inNeuCooData[current_nnz+k] << endl;
			// cout << "inNeuCooRow["  << current_nnz+k << "] =" << inNeuCooRow[current_nnz+k]  << endl;
			// cout << "inNeuCooCol["  << current_nnz+k << "] =" << inNeuCooCol[current_nnz+k]  << endl;
		// }
		
		current_nnz=current_nnz+nnz;
		
	}
	inNeuCooNNZ[FMDEPTH] = current_nnz;
	ifs.close();
	
	
	
	
}

void ending()
{
	delete [] filt;
	delete [] inNeu;
	delete [] outNeu;
	delete [] outCPU;
	delete [] outGPU;
	
	delete [] filtCooNNZ;
	delete [] filtCooData;
	delete [] filtCooRow;
	delete [] filtCooCol;
	delete [] tmp_filtCooData;
	delete [] tmp_filtCooRow;
	delete [] tmp_filtCooCol;

	delete [] inNeuCooNNZ;
	delete [] inNeuCooData;
	delete [] inNeuCooRow;
	delete [] inNeuCooCol;
	delete [] tmp_inNeuCooData;
	delete [] tmp_inNeuCooRow;
	delete [] tmp_inNeuCooCol;
	
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

void initGPU(){
	 
	cout<< cudaGetErrorString(cudaMalloc(&dev_inNeuCooNNZ,  sizeof(int)* (FMDEPTH+ 1)))<< endl;
	//cout<< cudaGetErrorString(cudaMalloc(&dev_inNeuCooRow,  sizeof(unsigned char)* inNeuCooNNZ[FMDEPTH]))<< endl;
	//cout<< cudaGetErrorString(cudaMalloc(&dev_inNeuCooCol,  sizeof(unsigned char)* inNeuCooNNZ[FMDEPTH]))<< endl;
	cout<< cudaGetErrorString(cudaMalloc(&dev_inNeuCooData, sizeof(int)* inNeuCooNNZ[FMDEPTH]))<< endl;
	cout<< cudaGetErrorString(cudaMalloc(&dev_filtCooNNZ,   sizeof(int)* (FILTNUM* FMDEPTH+ 1)))<< endl;
	//cout<< cudaGetErrorString(cudaMalloc(&dev_filtCooRow,   sizeof(unsigned char)* filtCooNNZ[FILTNUM* FMDEPTH]))<< endl;
	//cout<< cudaGetErrorString(cudaMalloc(&dev_filtCooCol,   sizeof(unsigned char)* filtCooNNZ[FILTNUM* FMDEPTH]))<< endl;
	cout<< cudaGetErrorString(cudaMalloc(&dev_filtCooData,  sizeof(int)* filtCooNNZ[FILTNUM* FMDEPTH]))<< endl;
	cout<< cudaGetErrorString(cudaMalloc(&dev_outGPU, 	  	sizeof(int)* (FILTNUM * (FMSIZE/3) * (FMSIZE/3))))<< endl;
	cout<< cudaGetErrorString(cudaMalloc(&dev_outNeu, 	  	sizeof(int)* (FILTNUM * FMSIZE * FMSIZE)))<< endl;
	
	cout<< cudaGetErrorString(cudaMemcpy(dev_inNeuCooNNZ,  inNeuCooNNZ,  sizeof(int)* (FMDEPTH+ 1),  							cudaMemcpyHostToDevice))<< endl;
	cout<< cudaGetErrorString(cudaMemcpy(dev_inNeuCooData, inNeuCooData, sizeof(int)* inNeuCooNNZ[FMDEPTH], 					cudaMemcpyHostToDevice))<< endl;
	//cout<< cudaGetErrorString(cudaMemcpy(dev_inNeuCooRow,  inNeuCooRow,  sizeof(unsigned char)* inNeuCooNNZ[FMDEPTH],  			cudaMemcpyHostToDevice))<< endl;
	//cout<< cudaGetErrorString(cudaMemcpy(dev_inNeuCooCol,  inNeuCooCol,  sizeof(unsigned char)* inNeuCooNNZ[FMDEPTH],  			cudaMemcpyHostToDevice))<< endl;
	cout<< cudaGetErrorString(cudaMemcpy(dev_filtCooNNZ,   filtCooNNZ,   sizeof(int)* (FILTNUM* FMDEPTH+ 1),  					cudaMemcpyHostToDevice))<< endl;
	cout<< cudaGetErrorString(cudaMemcpy(dev_filtCooData,  filtCooData,  sizeof(int)* filtCooNNZ[FILTNUM* FMDEPTH], 	   		cudaMemcpyHostToDevice))<< endl;
	//cout<< cudaGetErrorString(cudaMemcpy(dev_filtCooRow,   filtCooRow,   sizeof(unsigned char)* filtCooNNZ[FILTNUM* FMDEPTH],  	cudaMemcpyHostToDevice))<< endl;
	//cout<< cudaGetErrorString(cudaMemcpy(dev_filtCooCol,   filtCooCol,   sizeof(unsigned char)* filtCooNNZ[FILTNUM* FMDEPTH],  	cudaMemcpyHostToDevice))<< endl;
	
	//cout<< cudaGetErrorString(cudaMemset(dev_outNeu, 0, sizeof(int)* (FILTNUM * FMSIZE * FMSIZE)))<< endl;
	//cout<< "End of GPU initialization" << endl<< endl;
}
