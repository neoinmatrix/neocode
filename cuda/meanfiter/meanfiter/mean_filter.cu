// one dimension of mean filter designed and coded by neo 
/*
题目：
均值滤波器
对于一维数组进行均值滤波，
就是每个点的数据为，以此点为中心半径为r的数组单元中
取值并且，求平均值，以过滤数据中波动大的数据

要求:
1.C 语言实现代码
2.Cuda 语言实现代码
3.shared memory的使用
4.误差小数据访问不会越界
5.计算运行时间和计算误差
6.支持大数据的处理

thinking: 

the data of margin side can be dealed by this  (i-j+n)%n 
shaped the array  like circle 

in the same block ,the threads  visit the  data range in [r-i r r+i]
so copy global memory to shared memory to boost the speed

the shared memory is 48KB 
so the num of radius    3*r<= (48KB/4B)  =>  r <= 4K 

test data: 
10 3 1
100000 100 0
100000 50 0

*/
#include "cuda_runtime.h"
#include "cuda.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>

#define  MIN(a,b) (a<b?a:b)
#define MAX_BLOCK 1024   //  biggest block numbers 
using namespace std;

//just with gpu  
__global__ void calcWithGPU_filter(float *b, const float *a, const int n,const int r){
	//__shared__ mean   we can add the shared memory to boost calc
	int g = blockIdx.x;
	int t = threadIdx.x;
	int i_global,i_inner;
	i_inner=g*r+t;                   //将分配到 block 中的 thread 号 计算出 线程对应的 均值滤波点的位置
	if(i_inner>=n){                  //越界处理
		return ;
	}
	float sum=0;
	for(int i=-r;i<=r;i++){         //计算中值滤波的结果  取 2*r+1 个数的均值
		i_global=(i_inner+i+n)%n;   //我觉的可以 取模 将均值滤波的 边缘数据 头尾相接 进行滤波
		sum+=a[i_global];
	}
	b[i_inner]=sum/(2*r+1);
}
 
//using the shared memory to save time 
__global__ void calcWithGPU_filter_shared(float *b, const float *a, const int n,const int r){
	//__shared__ mean   we can add the shared memory to boost calc
	int g = blockIdx.x;
	int t = threadIdx.x;
	int i_global,i_inner;
	extern __shared__ float cache[]; //dynamic shared memory allocation
	i_inner=g*r+t;  

	// copy 数据 从global 中 到 shared memory 中  每个位置需要对称复制三次 
	i_global=(i_inner-r+n)%n; 
	cache[t]=a[i_global];
	i_global=(i_inner+n)%n; 
	cache[t+r]=a[i_global];
	i_global=(i_inner+r+n)%n; 
	cache[t+2*r]=a[i_global];

	__syncthreads();

	if(i_inner>=n){                            
		return ;
	}
	float sum=0;
	for(int i=-r;i<=r;i++){         //计算中值滤波的结果  取 2*r+1 个数的均值
		sum+=cache[r+t+i];
	}
	b[i_inner]=sum/(2*r+1);
}

//using the shared memory to save time 
__global__ void calcWithGPU_filter_shared_bd(float *b, const float *a, const int n,const int r){
	//__shared__ mean   we can add the shared memory to boost calc
	int g = blockIdx.x;
	int t = threadIdx.x;
	int i_global,i_inner;

	extern __shared__ float cache[]; //dynamic shared memory allocation
	
	int blocknum=r*MAX_BLOCK;    //the turns of  r*MAX_BLOCK 
	for(i_inner=g*r+t;i_inner<n+blocknum;i_inner=i_inner+blocknum){
		// copy 数据 从global 中 到 shared memory 中  每个位置需要对称复制三次 
		i_global=(i_inner-r+n)%n; 
		cache[t]=a[i_global];
		i_global=(i_inner+n)%n; 
		cache[t+r]=a[i_global];
		i_global=(i_inner+r+n)%n; 
		cache[t+2*r]=a[i_global];

		__syncthreads();

		if(i_inner>=n){                            
			return ;
		}
		float sum=0;
		for(int i=-r;i<=r;i++){         //calc the sum of 2*r+1 data
			sum+=cache[r+t+i];
		}
		b[i_inner]=sum/(2*r+1);        // get the result of mean filter

		__syncthreads();
	}
	
}

//cpu mean filter process in detail
void calcWithCPU_filter(float *b, const float *a, const int n,const int r){
	if(r>n)       //can not calc
		return ;
	int i,j,index;
	float sum=0;
	for(i=0;i<n;i++){
		sum=0;
		for(j=-r;j<=r;j++){
			index=(i+j+n)%n;  //我觉的可以 取模 将均值滤波的 边缘数据 头尾相接 进行滤波
			sum+=a[index];
		}
		b[i]=sum/(2*r+1);
	}
}

//calc the error between cpu and gpu data
void calcErrorBetweenData(const float *b, const float *a, const int n,const int type){
	float error=0.0f;
	float tmp;
	ofstream file;
	if(type>0){
		if(type==1){
			file.open("result_nobd.txt");
			file.clear();
		}
		if(type==2){
			file.open("result_bd.txt");
			file.clear();
		}
	}
	for(int i=0;i<n;i++){
		tmp=b[i]-a[i];
		error+=tmp;
		if(type>0){
			file<<i<<" "<<b[i]<<" "<<a[i]<<endl;
		}
	}
	if(type>0){
		file.close();
	}
	printf(" the error between two data : %.3f \n",error);
}

//data to print
void print_data(float *b,  float *a, const int n,const int r){
	for(int i=0;i<n;i++){
		printf(" %f %f \n",a[i],b[i]);
	}
}

//cpu block to filter data
void process_cpu(float *b,  float *a, const int n,const int r, double *time,const bool print=true){
	double  duration;  
	clock_t begin, end;  
	begin = clock();  
	calcWithCPU_filter(b,a,n,r);
	end = clock();  
	duration = (double)( end -begin )*1000  / CLOCKS_PER_SEC;
	printf(" cpu mean_filter result: \n");
	(*time)=duration;
	if(print){
		printf(" [data] [filter] \n");
		print_data(b,a,n,r);
	}
	printf(" Time elapsed :  %3.3f ms \n", duration);
}

// gpu block to filter data   contains two branches  the filter without shared memory  or without shared memory
void process_gpu(float *b,  float *a, const int n,const int r ,int type, double *time,const bool print=true){
	float *dev_a,*dev_b;
	cudaEvent_t     start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	cudaMalloc((void**)&dev_a, n * sizeof(float));
	cudaMalloc((void**)&dev_b, n * sizeof(float));
	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	if(type==1)
		calcWithGPU_filter<<<MIN(((n+r-1)/r),MAX_BLOCK),r>>>(dev_b, dev_a, n, r);
	if(type==2)
		calcWithGPU_filter_shared<<<MIN(((n+r-1)/r),MAX_BLOCK),r,r*3*sizeof(float)>>>(dev_b, dev_a, n, r);
	if(type==3)
		calcWithGPU_filter_shared_bd<<<MIN(((n+r-1)/r),MAX_BLOCK),r,r*3*sizeof(float)>>>(dev_b, dev_a, n, r);

	cudaMemcpy(b, dev_b, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float   elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf(" cuda mean_filter result: \n");
	(*time)=elapsedTime;
	if(print){
		printf(" [data] [filter] \n");
		print_data(b,a,n,r);
	}
	printf( " Time elapsed :  %3.3f ms \n", elapsedTime );
}

int main(){
	int n,r,debug_status;
	bool debug=false;
	printf("input the n and r [or is_debug(1)]to filter data : ");
	scanf(" %d %d %d",&n,&r,&debug_status);                           //n sizes of array      r the radius of filter
	float *a,*b,*c,*d,*e;
	double time_cpu,time_gpu_1,time_gpu_2,time_gpu_3;
	a=(float*)malloc(sizeof(float)*n);
	b=(float*)malloc(sizeof(float)*n);
	c=(float*)malloc(sizeof(float)*n);
	d=(float*)malloc(sizeof(float)*n);
	e=(float*)malloc(sizeof(float)*n);

	//srand((unsigned)time(NULL));/*播种子*/
	for(int i=0;i<n;i++){ 
		if(debug_status>0){
			a[i]=(float)i;
		}else{
			a[i]=rand()%RAND_MAX %100 ;
		}
		//a[i]=powf(a[i],7);
	}
	debug=(debug_status>0)?true:false;
	//CPU===============================================================
	process_cpu(b,a,n,r,&time_cpu,debug);
	//GPU===============================================================
	process_gpu(c,a,n,r,1,&time_gpu_1,debug);     // without shared_memory 
	process_gpu(d,a,n,r,2,&time_gpu_2,debug);     // with shared_memory
	process_gpu(e,a,n,r,3,&time_gpu_3,debug);     // with shared_memory big data
	//printf( "\n  %f %f %f \n",time_cpu,time_gpu_1,time_gpu_2);

	//printf( "\n the error: [ cpu mean filter data / cuda mf data without bd ] \n");
	//calcErrorBetweenData(b,d,n);
	printf( "\n the error: [ cpu mean filter data / cuda mf data without bd] \n");
	calcErrorBetweenData(b,d,n,1);
	printf( "\n the error: [ cpu mean filter data / cuda mf data with bd] \n");
	calcErrorBetweenData(b,e,n,2);

	printf("\n speedup rate:\n");
	if(time_cpu>time_gpu_1 &&time_cpu!=0&&time_gpu_1!=0){
		printf( " cuda without shared memory => speedup rate :  %d:1  \n", (int)ceil(time_cpu/time_gpu_1 ));
	}else{
		printf( " cuda without shared memory => no speed up \n");
	}

	if(time_cpu>time_gpu_2 &&time_cpu!=0&&time_gpu_2!=0){
		printf( " cuda with shared memory => speedup rate :  %d:1  \n", (int)ceil(time_cpu/time_gpu_3 ));
	}else{
		printf( " cuda with shared memory => no speed up \n");
	}
	
	if(time_gpu_1>time_gpu_2 &&time_gpu_1!=0&&time_gpu_2!=0){
		printf( " cuda without shared memory / cuda with shared memory => speedup rate :  %d:1  \n", (int)ceil(time_gpu_1/time_gpu_3 ));
	}else{
		printf( " cuda without shared memory / cuda with shared memory => no speed up \n");
	}
	
	
	return 0;
}
