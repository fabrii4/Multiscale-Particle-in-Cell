#include <math.h>
#include <iostream>
#include <iomanip>
//#include <string>
#include <fstream>
#include <vector>
#include <ctime> 
#include <stdlib.h> 
#include <chrono>
//#include <algorithm>
//#include <functional>

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>

#include <cufft.h>

#include <cooperative_groups.h>

//#include "Poisson.h"


using namespace std;
//using namespace thrust;
namespace cg = cooperative_groups;

#define MPI 3.1415926535897932385 //greek pi


//typedef struct { float s0; float s1; float s2; float s3;
//                 float s4; float s5; float s6; float s7; } float8;

//static __inline__ __host__ __device__ float8 make_float8(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
//{
//   float8 t; 
//   t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3;
//   t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
//   return t;
//} 

//float3 sum
__device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}
//int3 sum
__device__ int3 operator+(const int3 &a, const int3 &b) {

  return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);

}

//modulus operation
__device__ __host__ int mod(int x, int divisor)
{
    int m = x % divisor;
    return m + (m < 0 ? divisor : 0);
}

//sign operation
__device__ int sign(float x)
{
    return (x >= 0 ? 1 : -1);
}

__device__ float iX(float x, float A, float B, int Nx) //position x to index i
{
   //A=dx; B=xm+dx/2;
   //return mod((int)round((x-B)/A),Nx);
   return (int)round((x-B)/A);
}
__host__ __device__ float xI(int i, float A, float B, int Nx) //index i to position x
{
   //A=dx; B=xm+dx/2;
   //return A*(i%Nx)+B;
   return A*i+B;
}


//cuda error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//cufft error checking
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

#define cufftCheck(err)      __cufftCheck(err, __FILE__, __LINE__)
inline void __cufftCheck(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
									_cudaGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
    }
}

__global__ void ParticleToMesh_ker(float4 *xp, float *rho, int *cell, int Np, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   float cden=1./(Ax*Ay*Az);
   //extern __shared__ float rho_s[];
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   //for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx<Np; idx += blockDim.x*gridDim.x) 
   {
      //the 4th component of x is the charge (this allow coalesced memory access)
      float4 x;
      float3 xc, xcp, axcp;
      int3 ic, scp, ic1, ic2, ic3, ic4, ic5, ic6, ic7;
      //particle coordinates and charge
      x=xp[idx];
      int c=(int)x.w;
      //nearest grid point index ic, position xc, distance xcp
      ic.x=iX(x.x,Ax,Bx,Nx); ic.y=iX(x.y,Ay,By,Ny); ic.z=iX(x.z,Az,Bz,Nz);
      xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
      xcp.x=x.x-xc.x; xcp.y=x.y-xc.y; xcp.z=x.z-xc.z;
      axcp.x=abs(xcp.x); axcp.y=abs(xcp.y); axcp.z=abs(xcp.z);
      scp.x=sign(xcp.x); scp.y=sign(xcp.y); scp.z=sign(xcp.z);
      //other enclosing nearest grid points indices
      ic1.x=ic.x+scp.x; ic1.y=ic.y; ic1.z=ic.z;
      ic2.x=ic.x+scp.x; ic2.y=ic.y+scp.y; ic2.z=ic.z;
      ic3.x=ic.x+scp.x; ic3.y=ic.y+scp.y; ic3.z=ic.z+scp.z;
      ic4.x=ic.x; ic4.y=ic.y+scp.y; ic4.z=ic.z;
      ic5.x=ic.x; ic5.y=ic.y+scp.y; ic5.z=ic.z+scp.z;
      ic6.x=ic.x; ic6.y=ic.y; ic6.z=ic.z+scp.z;
      ic7.x=ic.x+scp.x; ic7.y=ic.y; ic7.z=ic.z+scp.z;

      //calculate cell index
      int i_c=min(ic.x,ic1.x);
      int j_c=min(ic.y,ic4.y);
      int k_c=min(ic.z,ic6.z);
      cell[idx]=i_c*(Ny-1)*(Nz-1)+j_c*(Nz-1)+k_c;
      //particle volume contribution to each nearest grid points
      float a0,a1,a2,a3,a4,a5,a6,a7;
      a0=((Ax-axcp.x)*(Ay-axcp.y)*(Az-axcp.z))*cden*c;
      a1=(axcp.x*(Ay-axcp.y)*(Az-axcp.z))*cden*c;
      a2=(axcp.x*axcp.y*(Az-axcp.z))*cden*c;
      a3=(axcp.x*axcp.y*axcp.z)*cden*c;
      a4=((Ax-axcp.x)*axcp.y*(Az-axcp.z))*cden*c;
      a5=((Ax-axcp.x)*axcp.y*axcp.z)*cden*c;
      a6=((Ax-axcp.x)*(Ay-axcp.y)*axcp.z)*cden*c;
      a7=(axcp.x*(Ay-axcp.y)*axcp.z)*cden*c;

      //charge density at each nearest grid point
      int i0, i1, i2, i3, i4, i5, i6, i7;
      i0=ic.x*Ny*Nz+ic.y*Nz+ic.z; i1=ic1.x*Ny*Nz+ic1.y*Nz+ic1.z;
      i2=ic2.x*Ny*Nz+ic2.y*Nz+ic2.z; i3=ic3.x*Ny*Nz+ic3.y*Nz+ic3.z;
      i4=ic4.x*Ny*Nz+ic4.y*Nz+ic4.z; i5=ic5.x*Ny*Nz+ic5.y*Nz+ic5.z;
      i6=ic6.x*Ny*Nz+ic6.y*Nz+ic6.z; i7=ic7.x*Ny*Nz+ic7.y*Nz+ic7.z;
//if(i0<0 || i0>=Nx*Ny*Nz || i1<0 || i1>=Nx*Ny*Nz || i2<0 || i2>=Nx*Ny*Nz || i3<0 || i3>=Nx*Ny*Nz || i4<0 || i4>=Nx*Ny*Nz || i5<0 || i5>=Nx*Ny*Nz || i6<0 || i6>=Nx*Ny*Nz || i7<0 || i7>=Nx*Ny*Nz)
//   //printf("ERROR! Nrho %d| %d %d %d %d %d %d %d %d\n", Nx*Ny*Nz, i0,i1,i2,i3,i4,i5,i6,i7);
////printf("i %d\n",i0);
      atomicAdd(&rho[i0],a0); atomicAdd(&rho[i1],a1);
      atomicAdd(&rho[i2],a2); atomicAdd(&rho[i3],a3);
      atomicAdd(&rho[i4],a4); atomicAdd(&rho[i5],a5);
      atomicAdd(&rho[i6],a6); atomicAdd(&rho[i7],a7);

//      rho_v[ic[0]*Ny*Nz+ic[1]*Nz+ic[2]]+=a0;
//      rho_v[ic1[0]*Ny*Nz+ic1[1]*Nz+ic1[2]]+=a1;
//      rho_v[ic2[0]*Ny*Nz+ic2[1]*Nz+ic2[2]]+=a2;
//      rho_v[ic3[0]*Ny*Nz+ic3[1]*Nz+ic3[2]]+=a3;
//      rho_v[ic4[0]*Ny*Nz+ic4[1]*Nz+ic4[2]]+=a4;
//      rho_v[ic5[0]*Ny*Nz+ic5[1]*Nz+ic5[2]]+=a5;
//      rho_v[ic6[0]*Ny*Nz+ic6[1]*Nz+ic6[2]]+=a6;
//      rho_v[ic7[0]*Ny*Nz+ic7[1]*Nz+ic7[2]]+=a7;
   }
}

void ParticleToMesh(float4 *xp, float *rho, int *cell, int Np, int3 N, float3 A, float3 B)
{
   int threads=256;
   int blocks=Np/threads+1;
   ParticleToMesh_ker<<<blocks,threads>>>(xp, rho, cell, Np, N, A, B);
}



__global__ void MeshToParticle_ker(float4 *xp, float *fp, float *fg, int Np, int3 N, float3 A, float3 B, int sgn)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   float cden=1./(Ax*Ay*Az);
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   //for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx<Np; idx += blockDim.x*gridDim.x) 
   {
      //the 4th component of x is the charge (this allow coalesced memory access)
      float4 x;
      float3 xc, xcp, axcp;
      int3 ic, scp, ic1, ic2, ic3, ic4, ic5, ic6, ic7;
      //particle coordinates and charge
      x=xp[idx];
      //int c=(int)x.w;
      //nearest grid point index ic, position xc, distance, xcp
      ic.x=iX(x.x,Ax,Bx,Nx); ic.y=iX(x.y,Ay,By,Ny); ic.z=iX(x.z,Az,Bz,Nz);
      xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
      xcp.x=x.x-xc.x; xcp.y=x.y-xc.y; xcp.z=x.z-xc.z;
      axcp.x=abs(xcp.x); axcp.y=abs(xcp.y); axcp.z=abs(xcp.z);
      scp.x=sign(xcp.x); scp.y=sign(xcp.y); scp.z=sign(xcp.z);
      //other enclosing nearest grid points indices
      ic1.x=ic.x+scp.x; ic1.y=ic.y; ic1.z=ic.z;
      ic2.x=ic.x+scp.x; ic2.y=ic.y+scp.y; ic2.z=ic.z;
      ic3.x=ic.x+scp.x; ic3.y=ic.y+scp.y; ic3.z=ic.z+scp.z;
      ic4.x=ic.x; ic4.y=ic.y+scp.y; ic4.z=ic.z;
      ic5.x=ic.x; ic5.y=ic.y+scp.y; ic5.z=ic.z+scp.z;
      ic6.x=ic.x; ic6.y=ic.y; ic6.z=ic.z+scp.z;
      ic7.x=ic.x+scp.x; ic7.y=ic.y; ic7.z=ic.z+scp.z;
      //particle volume contribution to each nearest grid points
      float a0,a1,a2,a3,a4,a5,a6,a7;
      a0=((Ax-axcp.x)*(Ay-axcp.y)*(Az-axcp.z))*cden;
      a1=(axcp.x*(Ay-axcp.y)*(Az-axcp.z))*cden;
      a2=(axcp.x*axcp.y*(Az-axcp.z))*cden;
      a3=(axcp.x*axcp.y*axcp.z)*cden;
      a4=((Ax-axcp.x)*axcp.y*(Az-axcp.z))*cden;
      a5=((Ax-axcp.x)*axcp.y*axcp.z)*cden;
      a6=((Ax-axcp.x)*(Ay-axcp.y)*axcp.z)*cden;
      a7=(axcp.x*(Ay-axcp.y)*axcp.z)*cden;

      //field density at each nearest grid point to particle
      int i0, i1, i2, i3, i4, i5, i6, i7;
      i0=ic.x*Ny*Nz+ic.y*Nz+ic.z; i1=ic1.x*Ny*Nz+ic1.y*Nz+ic1.z;
      i2=ic2.x*Ny*Nz+ic2.y*Nz+ic2.z; i3=ic3.x*Ny*Nz+ic3.y*Nz+ic3.z;
      i4=ic4.x*Ny*Nz+ic4.y*Nz+ic4.z; i5=ic5.x*Ny*Nz+ic5.y*Nz+ic5.z;
      i6=ic6.x*Ny*Nz+ic6.y*Nz+ic6.z; i7=ic7.x*Ny*Nz+ic7.y*Nz+ic7.z;
      float fgt= fg[i0]*a0+fg[i1]*a1+fg[i2]*a2+fg[i3]*a3+
                 fg[i4]*a4+fg[i5]*a5+fg[i6]*a6+fg[i7]*a7;
      fp[idx] += sgn*fgt;
   }
}

void MeshToParticle(float4 *xp, float *fp, float *fg, int Np, int3 N, float3 A, float3 B, int sgn)
{
   int threads=256;
   int blocks=Np/threads+1;
   MeshToParticle_ker<<<blocks,threads>>>(xp, fp, fg, Np, N, A, B, sgn);
}

struct compare_x
{
  __host__ __device__
  bool operator()(float4 lhs, float4 rhs)
  {
    return lhs.x < rhs.x;
  }
};

struct compare_y
{
  __host__ __device__
  bool operator()(float4 lhs, float4 rhs)
  {
    return lhs.y < rhs.y;
  }
};

struct compare_z
{
  __host__ __device__
  bool operator()(float4 lhs, float4 rhs)
  {
    return lhs.z < rhs.z;
  }
};

void init_mesh_struct(thrust::device_vector<float4>& xp_d, int Np, float3& A, float3& B, int3& N, float fgrid)
{
   //cpu min Max
   thrust::host_vector<float4> xp(xp_d);
   float4 xt=xp[0];
   float xM, xm=xM=xt.x;
   float yM, ym=yM=xt.y;
   float zM, zm=zM=xt.z;
   for(int i=1;i<Np;i++)
   {
      xt=xp[i];
      float x=xt.x;
      float y=xt.y;
      float z=xt.z;
      if(x<xm) xm=x;
      if(x>xM) xM=x;
      if(y<ym) ym=y;
      if(y>yM) yM=y;
      if(z<zm) zm=z;
      if(z>zM) zM=z;
   }

//   //gpu min Max (slower!)
//   thrust::device_vector<float4> xp_d(xp);
//   thrust::pair<thrust::device_vector<float4>::iterator, thrust::device_vector<float4>::iterator > minmax;
//   minmax = minmax_element(thrust::device, xp_d.begin(), xp_d.end(), compare_x());
//   int mx=((float4)(*minmax.first)).x;
//   int Mx=((float4)(*minmax.second)).x;
//   minmax = minmax_element(thrust::device, xp_d.begin(), xp_d.end(), compare_y());
//   int my=((float4)(*minmax.first)).y;
//   int My=((float4)(*minmax.second)).y;
//   minmax = minmax_element(thrust::device, xp_d.begin(), xp_d.end(), compare_z());
//   int mz=((float4)(*minmax.first)).z;
//   int Mz=((float4)(*minmax.second)).z;

   float Dx=(xM-xm), Dy=(yM-ym), Dz=(zM-zm);
   float bd=0.1; //border 0.07
   xm+=-Dx*bd; xM+=Dx*bd;
   ym+=-Dy*bd; yM+=Dy*bd;
   zm+=-Dz*bd; zM+=Dz*bd;
   Dx=(xM-xm); Dy=(yM-ym); Dz=(zM-zm);
   float Ntot=(float)Np/fgrid;
   float rhoL=cbrt(Ntot/(Dx*Dy*Dz));
   float Nx=round(rhoL*Dx), Ny=round(rhoL*Dy), Nz=round(rhoL*Dz);
   float Ax=Dx/Nx, Ay=Dy/Ny, Az=Dz/Nz;
   float Bx=xm+Ax/2, By=ym+Ay/2, Bz=zm+Az/2;
   //float Bx=xm, By=ym, Bz=zm;
   A=make_float3(Ax, Ay, Az);
   B=make_float3(Bx, By, Bz);
   N=make_int3(Nx, Ny, Nz);
   //N=make_int3(Nx+1, Ny+1, Nz+1);
}

void init_mesh_struct_clust(thrust::host_vector<int>& cellind_h, int celloff, int Ncell, int Npc, float3 A, float3 B, int3 N, float3& cA, float3& cB, int3& cN, float3& fA, float3& fB, int3& fN, float fgrid)
{
   //cpu min Max
   int Nyc=N.y-1, Nzc=N.z-1;
   int3 ic;
   int icell=cellind_h[celloff];
   ic.x=(int)(icell)/(Nyc*Nzc);
   ic.y=(int)(icell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=(int)(icell-ic.x*Nyc*Nzc-ic.y*Nzc);
   int iM, im=iM=ic.x;
   int jM, jm=jM=ic.y;
   int kM, km=kM=ic.z;
   for(int i=celloff;i<celloff+Ncell;i++)
   {
      icell=cellind_h[i];
      ic.x=(int)(icell)/(Nyc*Nzc);
      ic.y=(int)(icell-ic.x*Nyc*Nzc)/Nzc;
      ic.z=(int)(icell-ic.x*Nyc*Nzc-ic.y*Nzc);
      if(ic.x<im) im=ic.x;
      if(ic.x>iM) iM=ic.x;
      if(ic.y<jm) jm=ic.y;
      if(ic.y>jM) jM=ic.y;
      if(ic.z<km) km=ic.z;
      if(ic.z>kM) kM=ic.z;
   }
   iM+=1; jM+=1; kM+=1;
   float xm, ym, zm, xM, yM, zM;
   xm=xI(im,A.x,B.x,N.x); ym=xI(jm,A.y,B.y,N.y); zm=xI(km,A.z,B.z,N.z);
   //xM=xI(iM,A.x,B.x,N.x); yM=xI(jM,A.y,B.y,N.y); zM=xI(kM,A.z,B.z,N.z);
   float Bx=xm, By=ym, Bz=zm;
   int Nx=iM-im+1, Ny=jM-jm+1, Nz=kM-km+1;
   cA=A;
   cB=make_float3(Bx,By,Bz);
   cN=make_int3(Nx,Ny,Nz);
   float f=Npc/(Nx*Ny*Nz*fgrid);
   float s=max(mod(f,8),2);
   fA=make_float3(A.x/s,A.y/s,A.z/s);
   fB=cB;
   fN=make_int3(Nx*s-1,Ny*s-1,Nz*s-1);   
}

void init_p(thrust::host_vector<float4>& xp, thrust::host_vector<float3>& vp, thrust::host_vector<float3>& ap)
{
   int Np=xp.size();
   //initialize positions
   for(int i=0;i<50;i++)
   {
      //float xmin=-0.01;
      float xmax=20, xmin=15;
      //xe[i][0]=(float)rand()/RAND_MAX*(xmax-xmin)*2+xmin*2;
      //xe[i][1]=(float)rand()/RAND_MAX*(xmax-xmin)*2+xmin*2;
      //xe[i][2]=(float)rand()/RAND_MAX*(xmax-xmin)+xmin;
      //float d=(float)rand()/RAND_MAX*xmax;
      float d=(float)rand()/RAND_MAX*(xmax-xmin)+xmin;
      float t=(float)rand()/RAND_MAX*2.*MPI;
      float f=(float)rand()/RAND_MAX*MPI;
      //xt.w=charge
      float4 xt=make_float4(d*cos(t)*sin(f),d*sin(t)*sin(f),d*cos(f),1);
      //float4 xt=make_float4(d*cos(t)*sin(f),d*sin(t)*sin(f),d*cos(f)*0.03,1);
      xp[i]=xt;
   }
   float r0=10;
   for(int i=50;i<Np;i++)
   {
      if(i>Np/2) r0=-10;
      //float xmin=-0.01;
      float xmax=10;
      //xe[i][0]=(float)rand()/RAND_MAX*(xmax-xmin)*2+xmin*2;
      //xe[i][1]=(float)rand()/RAND_MAX*(xmax-xmin)*2+xmin*2;
      //xe[i][2]=(float)rand()/RAND_MAX*(xmax-xmin)+xmin;
      float d=(float)rand()/RAND_MAX*xmax;
      float t=(float)rand()/RAND_MAX*2.*MPI;
      float f=(float)rand()/RAND_MAX*MPI;
      //xt.w=charge
      float4 xt=make_float4(d*cos(t)*sin(f)+r0,d*sin(t)*sin(f)+r0,d*cos(f)+r0,1);
      //float4 xt=make_float4(d*cos(t)*sin(f)+r0,d*sin(t)*sin(f)+r0,d*cos(f)*0.03,1);
      xp[i]=xt;
   }
   //initialize velocities and accelerations
   for(int i=0;i<Np;i++)
   {
      float vmax=2000;
      float r1=(float)rand()/RAND_MAX*vmax*2-vmax;
      float r2=(float)rand()/RAND_MAX*vmax*2-vmax;
      float r3=(float)rand()/RAND_MAX*vmax*2-vmax;
      float3 at=make_float3(r1,r2,r3);
      float3 ap=make_float3(0,0,0);
   }
}

__global__ void copy_in(float *in_D, float *rho_D, int3 N)
{
   int Nx=2*N.x, Ny=2*N.y, Nz=2*N.z;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int i=(int)idx/(N.y*N.z); 
   int j=(int)(idx-i*N.y*N.z)/N.z;
   int k=(int)(idx-i*N.y*N.z-j*N.z);
   if(idx<N.x*N.y*N.z)
   {
//      if(N.x*N.y*N.z==72)
//         //printf("idx %d ind %d in %f r %f\n",idx, i*Ny*Nz+j*Nz+k, (float)in_D[i*Ny*Nz+j*Nz+k], rho_D[idx]);
      in_D[i*Ny*Nz+j*Nz+k]=rho_D[idx];
      //in_D[idx]=rho_D[idx];
   }
}

__global__ void copy_in1(float *in_D, float *rho_D, int3 N)
{
   int Nx=2*N.x, Ny=2*N.y, Nz=2*N.z;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int i=(int)idx/(Ny*Nz); 
   int j=(int)(idx-i*Ny*Nz)/Nz;
   int k=(int)(idx-i*Ny*Nz-j*Nz);
   if(idx<Nx*Ny*Nz)
   {
      if(i*N.y*N.z+j*N.z+k<N.x*N.y*N.z)
         in_D[idx]=0;//rho_D[i*N.y*N.z+j*N.z+k];
      else in_D[idx]=0;
   }
}

__device__ float lambda(int i, int Nx, float Dx)
{
   float l = Dx*(i < Nx/2 ? i : i-Nx);
    return l*l;
}

__global__ void ker(cufftReal *ker_D, float3 A, float3 B, int3 N)
{
   int Nx=2*N.x;
   int Ny=2*N.y;
   int Nz=2*N.z;
   int Ntot=Nx*Ny*Nz;
   //float c=(A.x*A.y*A.z)/(4*MPI);
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Ntot)
   //for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx<Ntot; idx += blockDim.x*gridDim.x) 
   {
      int i=(int)idx/(Ny*Nz); 
      int j=(int)(idx-i*Ny*Nz)/Nz;
      int k=(int)(idx-i*Ny*Nz-j*Nz);

      float lx=lambda(i,Nx,A.x);
      float ly=lambda(j,Ny,A.y);
      float lz=lambda(k,Nz,A.z);
      float ltot=lx+ly+lz;
      float den=4*MPI*sqrtf(ltot);
      if(idx!=0) ker_D[idx]=(cufftReal)1./den;
      else ker_D[idx]=0;
   }
}


void fft(float *rho_D, cufftComplex *frho_D, cufftComplex *fker_D, float3 A, float3 B, int3 N)
{
   int Nx=2*N.x;
   int Ny=2*N.y;
   int Nz=2*N.z;
   int Ntot=Nx*Ny*Nz;

   //cudaStream_t s1, s2;
   //cudaStreamCreate(&s1);
   //cudaStreamCreate(&s2);

   // cuFFT 3D plans for FFT
   cufftHandle f_plan, f1_plan;
   int rank = 3;
   int n[3] = {Nx, Ny, Nz};
   int idist = Nx*Ny*Nz, odist = Nx*Ny*(Nz/2+1);
   int inembed[] = {Nx, Ny, Nz};
   int onembed[] = {Nx, Ny, Nz/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&f_plan,rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,1);
   //cufftPlanMany(&f1_plan,rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,1);
   f1_plan=f_plan;
   //cufftSetStream(f_plan, s1);
   //cufftSetStream(f1_plan, s2);

   //input vectors
   int threads=64;
   int blocks=(N.x*N.y*N.z)/threads+1;
   int blocksL=Ntot/threads+1;
   //thrust::device_vector<cufftReal> in_d(Ntot,0);
   //cufftReal* in_D = raw_pointer_cast(&in_d[0]);

   float *in_D;
   gpuErrchk(cudaMalloc(&in_D, Ntot*sizeof(float)));
   gpuErrchk(cudaMemset(in_D, 0, Ntot*sizeof(float)));
   //copy_in<<<blocks,threads,0,s1>>>(in_D, rho_D, N);
////printf("N size %d Ntot %d\n", N.x*N.y*N.z, Ntot);
   copy_in<<<blocks,threads>>>(in_D, rho_D, N);
//gpuErrchk( cudaPeekAtLastError() );
//gpuErrchk( cudaDeviceSynchronize() );
   cufftExecR2C(f_plan, in_D, frho_D);
//gpuErrchk( cudaPeekAtLastError() );
//gpuErrchk( cudaDeviceSynchronize() );
   cudaFree(in_D);


//size_t free_byte ;
//size_t total_byte ;
//cudaMemGetInfo( &free_byte, &total_byte ) ;
//double free_db = (double)free_byte ;
//double total_db = (double)total_byte ;
//double used_db = total_db - free_db ;
////printf("Mem in fft: used = %f, free = %f MB, total = %f MB\n",
//            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

   //thrust::device_vector<cufftReal> ker_d(Ntot);
   //cufftReal* ker_D = raw_pointer_cast(&ker_d[0]);
   cufftReal *ker_D;
   cudaMalloc(&ker_D, Ntot*sizeof(cufftReal));
   //ker<<<blocksL,threads,0,s2>>>(ker_D, A, B, N);
   ker<<<blocksL,threads>>>(ker_D, A, B, N);
   cufftExecR2C(f1_plan, ker_D, fker_D);


   //cudaFree(in_D);
   cudaFree(ker_D);

   cufftDestroy(f_plan);
   cufftDestroy(f1_plan);

   //cudaStreamDestroy(s1);
   //cudaStreamDestroy(s2);
}

//void fft_ker(cufftComplex *fker_D, float3 A, float3 B, int3 N)
//{
//   int Nx=2*N.x;
//   int Ny=2*N.y;
//   int Nz=2*N.z;
//   int Ntot=Nx*Ny*Nz;

//   // cuFFT 3D plans for FFT
//   cufftHandle f_plan;
//   int rank = 3;
//   int n[3] = {Nx, Ny, Nz};
//   int idist = Nx*Ny*Nz, odist = Nx*Ny*(Nz/2+1);
//   int inembed[] = {Nx, Ny, Nz};
//   int onembed[] = {Nx, Ny, Nz/2+1};
//   int istride = 1, ostride = 1;
//   cufftPlanMany(&f_plan,rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,1);

//   //input vectors
//   cufftReal *ker_D;
//   cudaMalloc(&ker_D, Ntot*sizeof(cufftReal));
//   int threads=256;
//   int blocks=Ntot/threads+1;
//   ker<<<blocks,threads>>>(ker_D, A, B, N);
//   //cudaMemcpy(in_D, rho_D, Ntot*sizeof(cufftReal), cudaMemcpyDeviceToDevice);

//   //Compute Forward FFT
//   cufftExecR2C(f_plan, ker_D, fker_D);

//   cufftDestroy(f_plan);
//   cudaFree(ker_D);
//}

__device__ cufftComplex multiply(cufftComplex z1, cufftComplex z2)
{
   cufftComplex c;
//   float re = z1.x*z2.x-z1.y*z2.y;
//   float im = z1.y*z2.x+z1.x*z2.y;
   float re = z1.x*z2.x; //z2 = fker is real
   float im = z1.y*z2.x;
   c.x=re; c.y=im;
   return c;
}


__global__ void solver(cufftComplex *frho_D, cufftComplex *fker_D, int Ntot)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Ntot)
   {
      cufftComplex fr = frho_D[idx];
      cufftComplex fk = fker_D[idx];
      cufftComplex fm = multiply(fr,fk);
      frho_D[idx] = fm;
   }
}


void Poisson_Solver(cufftComplex *frho_D, cufftComplex *fker_D, int3 N)
{
   int Ntot=2*N.x*2*N.y*(2*N.z/2+1);
   int threads=256;
   int blocks=Ntot/threads+1;
   solver<<<blocks,threads>>>(frho_D, fker_D, Ntot);
}

void Poisson_Solver_thrust(thrust::device_vector<thrust::complex<float> >& frho_d, thrust::device_vector<thrust::complex<float> >& fker_d)
{
   transform(frho_d.begin(), frho_d.end(), fker_d.begin(), frho_d.begin(), 
                         thrust::multiplies<thrust::complex<float> >());
}

__global__ void copy_out(float *rho_D, cufftReal *in_D, int3 N)
{
   int Nx=2*N.x, Ny=2*N.y, Nz=2*N.z;
   float norm=1./((float)(Nx*Ny*Nz));
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int i=(int)idx/(N.y*N.z); 
   int j=(int)(idx-i*N.y*N.z)/N.z;
   int k=(int)(idx-i*N.y*N.z-j*N.z);
   if(idx<N.x*N.y*N.z)
      rho_D[idx]=(float)in_D[i*Ny*Nz+j*Nz+k]*norm;
}


void fft_inv(cufftComplex *frho_D, float *rho_D, int3 N)
{
   int Nx=2*N.x;
   int Ny=2*N.y;
   int Nz=2*N.z;
   int Ntot=Nx*Ny*Nz;

   // cuFFT 3D plans for FFT
   cufftHandle i_plan;
   int rank = 3;
   int n[3] = {Nx, Ny, Nz};
   int idist = Nx*Ny*Nz, odist = Nx*Ny*(Nz/2+1);
   int inembed[] = {Nx, Ny, Nz};
   int onembed[] = {Nx, Ny, Nz/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&i_plan,rank,n,onembed,ostride,odist,inembed,istride,idist,CUFFT_C2R,1);

   //input vectors
   cufftReal *in_D;
   cudaMalloc(&in_D, Ntot*sizeof(cufftReal));
   //cudaMemset(in_D, 0, Ntot*sizeof(cufftReal));

   //Compute Forward FFT
   cufftExecC2R(i_plan, frho_D, in_D);
   int threads=256;
   int blocks=(N.x*N.y*N.z)/threads+1;
   copy_out<<<blocks,threads>>>(rho_D, in_D, N);

   cufftDestroy(i_plan);
   cudaFree(in_D);
}


void save_traj(thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fp_d, thrust::device_vector<float>& rho_d, thrust::device_vector<int>& Nc_d, float3 A, float3 B, int3 N)
{
   //save result on file
   bool save=true;
   bool gnuplot=true;
   if(save)
   {
      ofstream ofile;
      if(gnuplot)  //format output for gnuplot
      {
         ofile.open("./xp.bin", ios::out | ios::binary);
         thrust::host_vector<float4> xp_h(xp_d);
         for(int j=0;j<xp_h.size();j++)
         {
            struct X
            {
                float a, b, c;
            } x;
            x.a = xp_h[j].x;
            x.b = xp_h[j].y;
            x.c = xp_h[j].z;
            ofile.write(reinterpret_cast<char *>(&x), sizeof(x));
         }
         ofile.close();
         ofile.open("./rho.bin", ios::out | ios::binary);
         thrust::host_vector<float> rho_h(rho_d);
         for(int i=0;i<N.x;i++)
         {
            for(int j=0;j<N.y;j++)
            {
               for(int k=0;k<N.z;k++)
               {
                  float x=rho_h[i*N.y*N.z+j*N.z+k];
                  ofile.write(reinterpret_cast<char *>(&x), sizeof(float));
               }
            }
         }
         ofile.close();
         ofile.open("./fp.bin", ios::out | ios::binary);
         thrust::host_vector<float> fp_h(fp_d);
         for(int i=0;i<xp_h.size();i++)
         {
            float x=fp_h[i];
            ofile.write(reinterpret_cast<char *>(&x), sizeof(float));
         }
         ofile.close();
         ofile.open("./Nc.bin", ios::out | ios::binary);
         thrust::host_vector<int> Nc_h(Nc_d);
         for(int i=0;i<Nc_h.size();i++)
         {
            int x=Nc_h[i];
            ofile.write(reinterpret_cast<char *>(&x), sizeof(int));
         }
         ofile.close();
         ofile.open("./param.txt", ios::out | ios::out);
         ofile << "A={"<<A.x<<","<<A.y<<","<<A.z<<"};"<<endl;
         ofile << "B={"<<B.x<<","<<B.y<<","<<B.z<<"};"<<endl;
         ofile << "Ng={"<<N.x<<","<<N.y<<","<<N.z<<"};"<<endl;
         ofile << "Np="<<xp_h.size()<<";"<<endl;
         ofile.close();

      }
   }
}

void save_traj_tot(thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fp_d, float3 A, float3 B, int3 N)
{
   //save result on file
   bool save=true;
   bool gnuplot=true;
   if(save)
   {
      ofstream ofile;
      if(gnuplot)  //format output for gnuplot
      {
         ofile.open("./xp.bin", ios::out | ios::binary);
         thrust::host_vector<float4> xp_h(xp_d);
         for(int j=0;j<xp_h.size();j++)
         {
            struct X
            {
                float a, b, c;
            } x;
            x.a = xp_h[j].x;
            x.b = xp_h[j].y;
            x.c = xp_h[j].z;
            ofile.write(reinterpret_cast<char *>(&x), sizeof(x));
         }
         ofile.close();
         ofile.open("./fp.bin", ios::out | ios::binary);
         thrust::host_vector<float> fp_h(fp_d);
         for(int i=0;i<xp_h.size();i++)
         {
            float x=fp_h[i];
            ofile.write(reinterpret_cast<char *>(&x), sizeof(float));
         }
         ofile.close();
         ofile.open("./param.txt", ios::out | ios::out);
         ofile << "A={"<<A.x<<","<<A.y<<","<<A.z<<"};"<<endl;
         ofile << "B={"<<B.x<<","<<B.y<<","<<B.z<<"};"<<endl;
         ofile << "Ng={"<<N.x<<","<<N.y<<","<<N.z<<"};"<<endl;
         ofile << "Np="<<xp_h.size()<<";"<<endl;
         ofile.close();

      }
   }
}



void save_fft(cufftComplex *frho_D, cufftComplex *fker_D, int3 N)
{
   //save result on file
   bool save=true;
   bool gnuplot=true;
   if(save)
   {
      ofstream ofile;
      if(gnuplot)  //format output for gnuplot
      {
         int Ntot=2*N.x*2*N.y*(2*N.z/2+1);
         cufftComplex *frho_h, *fker_h;
         cudaMallocHost(&frho_h, Ntot*sizeof(cufftComplex));
         cudaMallocHost(&fker_h, Ntot*sizeof(cufftComplex));
         cudaMemcpy(frho_h, frho_D, Ntot*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
         cudaMemcpy(fker_h, fker_D, Ntot*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
         ofile.open("./frhoRe.bin", ios::out | ios::binary);
         for(int i=0;i<Ntot;i++)
         {
            float x=frho_h[i].x;
            ofile.write(reinterpret_cast<char *>(&x), sizeof(float));
         }
         ofile.close();
         ofile.open("./frhoIm.bin", ios::out | ios::binary);
         for(int i=0;i<Ntot;i++)
         {
            float x=frho_h[i].y;
            ofile.write(reinterpret_cast<char *>(&x), sizeof(float));
         }
         ofile.close();
         ofile.open("./fker.bin", ios::out | ios::binary);
         for(int i=0;i<Ntot;i++)
         {
            float x=fker_h[i].x;
            ofile.write(reinterpret_cast<char *>(&x), sizeof(float));
         }
         ofile.close();
         cudaFreeHost(frho_h);
         cudaFreeHost(fker_h);
      }
   }
}

__global__ void swap(int *offset_d, int *Ioff_d, int Nc)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Nc)
   {
      int ic=offset_d[idx];
      Ioff_d[ic]=idx;
   }
}

void OrderCell(thrust::device_vector<int>& XC_d, thrust::device_vector<int>& CX_d, thrust::device_vector<int>& Nc_d, int Np, int3 N)
{
   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   thrust::device_vector<int> XCt_d(XC_d);
   //CX_d
   thrust::host_vector<int> IP_h(thrust::make_counting_iterator(0), thrust::make_counting_iterator(Np)); 
   //thrust::device_vector<int> IP_d(thrust::make_counting_iterator(0), thrust::make_counting_iterator(Np)); 
   thrust::device_vector<int> IP_d(Np); 
   IP_d=IP_h;
   thrust::device_vector<int> PI_d(Np);
   PI_d=IP_h;
   sort_by_key(thrust::device, XCt_d.begin(), XCt_d.end(), IP_d.begin());
//   cudaDeviceSynchronize();
   copy(thrust::device, IP_d.begin(), IP_d.end(), CX_d.begin());
   //CX_d=IP_d;
   //XC_d
   thrust::device_vector<int> PIt_d(IP_d);
   sort_by_key(thrust::device, PIt_d.begin(), PIt_d.end(), PI_d.begin());
//   cudaDeviceSynchronize();
   XC_d=PI_d;
   //compute number of particles per cell Npc and cells offset 
   thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> end;
   thrust::device_vector<int> offset_d(Np);
   thrust::device_vector<int> Npc_d(Np);
   thrust::device_vector<int> one_d(Np,1);
   cudaDeviceSynchronize();
   end = reduce_by_key(thrust::device, XCt_d.begin(), XCt_d.end(), one_d.begin(), offset_d.begin(), Npc_d.begin());
//   cudaDeviceSynchronize();
   offset_d.resize(end.first- offset_d.begin());
   Npc_d.resize(end.second- Npc_d.begin());
   //compute Nc_d from Npc and offset
   thrust::device_vector<int> Ioff_d(Nc,-1);
   int* offset_D = raw_pointer_cast(&offset_d[0]);
   int* Ioff_D = raw_pointer_cast(&Ioff_d[0]);
   int threads=256;
   int blocks=offset_d.size()/threads+1;
   swap<<<blocks, threads>>>(offset_D, Ioff_D, offset_d.size());
//   cudaDeviceSynchronize();
   gather(thrust::device, Ioff_d.begin(), Ioff_d.end(), Npc_d.begin(), Nc_d.begin());

int sum=thrust::reduce(thrust::device, Nc_d.begin(), Nc_d.end());
}

__device__ void ParticleToCell(float4 x, float fg[], int i_cell, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   int Nyc=N.y-1, Nzc=N.z-1;
   float den=Ax*Ay*Az;
   int3 ic;//,ic1,ic2,ic3,ic4,ic5,ic6,ic7;
   //particle charge
   int c=(int)x.w;
   //cell vertices
   ic.x=(int)(i_cell)/(Nyc*Nzc);
   ic.y=(int)(i_cell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=(int)(i_cell-ic.x*Nyc*Nzc-ic.y*Nzc);
//   ic1.x=ic.x+1; ic1.y=ic.y; ic1.z=ic.z;
//   ic2.x=ic.x+1; ic1.y=ic.y+1; ic1.z=ic.z;
//   ic3.x=ic.x+1; ic1.y=ic.y+1; ic1.z=ic.z+1;
//   ic4.x=ic.x; ic1.y=ic.y+1; ic1.z=ic.z;
//   ic5.x=ic.x; ic1.y=ic.y+1; ic1.z=ic.z+1;
//   ic6.x=ic.x; ic1.y=ic.y; ic1.z=ic.z+1;
//   ic7.x=ic.x+1; ic1.y=ic.y; ic1.z=ic.z+1;
   //first grid point position xc, distance, xcp
   float3 xc, xcp;
   xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
   xcp.x=x.x-xc.x; xcp.y=x.y-xc.y; xcp.z=x.z-xc.z;
   //particle volume contribution to each cell grid points
   float a0,a1,a2,a3,a4,a5,a6,a7;
   a0=((Ax-abs(xcp.x))*(Ay-abs(xcp.y))*(Az-abs(xcp.z)))/den;
   a1=(abs(xcp.x)*(Ay-abs(xcp.y))*(Az-abs(xcp.z)))/den;
   a2=(abs(xcp.x)*abs(xcp.y)*(Az-abs(xcp.z)))/den;
   a3=(abs(xcp.x)*abs(xcp.y)*abs(xcp.z))/den;
   a4=((Ax-abs(xcp.x))*abs(xcp.y)*(Az-abs(xcp.z)))/den;
   a5=((Ax-abs(xcp.x))*abs(xcp.y)*abs(xcp.z))/den;
   a6=((Ax-abs(xcp.x))*(Ay-abs(xcp.y))*abs(xcp.z))/den;
   a7=(abs(xcp.x)*(Ay-abs(xcp.y))*abs(xcp.z))/den;
   //charge density at each nearest grid point
   fg[0]+=c*a0;
   fg[1]+=c*a1;
   fg[2]+=c*a2;
   fg[3]+=c*a3;
   fg[4]+=c*a4;
   fg[5]+=c*a5;
   fg[6]+=c*a6;
   fg[7]+=c*a7;
//   atomicAdd(&fg[0],c*a0);
//   atomicAdd(&fg[1],c*a1);
//   atomicAdd(&fg[2],c*a2);
//   atomicAdd(&fg[3],c*a3);
//   atomicAdd(&fg[4],c*a4);
//   atomicAdd(&fg[5],c*a5);
//   atomicAdd(&fg[6],c*a6);
//   atomicAdd(&fg[7],c*a7);
}

__device__ void CellToParticle(float4 x, float &fp, float fg[], int i_cell, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   int Nyc=N.y-1, Nzc=N.z-1;
   float den=Ax*Ay*Az;
   int3 ic;//,ic1,ic2,ic3,ic4,ic5,ic6,ic7;
   //cell vertices
   ic.x=(int)(i_cell)/(Nyc*Nzc);
   ic.y=(int)(i_cell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=(int)(i_cell-ic.x*Nyc*Nzc-ic.y*Nzc);
   //first grid point position xc, distance, xcp
   float3 xc, xcp;
   xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
   xcp.x=x.x-xc.x; xcp.y=x.y-xc.y; xcp.z=x.z-xc.z;
   //particle volume contribution to each cell grid points
   float a0,a1,a2,a3,a4,a5,a6,a7;
   a0=((Ax-abs(xcp.x))*(Ay-abs(xcp.y))*(Az-abs(xcp.z)))/den;
   a1=(abs(xcp.x)*(Ay-abs(xcp.y))*(Az-abs(xcp.z)))/den;
   a2=(abs(xcp.x)*abs(xcp.y)*(Az-abs(xcp.z)))/den;
   a3=(abs(xcp.x)*abs(xcp.y)*abs(xcp.z))/den;
   a4=((Ax-abs(xcp.x))*abs(xcp.y)*(Az-abs(xcp.z)))/den;
   a5=((Ax-abs(xcp.x))*abs(xcp.y)*abs(xcp.z))/den;
   a6=((Ax-abs(xcp.x))*(Ay-abs(xcp.y))*abs(xcp.z))/den;
   a7=(abs(xcp.x)*(Ay-abs(xcp.y))*abs(xcp.z))/den;
   //charge density at each nearest grid point
   fp = fg[0]*a0+fg[1]*a1+fg[2]*a2+fg[3]*a3+
        fg[4]*a4+fg[5]*a5+fg[6]*a6+fg[7]*a7;
}


__device__ void CellVertex_x(float3 xg[], int i_cell, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   int Nyc=Ny-1, Nzc=Nz-1;
   int3 ic,ic1,ic2,ic3,ic4,ic5,ic6,ic7;
   //cell vertices
   ic.x=(int)(i_cell)/(Nyc*Nzc);
   ic.y=(int)(i_cell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=(int)(i_cell-ic.x*Nyc*Nzc-ic.y*Nzc);
   ic1.x=ic.x+1; ic1.y=ic.y; ic1.z=ic.z;
   ic2.x=ic.x+1; ic2.y=ic.y+1; ic2.z=ic.z;
   ic3.x=ic.x+1; ic3.y=ic.y+1; ic3.z=ic.z+1;
   ic4.x=ic.x; ic4.y=ic.y+1; ic4.z=ic.z;
   ic5.x=ic.x; ic5.y=ic.y+1; ic5.z=ic.z+1;
   ic6.x=ic.x; ic6.y=ic.y; ic6.z=ic.z+1;
   ic7.x=ic.x+1; ic7.y=ic.y; ic7.z=ic.z+1;
   //vertices positions xc
   float3 xc,xc1,xc2,xc3,xc4,xc5,xc6,xc7;
   xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
   xc1.x=xI(ic1.x,Ax,Bx,Nx); xc1.y=xI(ic1.y,Ay,By,Ny); xc1.z=xI(ic1.z,Az,Bz,Nz);
   xc2.x=xI(ic2.x,Ax,Bx,Nx); xc2.y=xI(ic2.y,Ay,By,Ny); xc2.z=xI(ic2.z,Az,Bz,Nz);
   xc3.x=xI(ic3.x,Ax,Bx,Nx); xc3.y=xI(ic3.y,Ay,By,Ny); xc3.z=xI(ic3.z,Az,Bz,Nz);
   xc4.x=xI(ic4.x,Ax,Bx,Nx); xc4.y=xI(ic4.y,Ay,By,Ny); xc4.z=xI(ic4.z,Az,Bz,Nz);
   xc5.x=xI(ic5.x,Ax,Bx,Nx); xc5.y=xI(ic5.y,Ay,By,Ny); xc5.z=xI(ic5.z,Az,Bz,Nz);
   xc6.x=xI(ic6.x,Ax,Bx,Nx); xc6.y=xI(ic6.y,Ay,By,Ny); xc6.z=xI(ic6.z,Az,Bz,Nz);
   xc7.x=xI(ic7.x,Ax,Bx,Nx); xc7.y=xI(ic7.y,Ay,By,Ny); xc7.z=xI(ic7.z,Az,Bz,Nz);
   xg[0]=xc; xg[1]=xc1; xg[2]=xc2; xg[3]=xc3;
   xg[4]=xc4; xg[5]=xc5; xg[6]=xc6; xg[7]=xc7;
}

__host__ __device__ void CellVertex_ix(int3 ig[], float3 xg[], int i_cell, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   int Nyc=Ny-1, Nzc=Nz-1;
   int3 ic,ic1,ic2,ic3,ic4,ic5,ic6,ic7;
   //cell vertices
   ic.x=(int)(i_cell)/(Nyc*Nzc);
   ic.y=(int)(i_cell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=(int)(i_cell-ic.x*Nyc*Nzc-ic.y*Nzc);
   ic1.x=ic.x+1; ic1.y=ic.y; ic1.z=ic.z;
   ic2.x=ic.x+1; ic2.y=ic.y+1; ic2.z=ic.z;
   ic3.x=ic.x+1; ic3.y=ic.y+1; ic3.z=ic.z+1;
   ic4.x=ic.x; ic4.y=ic.y+1; ic4.z=ic.z;
   ic5.x=ic.x; ic5.y=ic.y+1; ic5.z=ic.z+1;
   ic6.x=ic.x; ic6.y=ic.y; ic6.z=ic.z+1;
   ic7.x=ic.x+1; ic7.y=ic.y; ic7.z=ic.z+1;
   ig[0]=ic; ig[1]=ic1; ig[2]=ic2; ig[3]=ic3;
   ig[4]=ic4; ig[5]=ic5; ig[6]=ic6; ig[7]=ic7;
   //vertices positions xc
   float3 xc,xc1,xc2,xc3,xc4,xc5,xc6,xc7;
   xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
   xc1.x=xI(ic1.x,Ax,Bx,Nx); xc1.y=xI(ic1.y,Ay,By,Ny); xc1.z=xI(ic1.z,Az,Bz,Nz);
   xc2.x=xI(ic2.x,Ax,Bx,Nx); xc2.y=xI(ic2.y,Ay,By,Ny); xc2.z=xI(ic2.z,Az,Bz,Nz);
   xc3.x=xI(ic3.x,Ax,Bx,Nx); xc3.y=xI(ic3.y,Ay,By,Ny); xc3.z=xI(ic3.z,Az,Bz,Nz);
   xc4.x=xI(ic4.x,Ax,Bx,Nx); xc4.y=xI(ic4.y,Ay,By,Ny); xc4.z=xI(ic4.z,Az,Bz,Nz);
   xc5.x=xI(ic5.x,Ax,Bx,Nx); xc5.y=xI(ic5.y,Ay,By,Ny); xc5.z=xI(ic5.z,Az,Bz,Nz);
   xc6.x=xI(ic6.x,Ax,Bx,Nx); xc6.y=xI(ic6.y,Ay,By,Ny); xc6.z=xI(ic6.z,Az,Bz,Nz);
   xc7.x=xI(ic7.x,Ax,Bx,Nx); xc7.y=xI(ic7.y,Ay,By,Ny); xc7.z=xI(ic7.z,Az,Bz,Nz);
   xg[0]=xc; xg[1]=xc1; xg[2]=xc2; xg[3]=xc3;
   xg[4]=xc4; xg[5]=xc5; xg[6]=xc6; xg[7]=xc7;
}


__global__ void 
//__launch_bounds__(512,3)
PPField_ker(const float4* __restrict__ xp_D, float* __restrict__ fp_D, const int* __restrict__ Nc_D, const int* __restrict__ cellind_d, const int* __restrict__ offset_d, int Np, int Nc, int3 N, int lc)
//__global__ void PPField_ker(float4 *xp_D, float *fp_D, int *XC_D,int *CX_D, int *Nc_D, int Np)
{
   //extern __shared__ float4 xp_S[];
   int Ncx=N.x-1, Ncy=N.y-1, Ncz=N.z-1;
   //for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx<Np; idx += blockDim.x*gridDim.x) 
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   {

      float4 x=xp_D[idx];
//      xp_S[threadIdx.x]=x;
//      xp_S[threadIdx.x+blockDim.x]=xp_D[idx+blockDim.x];
//      xp_S[threadIdx.x+2*blockDim.x]=xp_D[idx+2*blockDim.x];
//      xp_S[threadIdx.x+3*blockDim.x]=xp_D[idx+3*blockDim.x];
//      __syncthreads();

      float fp=0;

      int ic=cellind_d[idx];
      int i_c=ic/(Ncy*Ncz);
      int j_c=(ic-i_c*(Ncy*Ncz))/Ncz;
      int k_c=(ic-i_c*(Ncy*Ncz)-j_c*Ncz);

      #pragma unroll 3
      for(int i=-lc;i<=lc;i++)
      {
         #pragma unroll 3
         for(int j=-lc;j<=lc;j++)
         {
            #pragma unroll 3
            for(int k=-lc;k<=lc;k++)
            {
               int i1=i_c+i, j1=j_c+j, k1=k_c+k;
               if(i1<0 || i1>=Ncx || j1<0 || j1>=Ncy || k1<0 || k1>=Ncz) 
                  continue;

               ic=i1*Ncy*Ncz+j1*Ncz+k1;
               int Ncell=Nc_D[ic];
               int offset=offset_d[ic];

         //      extern __shared__ float4 xp_S[];
         //      int numTiles=1;
         //      if(Ncell>blockDim.x) numTiles+=(int)Ncell/blockDim.x;
         //      for (int tile = 0; tile < numTiles; tile++)
         //      {
         //         xp_S[threadIdx.x]=xp_D[tile * blockDim.x + threadIdx.x + offset];
         //         __syncthreads();
         //         int Nm = (tile < numTiles-1 ? blockDim.x : Ncell-(numTiles-1)*blockDim.x);
                  #pragma unroll 128
                  for(int n=offset;n<offset+Ncell;n++)
                  {
//                     float4 xp;
//                     if(n>=blockDim.x*blockIdx.x && n<blockDim.x*(blockIdx.x+4))
//                        xp=xp_S[n-blockDim.x*blockIdx.x];
//                     else xp=xp_D[n];
                     float4 xp=xp_D[n];
                     float dx=x.x-xp.x;
                     float dy=x.y-xp.y;
                     float dz=x.z-xp.z;
                     dx*=dx; dy*=dy; dz*=dz;
                     float den=dx+dy+dz;
                     den=(den == 0 ? 0 : 4*MPI*sqrtf(den));//sqrtf(den);
                     //fp+=xp.w/den;
                     fp+=(den == 0. ? 0 : xp.w/den);
                  }
         //         __syncthreads();
         //      }
            }
         }
      }
      fp_D[idx]+=fp;
      ////printf("%d, %f\n",idx,fp);
      //fp_D[idx]=10.;
   }
}


__global__ void CellDensity_ker(const float4* __restrict__ xp_D, float4* __restrict__ rg0_D, float4* __restrict__ rg1_D,  const int* __restrict__ Nc_D, const int* __restrict__ offset_d, int Nc, int3 N, float3 A, float3 B)
{
   const int Ng=8;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Nc)
   {
      int Ncell=Nc_D[idx];
      int offset=offset_d[idx];
      float rg[Ng]={0};
      for(int i=offset;i<offset+Ncell;i++)
      {
         float4 xp=xp_D[i];
         ParticleToCell(xp, rg, idx, N, A, B);
      }
      rg0_D[idx]=make_float4(rg[0],rg[1],rg[2],rg[3]);
      rg1_D[idx]=make_float4(rg[4],rg[5],rg[6],rg[7]);
      //fg_D[idx]=make_float8(0,0,0,0,0,0,0,0);
   }
}

__host__ void CellDistance(float* dc_h, int3 N, float3 A, float3 B, int lc)
{
   const int Ng=8;
   int Ncx=N.x-1, Ncy=N.y-1, Ncz=N.z-1;
   int i_c=Ncx/2, j_c=Ncy/2, k_c=Ncz/2;
   int ic=i_c*Ncy*Ncz+j_c*Ncz+k_c;
   float3 xg[Ng];
   int3 ig[Ng];
   CellVertex_ix(ig, xg, ic, N, A, B);

   #pragma unroll 12
   for(int i=-lc;i<=lc;i++)
   {
      #pragma unroll 12
      for(int j=-lc;j<=lc;j++)
      {
         #pragma unroll 12
         for(int k=-lc;k<=lc;k++)
         {
            int i1=i_c+i, j1=j_c+j, k1=k_c+k;
            if(i1<0 || i1>=Ncx || j1<0 || j1>=Ncy || k1<0 || k1>=Ncz) 
               continue;
            ic=i1*Ncy*Ncz+j1*Ncz+k1;
            //if(ic==idx && Ncell<2) continue;
            float3 xg1[Ng];
            int3 ig1[Ng];
            CellVertex_ix(ig1, xg1, ic, N, A, B);
            //cycle over single cell
            #pragma unroll Ng
            for(int n=0;n<Ng;n++)
            {
               //cycle over all considered cells
               #pragma unroll Ng
               for(int m=0;m<Ng;m++)
               {
                  //if(n==m) continue;
                  float dx=xg[n].x-xg1[m].x;
                  float dy=xg[n].y-xg1[m].y;
                  float dz=xg[n].z-xg1[m].z;
                  int di=abs(ig[n].x-ig1[m].x);
                  int dj=abs(ig[n].y-ig1[m].y);
                  int dk=abs(ig[n].z-ig1[m].z);
                  dx*=dx; dy*=dy; dz*=dz;
                  float den=dx+dy+dz;
                  den=(den == 0 ? 0 : 4*MPI*sqrtf(den));
                  dc_h[di*(lc+2)*(lc+2)+dj*(lc+2)+dk]=(den == 0 ? 0 : 1./den);
               }
            }
         }
      }
   }
}

__global__ void 
//__launch_bounds__(512,3)
PGField_ker(const float4* __restrict__ rg0_D, const float4* __restrict__ rg1_D, float4* __restrict__ fg0_D, float4* __restrict__ fg1_D, const float* __restrict__ dc_D, int Nc, int3 N, float3 A, float3 B, int lc, int Ndc)
{
   const int Ng=8;
   int Ndx=(lc+2)*(lc+2), Ndy=(lc+2);
   int Ncx=N.x-1, Ncy=N.y-1, Ncz=N.z-1;
   extern __shared__ float dc[]; //cells vertices distances
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(threadIdx.x<Ndc)
   {
      dc[threadIdx.x]=dc_D[threadIdx.x];
   }
   __syncthreads();
   if (idx < Nc)
   {
      int ic=idx;
      int i_c=ic/(Ncy*Ncz);
      int j_c=(ic-i_c*(Ncy*Ncz))/Ncz;
      int k_c=(ic-i_c*(Ncy*Ncz)-j_c*Ncz);

      float3 xg[Ng];
      int3 ig[Ng];
      CellVertex_ix(ig, xg, ic, N, A, B);

      float fg[Ng]={0};
      #pragma unroll 12
      for(int i=-lc;i<=lc;i++)
      {
         #pragma unroll 12
         for(int j=-lc;j<=lc;j++)
         {
            #pragma unroll 12
            for(int k=-lc;k<=lc;k++)
            {
               int i1=i_c+i, j1=j_c+j, k1=k_c+k;
               if(i1<0 || i1>=Ncx || j1<0 || j1>=Ncy || k1<0 || k1>=Ncz) 
                  continue;

               ic=i1*Ncy*Ncz+j1*Ncz+k1;
               //if(ic==idx && Ncell<2) continue;
               float3 xg1[Ng];
               int3 ig1[Ng];
               CellVertex_ix(ig1, xg1, ic, N, A, B);
               float4 rgt0=rg0_D[ic];
               float4 rgt1=rg1_D[ic];
               float rg1[Ng]={rgt0.x,rgt0.y,rgt0.z,rgt0.w,rgt1.x,rgt1.y,rgt1.z,rgt1.w};
               //cycle over single cell
               #pragma unroll Ng
               for(int n=0;n<Ng;n++)
               {
                  float fgt=0;
                  //cycle over all considered cells
                  #pragma unroll Ng
                  for(int m=0;m<Ng;m++)
                  {
                     int di=abs(ig[n].x-ig1[m].x);
                     int dj=abs(ig[n].y-ig1[m].y);
                     int dk=abs(ig[n].z-ig1[m].z);
                     int idc=di*Ndx+dj*Ndy+dk;
                     float dct=dc[idc];
                     fgt-=rg1[m]*dct;
                  }
                  fg[n]+=fgt;
               }
            }
         }
      }
      //float8 fg8=make_float8(fg[0],fg[1],fg[2],fg[3],fg[4],fg[5],fg[6],fg[7]);
      //fg_D[idx]=fg8;
      float4 fg0=make_float4(fg[0],fg[1],fg[2],fg[3]);
      float4 fg1=make_float4(fg[4],fg[5],fg[6],fg[7]);
      fg0_D[idx]=fg0;
      fg1_D[idx]=fg1;
   }
}

__global__ void CellFieldToParticle_ker(const float4* __restrict__ xp_D, float* __restrict__ fp_D, const float4* __restrict__ fg0_D, const float4* __restrict__ fg1_D, const int* __restrict__ cellind_d, int Np, int3 N, float3 A, float3 B)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   {
      int ic=cellind_d[idx];
      float4 fg0=fg0_D[ic];
      float4 fg1=fg1_D[ic];
      float fg[8]={fg0.x, fg0.y, fg0.z, fg0.w, fg1.x, fg1.y, fg1.z, fg1.w};
      float fp;
      float4 xp=xp_D[idx];
      CellToParticle(xp, fp, fg, ic, N, A, B);
      fp_D[idx]=fp;
   }
}


void FillCell(int *offset_h, int *cellind_h, int *Nc_h, int Nc, int Np)
{
   int offset=0, ind=0;
   for(int i=0;i<Nc;i++)
   {
      offset_h[i]=offset;
      for(int j=offset;j<offset+Nc_h[i];j++)
      {
////printf("j %d size %d offs %d i %d Nc %d\n",j,Np,offset,i,Nc_h[i]);
         cellind_h[j]=ind;
      }
      offset+=Nc_h[i];
      ind++;
   }
}

void OrderXpToCell(thrust::device_vector<float4>& xp_d, thrust::device_vector<float4>& xpc_d, thrust::device_vector<int>& CX_d)
{
   thrust::gather(CX_d.begin(), CX_d.end(), xp_d.begin(), xpc_d.begin());
}

void OrderCellToFp(thrust::device_vector<float>& fp_d, thrust::device_vector<float>& fpc_d, thrust::device_vector<int>& XC_d)
{
   int Np=fp_d.size();
   thrust::device_vector<float>fpp_d(Np);
   thrust::gather(XC_d.begin(), XC_d.end(), fpc_d.begin(), fpp_d.begin());
   transform(fp_d.begin(), fp_d.end(), fpp_d.begin(), fp_d.begin(), thrust::plus<float>());
}

__global__ void SumF(float *fp_D, const float* __restrict__ fpG_D, int Np)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   {
      float fpGt=fpG_D[idx];
      float fpt=fp_D[idx];
      fp_D[idx]=fpt+fpGt;
   }
}

void PPField(float4 *xp_D, float *fp_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc)
{
   //int lc=1; //number of cell to consider for PP interaction (0->1, 1->27)
   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   int threads=256;
   int blocks=Np/threads+1;
   int smem=0;//4*threads*sizeof(float4);
   cudaStream_t s1, s2;
   cudaStreamCreate(&s1);
   cudaStreamCreate(&s2);
   float *fpG_D;
   cudaMalloc(&fpG_D, Np*sizeof(float));
   cudaMemset(fpG_D, 0, Np*sizeof(float));
   int *offset_h, *cellind_h, *Nc_h;
   int *offset_d, *cellind_d;
   cudaMallocHost(&offset_h, Nc*sizeof(int));
   cudaMallocHost(&cellind_h, Np*sizeof(int));
   cudaMallocHost(&Nc_h, Nc*sizeof(int));
   cudaMalloc(&offset_d, Nc*sizeof(int));
   cudaMalloc(&cellind_d, Np*sizeof(int));
   cudaMemcpy(Nc_h, Nc_D, Nc*sizeof(int), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
   FillCell(offset_h, cellind_h, Nc_h, Nc, Np);
int sss=0;
for(int i=0;i<Nc;i++) sss+=Nc_h[i];
   cudaMemcpy(offset_d, offset_h, Nc*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cellind_d, cellind_h, Np*sizeof(int), cudaMemcpyHostToDevice);
   cudaFreeHost(offset_h);
   cudaFreeHost(cellind_h);
   cudaFreeHost(Nc_h);

   PPField_ker<<<blocks,threads,smem,s1>>>(xp_D, fp_D, Nc_D, cellind_d, offset_d, Np, Nc, N, lc);

   float4 *rg0_D, *rg1_D;
   cudaMalloc(&rg0_D, Nc*sizeof(float4));
   cudaMalloc(&rg1_D, Nc*sizeof(float4));
   
   float4 *fg0_D, *fg1_D;
   cudaMalloc(&fg0_D, Nc*sizeof(float4));
   cudaMalloc(&fg1_D, Nc*sizeof(float4));

   int Ndc=(lc+2)*(lc+2)*(lc+2);
   int gmem=Ndc*sizeof(float);
   float *dc_h, *dc_D;
   cudaMallocHost(&dc_h, gmem);
   cudaMalloc(&dc_D, gmem);
   CellDistance(dc_h, N, A, B, lc);
   cudaMemcpy(dc_D, dc_h, gmem, cudaMemcpyHostToDevice);

   int blocksG=Nc/threads+1;

size_t free_byte ;
size_t total_byte ;
cudaMemGetInfo( &free_byte, &total_byte ) ;
double free_db = (double)free_byte ;
double total_db = (double)total_byte ;
double used_db = total_db - free_db ;

   CellDensity_ker<<<blocksG,threads,0,s2>>>(xp_D, rg0_D, rg1_D, Nc_D, offset_d, Nc, N, A, B);
   PGField_ker<<<blocksG,threads,gmem,s2>>>(rg0_D, rg1_D, fg0_D, fg1_D, dc_D, Nc, N, A, B, lc, Ndc);
   CellFieldToParticle_ker<<<blocks,threads,0,s2>>>(xp_D, fpG_D, fg0_D, fg1_D, cellind_d, Np, N, A, B);
   cudaDeviceSynchronize();
   SumF<<<blocks,threads>>>(fp_D, fpG_D, Np);

   cudaFree(rg0_D);
   cudaFree(rg1_D);
   cudaFree(fg0_D);
   cudaFree(fg1_D);
   cudaFree(dc_D);
   cudaFreeHost(dc_h);

   cudaFree(fpG_D);
   cudaFree(offset_d);
   cudaFree(cellind_d);

   cudaStreamDestroy(s1);
   cudaStreamDestroy(s2);
}



//PP field from all particles
__global__ void 
__launch_bounds__(512,3)
PPField_Tot_ker(const float4* __restrict__ xp_D, float* __restrict__ fp_D, int Np, int Ntile)
{
   //cg::thread_block cta = cg::this_thread_block();
   //for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx<Np; idx += blockDim.x*gridDim.x) 
   float4 x0=make_float4(0,0,0,0);
   int idx = threadIdx.x + blockDim.x*blockIdx.x;

   float4 x=xp_D[idx];
   float fp=0;
   extern __shared__ float4 xp_S[];
   for (int tile = 0; tile < Ntile; tile++) 
   {
      int ind = tile * blockDim.x + threadIdx.x;
      if(ind<Np) xp_S[threadIdx.x] = xp_D[ind];
      else xp_S[threadIdx.x] = x0;
      __syncthreads();
      //cg::sync(cta);
      #pragma unroll 128
      for(int i=0;i<blockDim.x;i++)
      {
         float4 xp=xp_S[i];
         float dx=x.x-xp.x;
         float dy=x.y-xp.y;
         float dz=x.z-xp.z;
         dx*=dx; dy*=dy; dz*=dz;
         float den=dx+dy+dz;
         //den=sqrtf(den);
         den=(den == 0 || xp.w == 0 ? 0 : sqrtf(den));
         fp+=(den == 0 ? 0 : xp.w/den);
      }
      __syncthreads();
      //cg::sync(cta);
   }
   if(idx< Np)
      fp_D[idx]=fp/(4*MPI);
}

void PPField_Tot(float4 *xp_D, float *fp_D, int Np)
{
   int threads=256;
   int blocks=(Np+threads-1)/threads;

   int Ntile=blocks;
   int smem = threads*sizeof(float4);
   PPField_Tot_ker<<<blocks,threads,smem>>>(xp_D, fp_D, Np, Ntile);
}





__global__ void threshold(int *Nc_D, int *cellind_D, int Nc, int thresh)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Nc)
   {
      int Ncp=Nc_D[idx];
      if(Ncp > thresh)
      {
         cellind_D[idx]=idx;
      }
   }
}

__global__ void cluster(int *cellclust_D, int *cellind_D, int Ncl, int3 N)
{
   __shared__ bool cont[1];
   int Nx=N.x-1, Ny=N.y-1, Nz=N.z-1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Ncl)
   {
      cont[0]=true;
      while(cont[0])
      {
         //if(threadIdx.x==0) cont[0]=false;
         cont[0]=false;
         __syncthreads();
         int ind=cellind_D[idx];
         int i=(int)ind/(Ny*Nz); 
         int j=(int)(ind-i*Ny*Nz)/Nz;
         int k=(int)(ind-i*Ny*Nz-j*Nz);
         int cl=cellclust_D[ind];
         int cli1=-1, clj1=-1, clk1=-1;
         bool con=false;
         if(cl>=0)
         {
            if(i+1<Nx) cli1=cellclust_D[(i+1)*Ny*Nz+j*Nz+k];
            if(cli1>=0 && cli1 != cl)
            {
               con=true;
               if(cli1>cl) cellclust_D[(i+1)*Ny*Nz+j*Nz+k]=cl;
               if(cli1<cl) cl=cli1;
            }
            if(j+1<Ny) clj1=cellclust_D[i*Ny*Nz+(j+1)*Nz+k];
            if(clj1>=0 && clj1 != cl)
            {
               con=true;
               if(clj1>cl) cellclust_D[i*Ny*Nz+(j+1)*Nz+k]=cl;
               if(clj1<cl) cl=clj1;
            }
            if(k+1<Nz) clk1=cellclust_D[i*Ny*Nz+j*Nz+(k+1)];
            if(clk1>=0 && clk1 != cl)
            {
               con=true;
               if(clk1>cl) cellclust_D[i*Ny*Nz+j*Nz+(k+1)]=cl;
               if(clk1<cl) cl=clk1;
            }
            if(con) {cellclust_D[ind]=cl; cont[0]=true;}
         }
         __syncthreads();
      }
   }
}

void cluster_host(thrust::host_vector<int>& cellclust_h, thrust::host_vector<int>& cellind_h, int Ncl, int3 N)
{
   bool cont=true;
   int Nx=N.x-1, Ny=N.y-1, Nz=N.z-1;
   while(cont)
   {
      cont=false;
      for(int idx=0;idx<Ncl;idx++)
      {
         int ind=cellind_h[idx];
         int i=(int)ind/(Ny*Nz); 
         int j=(int)(ind-i*Ny*Nz)/Nz;
         int k=(int)(ind-i*Ny*Nz-j*Nz);
         int cl=cellclust_h[ind];
         int cli1=-1, clj1=-1, clk1=-1;
         bool con=false;
         if(cl>=0)
         {
            if(i+1<Nx) cli1=cellclust_h[(i+1)*Ny*Nz+j*Nz+k];
            if(cli1>=0 && cli1 != cl)
            {
               con=true;
               if(cli1>cl) cellclust_h[(i+1)*Ny*Nz+j*Nz+k]=cl;
               if(cli1<cl) cl=cli1;
            }
            if(j+1<Ny) clj1=cellclust_h[i*Ny*Nz+(j+1)*Nz+k];
            if(clj1>=0 && clj1 != cl)
            {
               con=true;
               if(clj1>cl) cellclust_h[i*Ny*Nz+(j+1)*Nz+k]=cl;
               if(clj1<cl) cl=clj1;
            }
            if(k+1<Nz) clk1=cellclust_h[i*Ny*Nz+j*Nz+(k+1)];
            if(clk1>=0 && clk1 != cl)
            {
               con=true;
               if(clk1>cl) cellclust_h[i*Ny*Nz+j*Nz+(k+1)]=cl;
               if(clk1<cl) cl=clk1;
            }
            if(con) {cellclust_h[ind]=cl; cont=true;}
         }
      }
   }
}

struct isMinus {
    __host__ __device__ bool operator()(int i) const {
        return i < 0;
    }
};

void ParticleCluster(thrust::device_vector<int>& cellind_d, thrust::device_vector<int>& iclus_d, thrust::device_vector<int>& nclus_d, thrust::device_vector<int>& Nc_d,  int3 N, int thresh)
{

   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   int* cellind_D = raw_pointer_cast(&cellind_d[0]);

   int threads=256;
   int blocks=Nc/threads+1;
   int* Nc_D = raw_pointer_cast(&Nc_d[0]);
   threshold<<<blocks,threads>>>(Nc_D, cellind_D, Nc, thresh);
   //cudaDeviceSynchronize();
   
   thrust::device_vector<int> cellclust_d(cellind_d);
   int* cellclust_D = raw_pointer_cast(&cellclust_d[0]);
   thrust::device_vector<int>::iterator iter;
   iter = remove_if(thrust::device, cellind_d.begin(), cellind_d.end(), isMinus());
   cudaDeviceSynchronize();
   cellind_d.resize(iter - cellind_d.begin());
   //cudaDeviceSynchronize();


//         auto start0 = std::chrono::high_resolution_clock::now(); 
   int Ncell=cellind_d.size();
//   threads=512;
//   blocks=Ncell/threads+1;
//   cluster<<<blocks,threads,sizeof(bool)>>>(cellclust_D, cellind_D, Ncell, N);
//   cudaDeviceSynchronize();

   thrust::host_vector<int> cellclust_h(cellclust_d);
   thrust::host_vector<int> cellind_h(cellind_d);
   cluster_host(cellclust_h, cellind_h, Ncell, N);
   cellind_d=cellind_h;
   cellclust_d=cellclust_h;
//         auto finish0 = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed0 = finish0 - start0;
//         std::cout<<"Time: " << elapsed0.count() << " s\n";


   // cluster k = [ cellind_d[iclus_d[k]] - cellind_d[iclus_d[k]+nclus_d[k]] ]
   thrust::device_vector<int>::iterator iter1;
   iter1 = remove_if(thrust::device, cellclust_d.begin(), cellclust_d.end(), isMinus());
   //cudaDeviceSynchronize();
   cellclust_d.resize(iter1 - cellclust_d.begin());
   //cudaDeviceSynchronize();

   sort_by_key(cellclust_d.begin(), cellclust_d.end(), cellind_d.begin());
   //cudaDeviceSynchronize();

   thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> end;
   iclus_d.resize(Ncell);
   nclus_d.resize(Ncell);
   thrust::device_vector<int> one(Ncell,1);
   end = reduce_by_key(thrust::device, cellclust_d.begin(), cellclust_d.end(), one.begin(), iclus_d.begin(), nclus_d.begin());
   //cudaDeviceSynchronize();

   iclus_d.resize(end.first- iclus_d.begin());
   nclus_d.resize(end.second- nclus_d.begin());
   //cudaDeviceSynchronize();
   thrust::host_vector<int> iclus_h(iclus_d.size()), nclus_h(nclus_d);
   int offcell=0;
   for(int i=0;i<nclus_h.size();i++)
   {
      iclus_h[i]=offcell;
      offcell+=nclus_h[i];
   }
   iclus_d=iclus_h;
   //cudaDeviceSynchronize();
//printf("Ncell %d Ncl %d\n",Ncell, iclus_d.size());
//         ofstream ofile;
//         ofile.open("./nclust.bin", ios::out | ios::binary);
//         thrust::host_vector<int> nnclus_h(nclus_d);
//         for(int i=0;i<nnclus_h.size();i++)
//         {
//            int x=nnclus_h[i];
//            ////printf("cluster %d Ncell %d\n", i ,Npclust_h[i]);
//            ofile.write(reinterpret_cast<char *>(&x), sizeof(int));
//         }
//         ofile.close();
//         ofile.open("./iclust.bin", ios::out | ios::binary);
//         thrust::host_vector<int> iiclus_h(iclus_d);
//         for(int i=0;i<iiclus_h.size();i++)
//         {
//            int x=iiclus_h[i];
//            ////printf("cluster %d Ncell %d\n", i ,Npclust_h[i]);
//            ofile.write(reinterpret_cast<char *>(&x), sizeof(int));
//         }
//         ofile.close();
//         ofile.open("./clust.bin", ios::out | ios::binary);
//         thrust::host_vector<int> cellind_h(cellclust_d);
//         for(int i=0;i<cellind_h.size();i++)
//         {
//            int x=cellind_h[i];
//            ofile.write(reinterpret_cast<char *>(&x), sizeof(int));
//         }
//         ofile.close();
}

void FillCluster(thrust::device_vector<float4>& xpc_d, thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fpc_d, thrust::device_vector<float>& fp_d, thrust::host_vector<int>& Nc_h, thrust::host_vector<int>& offset_h, thrust::host_vector<int>& cellind_h, int celloff, int Ncell)
{
   int off=0;
   for(int i=celloff;i<celloff+Ncell;i++)
   {
      int icell = cellind_h[i];
      int start=offset_h[icell];
      int ende=start+Nc_h[icell];
//printf("FillCluster start %d ende %d\n", start, ende);
      copy(thrust::device, xp_d.begin()+start, xp_d.begin()+ende, xpc_d.begin()+off);
      copy(thrust::device, fp_d.begin()+start, fp_d.begin()+ende, fpc_d.begin()+off);
      off+=ende-start;
   }
}

void ClusterToXp(thrust::device_vector<float>& fpc_d, thrust::device_vector<float>& fp_d, thrust::host_vector<int>& Nc_h, thrust::host_vector<int>& offset_h, thrust::host_vector<int>& cellind_h, int celloff, int Ncell)
{
   int off=0;
   for(int i=celloff;i<celloff+Ncell;i++)
   {
      int icell = cellind_h[i];
      int start=offset_h[icell];
      int ende=Nc_h[icell];
      copy(thrust::device, fpc_d.begin()+off, fpc_d.begin()+off+ende, fp_d.begin()+start);
      off+=ende;
   }
}

void save_clust(thrust::device_vector<float4> xpc_d, int i, int Np, float3 A, float3 B, int3 N, float3 cA, float3 cB, int3 cN, float3 fA, float3 fB, int3 fN)
{
   ofstream ofile;
   char fname[50];
   sprintf(fname, "./cluster/xpc%d.bin",i);
   ofile.open(fname, ios::out | ios::binary);
   thrust::host_vector<float4> xp_h(xpc_d);
//printf(fname);
//printf(" np: %d\n",xp_h.size());
   for(int j=0;j<xp_h.size();j++)
   {
      struct X
      {
          float a, b, c;
      } x;
      x.a = xp_h[j].x;
      x.b = xp_h[j].y;
      x.c = xp_h[j].z;
      ofile.write(reinterpret_cast<char *>(&x), sizeof(x));
   }
   ofile.close();
   ofile.open("./param.txt", ios::out | ios::binary | ios::app);
   ofile << "A={"<<A.x<<","<<A.y<<","<<A.z<<"};"<<endl;
   ofile << "B={"<<B.x<<","<<B.y<<","<<B.z<<"};"<<endl;
   ofile << "Ng={"<<N.x<<","<<N.y<<","<<N.z<<"};"<<endl;
   ofile << "Np="<<Np<<";"<<endl<<endl;
   ofile << "cA={"<<cA.x<<","<<cA.y<<","<<cA.z<<"};"<<endl;
   ofile << "cB={"<<cB.x<<","<<cB.y<<","<<cB.z<<"};"<<endl;
   ofile << "cNg={"<<cN.x<<","<<cN.y<<","<<cN.z<<"};"<<endl;
   ofile << "Npc="<<xpc_d.size()<<";"<<endl<<endl;
   ofile << "fA={"<<fA.x<<","<<fA.y<<","<<fA.z<<"};"<<endl;
   ofile << "fB={"<<fB.x<<","<<fB.y<<","<<fB.z<<"};"<<endl;
   ofile << "fNg={"<<fN.x<<","<<fN.y<<","<<fN.z<<"};"<<endl<<endl;
   ofile.close();
}

void RecursiveMeshField(thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fp_d, thrust::device_vector<int>& Nc_d, float3 A, float3 B, int3 N, int Nc, int fgrid, int depth, int depthmax)
{
   depth++;
   if(depth>=depthmax) return;
   int thresh=100;
   thrust::device_vector<int> cellind_d(Nc,-1);
   thrust::device_vector<int> iclus_d;
   thrust::device_vector<int> nclus_d;
   //find clusters
   ParticleCluster(cellind_d, iclus_d, nclus_d, Nc_d, N, thresh);
   int Ncl=iclus_d.size();
   if(Ncl<1) return;
   thrust::host_vector<int> cellind_h(cellind_d), iclus_h(iclus_d), nclus_h(nclus_d);
   thrust::host_vector<int> Nc_h(Nc_d), offset_h(Nc);
   int offcell=0;
   for(int i=0;i<Nc;i++)
   {
      offset_h[i]=offcell;
      offcell+=Nc_h[i];
   }
   //cudaStream_t st[Ncl];
   for(int i=0;i<Ncl;i++)
   {
      //cudaStreamCreate(&st[i]);
      //Host-Device particles thrust vectors
      int celloff=iclus_h[i];
      int Ncell=nclus_h[i];
      int Npc=0;
      for(int i=celloff;i<celloff+Ncell;i++)
      {
         int icell = cellind_h[i];
         Npc+=Nc_h[icell];
      }
      thrust::device_vector<float4> xpc_d(Npc); //particle positions
      //thrust::device_vector<float3> vpc_d(Npc), apc_d(Npc);
      thrust::device_vector<float> fpc_d(Npc, 0); //particle field
      //cast thrust vectors to pointers to be processed by kernels
      float4* xpc_D = raw_pointer_cast(&xpc_d[0]);
      float* fpc_D = raw_pointer_cast(&fpc_d[0]);
      //float3* vpc_D = raw_pointer_cast(&vpc_d[0]);
      //float3* apc_D = raw_pointer_cast(&apc_d[0]);
      //fill xpc with cluster particles
      FillCluster(xpc_d, xp_d, fpc_d, fp_d, Nc_h, offset_h, cellind_h, celloff, Ncell);
      //find mesh tree structure 
      float3 cA, cB; int3 cN; //coarse grid
      float3 fA, fB; int3 fN; //fine grid
      init_mesh_struct_clust(cellind_h, celloff, Ncell, Npc, A, B, N, 
                              cA, cB, cN, fA, fB, fN, fgrid);
      //save_clust(xpc_d, i, Np, A, B, N, cA, cB, cN, fA, fB, fN);

      //Assign particles to coarse mesh density
      if(true)
      {
         //define mesh depending vectors
         int cNtot2=2*cN.x*2*cN.y*(2*cN.z/2+1);
         int cNc=(cN.x-1)*(cN.y-1)*(cN.z-1);
         thrust::device_vector<float> crho_d(cN.x*cN.y*cN.z, 0), cfg_d(cN.x*cN.y*cN.z, 0);
         thrust::device_vector<int> cNc_d(cNc, 0);
         float* crho_D = raw_pointer_cast(&crho_d[0]);
         float* cfg_D = raw_pointer_cast(&cfg_d[0]);
         int* cNc_D = raw_pointer_cast(&cNc_d[0]);
         thrust::device_vector<int> XCc_d(Npc); //cell indeces of particles
         thrust::device_vector<int> CXc_d(Npc); //particles indeces for cell
         int* XCc_D = raw_pointer_cast(&XCc_d[0]);
         int* CXc_D = raw_pointer_cast(&CXc_d[0]);
         ParticleToMesh(xpc_D, crho_D, XCc_D, Npc, cN, cA, cB);
         //Poisson solver in fourier space
         thrust::device_vector<thrust::complex<float> > frho_d(cNtot2), fker_d(cNtot2);
         cufftComplex* frho_D = (cufftComplex*)raw_pointer_cast(&frho_d[0]);
         cufftComplex* fker_D = (cufftComplex*)raw_pointer_cast(&fker_d[0]);
         fft(crho_D, frho_D, fker_D, cA, cB, cN);
         Poisson_Solver(frho_D, fker_D, cN);
         //Poisson_Solver_thrust(frho_d, fker_d);
         fft_inv(frho_D, cfg_D, cN);
         //Assign negative internal mesh field to particles
         MeshToParticle(xpc_D, fpc_D, cfg_D, Npc, cN, cA, cB, -1);
      }   
      //Assign particles to fine mesh density
      if(true)
      {
         //define mesh depending vectors
         int cNtot2=2*fN.x*2*fN.y*(2*fN.z/2+1);
         int cNc=(fN.x-1)*(fN.y-1)*(fN.z-1);
         thrust::device_vector<float> crho_d(fN.x*fN.y*fN.z, 0), cfg_d(fN.x*fN.y*fN.z, 0);
         thrust::device_vector<int> cNc_d(cNc, 0);
         float* crho_D = raw_pointer_cast(&crho_d[0]);
         float* cfg_D = raw_pointer_cast(&cfg_d[0]);
         int* cNc_D = raw_pointer_cast(&cNc_d[0]);
         thrust::device_vector<int> XCc_d(Npc); //cell indeces of particles
         thrust::device_vector<int> CXc_d(Npc); //particles indeces for cell
         int* XCc_D = raw_pointer_cast(&XCc_d[0]);
         int* CXc_D = raw_pointer_cast(&CXc_d[0]);
         ParticleToMesh(xpc_D, crho_D, XCc_D, Npc, fN, fA, fB);
         if(true)
         {
            OrderCell(XCc_d, CXc_d, cNc_d, Npc, fN);
            thrust::device_vector<float4> xpo_d(Npc); //particle positions ordered by cell
            OrderXpToCell(xpc_d, xpo_d, CXc_d);
            xpc_d=xpo_d;
         }
         //Poisson solver in fourier space
         thrust::device_vector<thrust::complex<float> > frho_d(cNtot2), fker_d(cNtot2);
         cufftComplex* frho_D = (cufftComplex*)raw_pointer_cast(&frho_d[0]);
         cufftComplex* fker_D = (cufftComplex*)raw_pointer_cast(&fker_d[0]);
         fft(crho_D, frho_D, fker_D, fA, fB, fN);
         Poisson_Solver(frho_D, fker_D, fN);
         //Poisson_Solver_thrust(frho_d, fker_d);
         fft_inv(frho_D, cfg_D, fN);
         //Assign fine mesh field to particles
         thrust::device_vector<float> fpco_d(Npc, 0); //particle field
         float* fpco_D = raw_pointer_cast(&fpco_d[0]);
         MeshToParticle(xpc_D, fpco_D, cfg_D, Npc, fN, fA, fB, 1);
         RecursiveMeshField(xpc_d, fpco_d, cNc_d, fA, fB, fN, cNc, fgrid, depth, depthmax);
         OrderCellToFp(fpc_d, fpco_d, XCc_d);
      } 
      //Copy cluster field to global field
      ClusterToXp(fpc_d, fp_d, Nc_h, offset_h, cellind_h, celloff, Ncell);
   }
}


