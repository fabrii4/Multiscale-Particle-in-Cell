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
#include <thrust/scatter.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <map>
#include <tuple>
#include <thrust/iterator/counting_iterator.h>

#include <cufft.h>

#include <cooperative_groups.h>

//#include "Poisson.h"


using namespace std;
//using namespace thrust;
namespace cg = cooperative_groups;

#define N_THREADS 512 //number of threads used by CUDA kernels

#define MPI 3.1415926535897932385 //greek pi

//Physical constants (must match the values used in plasma.cu / plasma_pp.cpp)
#define E0    (8.854187817e-12)        //vacuum permittivity  C^2/(N*m^2)
#define M0    (4*MPI*1e-7)             //vacuum permeability  N/A^2
#define QE    (-1.60217733e-19) //electron charge      C   (scaled as in PP)
#define ECHG  (1.60217733e-19)  //elementary charge magnitude |e| (= -QE)
#define ME    (9.1093897e-31)   //electron mass        Kg  (scaled as in PP)
#define MPROT (1.67265e-27)     //proton mass          Kg  (scaled as in PP)
#define MP    (2*MPROT)               //deuteron mass        Kg
#define C3    26944002417373989539335912         // (speed of light)^3      (m/s)^3

//Particle "charge number" convention: xp.w holds the charge of the particle in
//units of the elementary charge |e| (a signed integer, e.g. -1 electron, +1
//proton, +2 alpha; 0 = inert/parked). The internal mesh field is gathered in
//"grid units" and multiplied by ECHG/E0 in the Boris push; the mass of each
//particle is selected from the sign of its charge number (positive species when
//w>0, negative species when w<0).

//External-field system parameters (charged/current rings), mirroring plasma_pp.cpp
#define EXT_AR 0.01f   //m   ring radius
#define EXT_DR 0.01f   //m   ring distance from the origin along z
#define EXT_RINGQ (-2e-9f/1000.f)        //C   total charge of each charged ring (E_ext)
#define EXT_RINGJ (3.5e4f/100.f)      //A   current of each magnetic ring (B_ext current loop)
#define EXT_B0 0.001f                    //T   uniform background field magnitude (B_ext uniform)
#define LH_NP 200000                   //number of samples in the hypergeometric tables

//External-field selectors (passed to the Boris push)
enum { BEXT_NONE=0, BEXT_UNIFORM=1, BEXT_LOOPS=2 };
enum { EEXT_NONE=0, EEXT_RINGS=1 };

//Device-resident hypergeometric tables for the current-loop magnetic field.
//LH1[i]=2F1(0.75,1.25,2,z), LH2[i]=2F1(1.75,2.25,3,z) sampled on z in [-1,1).
static float *LH1_dptr=NULL, *LH2_dptr=NULL; //device handles (allocated by load_ext_tables)

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

//============================ External fields ===============================//
//These reproduce the external electric and magnetic fields of plasma_pp.cpp so
//that the PIC code can be benchmarked against the PP code.

//Index into the hypergeometric tables for argument p (sampled on [-1,1)).
__device__ __forceinline__ int lh_idx(float p)
{
   int i=(int)floorf((p+1.f)*0.5f*LH_NP);
   if(i<0) i=0;
   if(i>=LH_NP) i=LH_NP-1;
   return i;
}

//Electric field of two coaxial uniformly charged rings (radius a, at z=+-d),
//evaluated by summing Na point charges around each ring (as in PP E_ext).
__device__ void Eext_rings(float3 x, float3 &E)
{
   const int Na=100;
   float a=EXT_AR, d=EXT_DR;
   float q=EXT_RINGQ/(4*MPI*E0);
   float ex=0.f, ey=0.f, ez=0.f;
   #pragma unroll 4
   for(int s=0;s<2;s++)               //two rings at +d and -d
   {
      float dz=(s==0? d : -d);
      for(int i=0;i<Na;i++)
      {
         float t=i*2.f*MPI/Na;
         float rx=x.x+a*cosf(t), ry=x.y+a*sinf(t), rz=x.z+dz;
         float norm=powf(rx*rx+ry*ry+rz*rz,1.5f);
         ex+=q*rx/norm; ey+=q*ry/norm; ez+=q*rz/norm;
      }
   }
//   //second larger ring with opposite charge
//   a=5*EXT_AR, d=5*EXT_DR;
//   q=-2*EXT_RINGQ/(4*MPI*E0);
//   #pragma unroll 4
//   for(int s=0;s<2;s++)               //two rings at +d and -d
//   {
//      float dz=(s==0? d : -d);
//      for(int i=0;i<Na;i++)
//      {
//         float t=i*2.f*MPI/Na;
//         float rx=x.x+a*cosf(t), ry=x.y+a*sinf(t), rz=x.z+dz;
//         float norm=powf(rx*rx+ry*ry+rz*rz,1.5f);
//         ex+=q*rx/norm; ey+=q*ry/norm; ez+=q*rz/norm;
//      }
//   }
   E=make_float3(ex,ey,ez);
}

//Magnetic field of a single current loop (radius a) centred at z=-d, built from
//the hypergeometric series solution of plasma_equations.pdf (sec. 4). Adds to B.
__device__ void Bext_loop(float3 x, float a, float d, float J, const float *LH1, const float *LH2, float3 &B)
{
   float rho2=x.x*x.x+x.y*x.y;
   float zz=d+x.z;
   float R2=a*a+rho2+zz*zz;
   float xyz=R2*R2;
   float C=a*a*J*M0/(8.f*powf(xyz,2.25f));
   float p=4.f*a*a*rho2/xyz;
   int id=lh_idx(p);
   float H1=LH1[id], H2=LH2[id];
   float bxy=3.f*C*zz*(2.f*xyz*H1+5.f*a*a*rho2*H2);
   B.x+=bxy*x.x;
   B.y+=bxy*x.y;
   B.z+=0.5f*C*(-4.f*(-2.f*a*a+rho2-2.f*zz*zz)*xyz*H1
               +15.f*a*a*rho2*(a*a-rho2+zz*zz)*H2);
}

//Total external magnetic field selector.
__device__ void Bext(float3 x, int mode, const float *LH1, const float *LH2, float3 &B)
{
   B=make_float3(0.f,0.f,0.f);
   if(mode==BEXT_UNIFORM)
   {
      B.z=EXT_B0;
   }
   else if(mode==BEXT_LOOPS)
   {
      //two coaxial loops at z=+-d (magnetic-bottle / cusp configuration)
      Bext_loop(x, EXT_AR,  EXT_DR, EXT_RINGJ, LH1, LH2, B);
      Bext_loop(x, EXT_AR, -EXT_DR, EXT_RINGJ, LH1, LH2, B);
   }
}

//Total external electric field selector.
__device__ void Eext(float3 x, int mode, float3 &E)
{
   E=make_float3(0.f,0.f,0.f);
   if(mode==EEXT_RINGS)
      Eext_rings(x, E);
}

//===================== Mesh potential -> electric field =====================//
//Central-difference gradient of the grid potential phi (layout i*Ny*Nz+j*Nz+k),
//giving the grid electric field eg = -grad(phi) in the same "grid units" as phi
//(the physical field is recovered by the QE/E0 factor applied in the push).
__global__ void GridField_ker(const float* __restrict__ fg, float3* __restrict__ eg, int3 N, float3 A)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Nx*Ny*Nz)
   {
      int i=idx/(Ny*Nz);
      int j=(idx-i*Ny*Nz)/Nz;
      int k=idx-i*Ny*Nz-j*Nz;
      int ip=min(i+1,Nx-1), im=max(i-1,0);
      int jp=min(j+1,Ny-1), jm=max(j-1,0);
      int kp=min(k+1,Nz-1), km=max(k-1,0);
      float ex=-(fg[ip*Ny*Nz+j*Nz+k]-fg[im*Ny*Nz+j*Nz+k])/((ip-im)*A.x);
      float ey=-(fg[i*Ny*Nz+jp*Nz+k]-fg[i*Ny*Nz+jm*Nz+k])/((jp-jm)*A.y);
      float ez=-(fg[i*Ny*Nz+j*Nz+kp]-fg[i*Ny*Nz+j*Nz+km])/((kp-km)*A.z);
      eg[idx]=make_float3(ex,ey,ez);
   }
}

void GridField(float* fg_D, float3* eg_D, int3 N, float3 A)
{
   int threads=N_THREADS;
   int blocks=(N.x*N.y*N.z)/threads+1;
   GridField_ker<<<blocks,threads>>>(fg_D, eg_D, N, A);
}

//Gather a vector grid field eg to the particles by tri-linear (CIC) weights,
//the vector counterpart of MeshToParticle. sgn lets the caller add (+1) or
//subtract (-1) the contribution (used by the multiscale correction).
__global__ void MeshToParticle3_ker(const float4* __restrict__ xp, float3* __restrict__ Ep, const float3* __restrict__ eg, int Np, int3 N, float3 A, float3 B, int sgn)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   float cden=1./(Ax*Ay*Az);
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   {
      float4 x=xp[idx];
      float3 xc, xcp, axcp;
      int3 ic, scp, ic1, ic2, ic3, ic4, ic5, ic6, ic7;
      ic.x=iX(x.x,Ax,Bx,Nx); ic.y=iX(x.y,Ay,By,Ny); ic.z=iX(x.z,Az,Bz,Nz);
      xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
      xcp.x=x.x-xc.x; xcp.y=x.y-xc.y; xcp.z=x.z-xc.z;
      axcp.x=abs(xcp.x); axcp.y=abs(xcp.y); axcp.z=abs(xcp.z);
      scp.x=sign(xcp.x); scp.y=sign(xcp.y); scp.z=sign(xcp.z);
      ic1.x=ic.x+scp.x; ic1.y=ic.y; ic1.z=ic.z;
      ic2.x=ic.x+scp.x; ic2.y=ic.y+scp.y; ic2.z=ic.z;
      ic3.x=ic.x+scp.x; ic3.y=ic.y+scp.y; ic3.z=ic.z+scp.z;
      ic4.x=ic.x; ic4.y=ic.y+scp.y; ic4.z=ic.z;
      ic5.x=ic.x; ic5.y=ic.y+scp.y; ic5.z=ic.z+scp.z;
      ic6.x=ic.x; ic6.y=ic.y; ic6.z=ic.z+scp.z;
      ic7.x=ic.x+scp.x; ic7.y=ic.y; ic7.z=ic.z+scp.z;
      float a0,a1,a2,a3,a4,a5,a6,a7;
      a0=((Ax-axcp.x)*(Ay-axcp.y)*(Az-axcp.z))*cden;
      a1=(axcp.x*(Ay-axcp.y)*(Az-axcp.z))*cden;
      a2=(axcp.x*axcp.y*(Az-axcp.z))*cden;
      a3=(axcp.x*axcp.y*axcp.z)*cden;
      a4=((Ax-axcp.x)*axcp.y*(Az-axcp.z))*cden;
      a5=((Ax-axcp.x)*axcp.y*axcp.z)*cden;
      a6=((Ax-axcp.x)*(Ay-axcp.y)*axcp.z)*cden;
      a7=(axcp.x*(Ay-axcp.y)*axcp.z)*cden;
      int i0,i1,i2,i3,i4,i5,i6,i7;
      i0=ic.x*Ny*Nz+ic.y*Nz+ic.z; i1=ic1.x*Ny*Nz+ic1.y*Nz+ic1.z;
      i2=ic2.x*Ny*Nz+ic2.y*Nz+ic2.z; i3=ic3.x*Ny*Nz+ic3.y*Nz+ic3.z;
      i4=ic4.x*Ny*Nz+ic4.y*Nz+ic4.z; i5=ic5.x*Ny*Nz+ic5.y*Nz+ic5.z;
      i6=ic6.x*Ny*Nz+ic6.y*Nz+ic6.z; i7=ic7.x*Ny*Nz+ic7.y*Nz+ic7.z;
      int Neg=Nx*Ny*Nz;
      i0=min(max(i0,0),Neg-1); i1=min(max(i1,0),Neg-1);
      i2=min(max(i2,0),Neg-1); i3=min(max(i3,0),Neg-1);
      i4=min(max(i4,0),Neg-1); i5=min(max(i5,0),Neg-1);
      i6=min(max(i6,0),Neg-1); i7=min(max(i7,0),Neg-1);
      float3 e0v=eg[i0], e1v=eg[i1], e2v=eg[i2], e3v=eg[i3];
      float3 e4v=eg[i4], e5v=eg[i5], e6v=eg[i6], e7v=eg[i7];
      float3 e;
      e.x=e0v.x*a0+e1v.x*a1+e2v.x*a2+e3v.x*a3+e4v.x*a4+e5v.x*a5+e6v.x*a6+e7v.x*a7;
      e.y=e0v.y*a0+e1v.y*a1+e2v.y*a2+e3v.y*a3+e4v.y*a4+e5v.y*a5+e6v.y*a6+e7v.y*a7;
      e.z=e0v.z*a0+e1v.z*a1+e2v.z*a2+e3v.z*a3+e4v.z*a4+e5v.z*a5+e6v.z*a6+e7v.z*a7;
      Ep[idx].x += sgn*e.x;
      Ep[idx].y += sgn*e.y;
      Ep[idx].z += sgn*e.z;
   }
}

void MeshToParticle3(float4* xp, float3* Ep, float3* eg, int Np, int3 N, float3 A, float3 B, int sgn)
{
   int threads=N_THREADS;
   int blocks=Np/threads+1;
   MeshToParticle3_ker<<<blocks,threads>>>(xp, Ep, eg, Np, N, A, B, sgn);
}

//Convenience wrapper: from a grid potential fg compute eg=-grad(phi) and gather
//it to the particles (sgn-weighted). Uses a scratch float3 grid.
void MeshFieldToParticle(float4* xp_D, float3* Ep_D, float* fg_D, int Np, int3 N, float3 A, float3 B, int sgn)
{
   float3 *eg_D;
   gpuErrchk(cudaMalloc(&eg_D, N.x*N.y*N.z*sizeof(float3)));
   GridField(fg_D, eg_D, N, A);
   MeshToParticle3(xp_D, Ep_D, eg_D, Np, N, A, B, sgn);
   cudaFree(eg_D);
}

//=============================== Boris push =================================//
//Boris integrator (plasma_equations.pdf sec 3.1). Eg is the *internal* mesh
//electric field in grid units (-grad(phi)); the physical internal field is
//(QE/E0)*Eg. External E/B fields are evaluated analytically per particle. The
//particle "charge sign" is stored in xp.w (+1 electron, -1 deuteron); mass and
//signed charge follow from it.
__global__ void Boris_ker(float4* __restrict__ xp, float3* __restrict__ vp, float3* __restrict__ ap, const float3* __restrict__ Eg, const float3* __restrict__ Bint, const float3* __restrict__ Eind, int Np, float dt, int bext_mode, int eext_mode, float kEint, float mpos, float mneg, const float* __restrict__ LH1, const float* __restrict__ LH2, int radiative_corr)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      float4 x=xp[idx];
      float3 xi=make_float3(x.x,x.y,x.z);
      float z=x.w;                              //charge number in units of |e|
      float mass=(z>0.f? mpos : mneg);          //positive / negative species mass
      float qom=z*ECHG/mass;                    //q/m  (q = z*|e|)

      //total electric field: external + internal mesh (grid units -> physical via
      //kEint) + optional retarded/inductive internal field (already physical).
      float3 Ee, Be; Eext(xi, eext_mode, Ee); Bext(xi, bext_mode, LH1, LH2, Be);
      float3 eg=Eg[idx];
      float3 E=make_float3(Ee.x+kEint*eg.x, Ee.y+kEint*eg.y, Ee.z+kEint*eg.z);
      if(Eind)
      {
         float3 ei=Eind[idx];
         E.x+=ei.x; E.y+=ei.y; E.z+=ei.z;
      }
      float3 B=Be;
      if(Bint)
      {
         float3 bi=Bint[idx];
         B.x+=bi.x; B.y+=bi.y; B.z+=bi.z;
      }

      //Boris rotation
      float3 v=vp[idx];
      float3 ef=make_float3(qom*dt*0.5f*E.x, qom*dt*0.5f*E.y, qom*dt*0.5f*E.z);
      float3 t =make_float3(qom*dt*0.5f*B.x, qom*dt*0.5f*B.y, qom*dt*0.5f*B.z);
      float t2=t.x*t.x+t.y*t.y+t.z*t.z;
      float3 sv=make_float3(2.f/(1.f+t2)*t.x, 2.f/(1.f+t2)*t.y, 2.f/(1.f+t2)*t.z);
      float3 v1=make_float3(v.x+ef.x, v.y+ef.y, v.z+ef.z);
      float3 v2=make_float3(v1.x+v1.y*t.z-v1.z*t.y,
                            v1.y+v1.z*t.x-v1.x*t.z,
                            v1.z+v1.x*t.y-v1.y*t.x);
      float3 v3=make_float3(v1.x+v2.y*sv.z-v2.z*sv.y,
                            v1.y+v2.z*sv.x-v2.x*sv.z,
                            v1.z+v2.x*sv.y-v2.y*sv.x);
      float3 vp1=make_float3(v3.x+ef.x, v3.y+ef.y, v3.z+ef.z);

      ap[idx]=make_float3((vp1.x-v.x)/dt,(vp1.y-v.y)/dt,(vp1.z-v.z)/dt);
      vp[idx]=vp1;
      xp[idx]=make_float4(x.x+vp1.x*dt, x.y+vp1.y*dt, x.z+vp1.z*dt, z);
      
      //Larmor radiation loss approximation (subtract radiated energy from current kinetic energy)
      if(radiative_corr) 
      {
         float a2 = ap->x*ap->x + ap->y*ap->y + ap->z*ap->z;
         float Prad = z*ECHG*z*ECHG * a2 / (6.f * MPI * E0 * C3);
         float K = 0.5f * mass * (vp->x*vp->x + vp->y*vp->y + vp->z*vp->z);
         float dE = Prad * dt;
         float factor = (K > 0.f ? max(0.f, sqrtf(max(0.f, K - dE) / K)) : 0.f);
         vp[idx].x *= factor;
         vp[idx].y *= factor;
         vp[idx].z *= factor;
      }
      
   }
}

//Host wrapper. Bint_D and Eind_D may be NULL to disable the internal magnetic
//field and the retarded/inductive internal electric field respectively. mpos and
//mneg are the masses (kg) of the positive- and negative-charge species.
void BorisPush(float4* xp_D, float3* vp_D, float3* ap_D, float3* Eg_D, float3* Bint_D, float3* Eind_D, int Np, float dt, int bext_mode, int eext_mode, float mpos, float mneg, int radiative_corr)
{
   int threads=N_THREADS;
   int blocks=Np/threads+1;
   float kEint=(float)(ECHG/E0);  //grid-units internal field -> physical field
   Boris_ker<<<blocks,threads>>>(xp_D, vp_D, ap_D, Eg_D, Bint_D, Eind_D, Np, dt, bext_mode, eext_mode, kEint, mpos, mneg, LH1_dptr, LH2_dptr, radiative_corr);
}

//===================== Temporal multiscale (sub-cycling) ====================//
//Cell index (clamped to the valid range) of a particle, matching ParticleToMesh.
__device__ int cellIndexClamped(float4 x, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   int icx=iX(x.x,A.x,B.x,Nx), icy=iX(x.y,A.y,B.y,Ny), icz=iX(x.z,A.z,B.z,Nz);
   int i_c=min(icx, icx+sign(x.x-xI(icx,A.x,B.x,Nx)));
   int j_c=min(icy, icy+sign(x.y-xI(icy,A.y,B.y,Ny)));
   int k_c=min(icz, icz+sign(x.z-xI(icz,A.z,B.z,Nz)));
   i_c=max(0,min(i_c,Nx-2)); j_c=max(0,min(j_c,Ny-2)); k_c=max(0,min(k_c,Nz-2));
   return i_c*(Ny-1)*(Nz-1)+j_c*(Nz-1)+k_c;
}

//Sub-cycling Boris push: a particle in a DENSE cell (count > thresh, i.e. the same
//cells the multiscale solver refines spatially) is advanced with nsub sub-steps
//of dt/nsub, re-evaluating the (sharp, fast-varying) external field at each
//sub-step; particles in sparse cells take a single dt step. The internal mesh
//field (Eg/Bint/Eind) is frozen over the step, as in a standard PIC cycle. This
//is the temporal counterpart of the spatial refinement in dense regions.
__global__ void BorisSub_ker(float4* __restrict__ xp, float3* __restrict__ vp, float3* __restrict__ ap, const float3* __restrict__ Eg, const float3* __restrict__ Bint, const float3* __restrict__ Eind, const int* __restrict__ Nc_D, int Np, int3 N, float3 A, float3 B, float dt, int nsub_dense, int thresh, int bext_mode, int eext_mode, float kEint, float mpos, float mneg, const float* __restrict__ LH1, const float* __restrict__ LH2, int radiative_corr)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      float4 x=xp[idx];
      float z=x.w;
      float mass=(z>0.f? mpos : mneg);
      float qom=z*ECHG/mass;
      float3 eg=Eg[idx];
      float3 ei = Eind? Eind[idx] : make_float3(0,0,0);
      float3 bi = Bint? Bint[idx] : make_float3(0,0,0);
      int ic=cellIndexClamped(x, N, A, B);
      int nsub = (Nc_D[ic] > thresh) ? nsub_dense : 1;
      float dts = dt/nsub;
      float3 v=vp[idx], v0=v, pos=make_float3(x.x,x.y,x.z);
      for(int s=0;s<nsub;s++)
      {
         float3 Ee, Be; Eext(pos, eext_mode, Ee); Bext(pos, bext_mode, LH1, LH2, Be);
         float3 E=make_float3(Ee.x+kEint*eg.x+ei.x, Ee.y+kEint*eg.y+ei.y, Ee.z+kEint*eg.z+ei.z);
         float3 Bf=make_float3(Be.x+bi.x, Be.y+bi.y, Be.z+bi.z);
         float3 ef=make_float3(qom*dts*0.5f*E.x, qom*dts*0.5f*E.y, qom*dts*0.5f*E.z);
         float3 t =make_float3(qom*dts*0.5f*Bf.x, qom*dts*0.5f*Bf.y, qom*dts*0.5f*Bf.z);
         float t2=t.x*t.x+t.y*t.y+t.z*t.z;
         float3 sv=make_float3(2.f/(1.f+t2)*t.x, 2.f/(1.f+t2)*t.y, 2.f/(1.f+t2)*t.z);
         float3 v1=make_float3(v.x+ef.x, v.y+ef.y, v.z+ef.z);
         float3 v2=make_float3(v1.x+v1.y*t.z-v1.z*t.y, v1.y+v1.z*t.x-v1.x*t.z, v1.z+v1.x*t.y-v1.y*t.x);
         float3 v3=make_float3(v1.x+v2.y*sv.z-v2.z*sv.y, v1.y+v2.z*sv.x-v2.x*sv.z, v1.z+v2.x*sv.y-v2.y*sv.x);
         v=make_float3(v3.x+ef.x, v3.y+ef.y, v3.z+ef.z);
         pos.x+=v.x*dts; pos.y+=v.y*dts; pos.z+=v.z*dts;
      }
      ap[idx]=make_float3((v.x-v0.x)/dt,(v.y-v0.y)/dt,(v.z-v0.z)/dt);
      vp[idx]=v;
      xp[idx]=make_float4(pos.x,pos.y,pos.z,z);
      
      //Larmor radiation loss approximation (subtract radiated energy from current kinetic energy)
      if(radiative_corr) 
      {
         float a2 = ap->x*ap->x + ap->y*ap->y + ap->z*ap->z;
         float Prad = z*ECHG*z*ECHG * a2 / (6.f * MPI * E0 * C3);
         float K = 0.5f * mass * (vp->x*vp->x + vp->y*vp->y + vp->z*vp->z);
         float dE = Prad * dt;
         float factor = (K > 0.f ? max(0.f, sqrtf(max(0.f, K - dE) / K)) : 0.f);
         vp[idx].x *= factor;
         vp[idx].y *= factor;
         vp[idx].z *= factor;
      }
   }
}

void BorisSubPush(float4* xp_D, float3* vp_D, float3* ap_D, float3* Eg_D, float3* Bint_D, float3* Eind_D, int* Nc_D, int Np, int3 N, float3 A, float3 B, float dt, int nsub, int thresh, int bext_mode, int eext_mode, float mpos, float mneg, int radiative_corr)
{
   int threads=N_THREADS, blocks=Np/threads+1;
   float kEint=(float)(ECHG/E0);
   BorisSub_ker<<<blocks,threads>>>(xp_D, vp_D, ap_D, Eg_D, Bint_D, Eind_D, Nc_D, Np, N, A, B, dt, nsub, thresh, bext_mode, eext_mode, kEint, mpos, mneg, LH1_dptr, LH2_dptr, radiative_corr);
}

//Load the hypergeometric tables from disk and upload them to the device.
void load_ext_tables(const char* f1, const char* f2)
{
   float *h1=new float[LH_NP];
   float *h2=new float[LH_NP];
   ifstream i1(f1, ios::in|ios::binary), i2(f2, ios::in|ios::binary);
   if(!i1.is_open() || !i2.is_open())
   {
      cerr<<"WARNING: hypergeometric tables ("<<f1<<", "<<f2<<") not found; "
            "current-loop B field will be unavailable."<<endl;
      delete[] h1; delete[] h2; return;
   }
   i1.read(reinterpret_cast<char*>(h1), LH_NP*sizeof(float));
   i2.read(reinterpret_cast<char*>(h2), LH_NP*sizeof(float));
   gpuErrchk(cudaMalloc(&LH1_dptr, LH_NP*sizeof(float)));
   gpuErrchk(cudaMalloc(&LH2_dptr, LH_NP*sizeof(float)));
   gpuErrchk(cudaMemcpy(LH1_dptr, h1, LH_NP*sizeof(float), cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(LH2_dptr, h2, LH_NP*sizeof(float), cudaMemcpyHostToDevice));
   delete[] h1; delete[] h2;
}

void free_ext_tables()
{
   if(LH1_dptr) cudaFree(LH1_dptr);
   if(LH2_dptr) cudaFree(LH2_dptr);
   LH1_dptr=LH2_dptr=NULL;
}

//=========================== Particle loss (GPU) ============================//
//Remove particles that cross a magnet ring or leave the box (|x|>Rmax) or become
//non-finite, parking them inertly at the origin (charge number 0). Done on the
//device so the very-large-Np runs are not bottlenecked by a per-step host copy +
//CPU scan; the number of newly lost particles is accumulated in nlost (device).
__global__ void Loss_ker(float4* __restrict__ xp, float3* __restrict__ vp, float3* __restrict__ ap, int Np, int* __restrict__ nlost, float Rmax2, float drb, float drt, float ar2b, float ar2t)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      float4 x=xp[idx];
      if(x.w==0.f) return;                       //already inert
      float X=x.x, Y=x.y, Z=x.z;
      float dist2=X*X+Y*Y, r2=dist2+Z*Z;
      bool hitring=(((Z>drb && Z<drt)||(Z<-drb && Z>-drt)) && dist2>ar2b && dist2<ar2t);
      bool escaped=!isfinite(r2) || r2>Rmax2;
      if(hitring||escaped)
      {
         atomicAdd(nlost,1);
         xp[idx]=make_float4(0,0,0,0);
         vp[idx]=make_float3(0,0,0);
         ap[idx]=make_float3(0,0,0);
      }
   }
}

void HandleLoss(float4* xp_D, float3* vp_D, float3* ap_D, int Np, int* nlost_D, float Rmax, float drb, float drt, float ar2b, float ar2t)
{
   int threads=N_THREADS, blocks=Np/threads+1;
   Loss_ker<<<blocks,threads>>>(xp_D, vp_D, ap_D, Np, nlost_D, Rmax*Rmax, drb, drt, ar2b, ar2t);
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

      //calculate cell index (clamped to the valid cell range [0,N-2] so that a
      //particle sitting on the last grid line cannot produce an out-of-range
      //cell, which would corrupt the per-cell ordering/counts).
      int i_c=min(ic.x,ic1.x);
      int j_c=min(ic.y,ic4.y);
      int k_c=min(ic.z,ic6.z);
      i_c=max(0,min(i_c,Nx-2));
      j_c=max(0,min(j_c,Ny-2));
      k_c=max(0,min(k_c,Nz-2));
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
      //clamp the stencil indices to the grid (safety net for boundary particles)
      int Nrho=Nx*Ny*Nz;
      i0=min(max(i0,0),Nrho-1); i1=min(max(i1,0),Nrho-1);
      i2=min(max(i2,0),Nrho-1); i3=min(max(i3,0),Nrho-1);
      i4=min(max(i4,0),Nrho-1); i5=min(max(i5,0),Nrho-1);
      i6=min(max(i6,0),Nrho-1); i7=min(max(i7,0),Nrho-1);
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
   int threads=N_THREADS;
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
      int Nfg=Nx*Ny*Nz;
      i0=min(max(i0,0),Nfg-1); i1=min(max(i1,0),Nfg-1);
      i2=min(max(i2,0),Nfg-1); i3=min(max(i3,0),Nfg-1);
      i4=min(max(i4,0),Nfg-1); i5=min(max(i5,0),Nfg-1);
      i6=min(max(i6,0),Nfg-1); i7=min(max(i7,0),Nfg-1);
      float fgt= fg[i0]*a0+fg[i1]*a1+fg[i2]*a2+fg[i3]*a3+
                 fg[i4]*a4+fg[i5]*a5+fg[i6]*a6+fg[i7]*a7;
      fp[idx] += sgn*fgt;
   }
}

void MeshToParticle(float4 *xp, float *fp, float *fg, int Np, int3 N, float3 A, float3 B, int sgn)
{
   int threads=N_THREADS;
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

//particle bounding box reduced in a single device pass (replaces the old
//per-step host copy + CPU min/max loop, which did not scale with Np).
struct BBox { float xm,xM,ym,yM,zm,zM; };
struct ToBBox { __host__ __device__ BBox operator()(const float4& p) const
   { BBox b; b.xm=b.xM=p.x; b.ym=b.yM=p.y; b.zm=b.zM=p.z; return b; } };
struct MergeBBox { __host__ __device__ BBox operator()(const BBox& a, const BBox& b) const
   { BBox r; r.xm=fminf(a.xm,b.xm); r.xM=fmaxf(a.xM,b.xM);
             r.ym=fminf(a.ym,b.ym); r.yM=fmaxf(a.yM,b.yM);
             r.zm=fminf(a.zm,b.zm); r.zM=fmaxf(a.zM,b.zM); return r; } };

void init_mesh_struct(thrust::device_vector<float4>& xp_d, int Np, float3& A, float3& B, int3& N, float fgrid)
{
   //particle bounding box, computed on the device in one reduction pass
   BBox init; init.xm=init.ym=init.zm=1e30f; init.xM=init.yM=init.zM=-1e30f;
   BBox bb=thrust::transform_reduce(thrust::device, xp_d.begin(), xp_d.end(), ToBBox(), init, MergeBBox());
   float xm=bb.xm, xM=bb.xM, ym=bb.ym, yM=bb.yM, zm=bb.zm, zM=bb.zM;

   float Dx=(xM-xm), Dy=(yM-ym), Dz=(zM-zm);
   float bd=0.1; //border 0.07
   xm+=-Dx*bd; xM+=Dx*bd;
   ym+=-Dy*bd; yM+=Dy*bd;
   zm+=-Dz*bd; zM+=Dz*bd;
   Dx=(xM-xm); Dy=(yM-ym); Dz=(zM-zm);
   float Ntot=(float)Np/fgrid;
   float rhoL=cbrt(Ntot/(Dx*Dy*Dz));
   float Nx=round(rhoL*Dx), Ny=round(rhoL*Dy), Nz=round(rhoL*Dz);
   //Clamp the grid to a sane range: a degenerate (collapsed) or non-finite
   //particle distribution can otherwise produce a huge/overflowing N and a
   //catastrophic allocation. The per-dimension count is kept in [2, NMAX].
   const int NMAX=256;
   if(!(Nx>=2)) Nx=2; if(Nx>NMAX) Nx=NMAX;
   if(!(Ny>=2)) Ny=2; if(Ny>NMAX) Ny=NMAX;
   if(!(Nz>=2)) Nz=2; if(Nz>NMAX) Nz=NMAX;
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

//Initialise the particles, mirroring the configuration of plasma_pp.cpp so the
//two codes can be compared. The 4th position component stores the particle's
//charge NUMBER (qneg for the negative species, qpos for the positive species,
//both in units of |e|); the actual charge/mass are recovered from it in the
//push kernel. The negative species (index < Nneg) starts in a small central
//sphere, the positive species in two polar caps of a shell. Velocities are
//uniform random.
void init_p(thrust::host_vector<float4>& xp, thrust::host_vector<float3>& vp, thrust::host_vector<float3>& ap, int Nneg, int qpos, int qneg)
{
   int Np=xp.size();
   float vmax=1000;
   //negative particles: central sphere of radius 0.003 m
   for(int i=0;i<Np;i++)
   {
      float d=(float)rand()/RAND_MAX*0.003f;
      float t=(float)rand()/RAND_MAX*2.f*MPI;
      float f=(float)rand()/RAND_MAX*MPI;
      float q = i<Nneg? (float)qneg : (float)qpos;
      xp[i]=make_float4(d*cos(t)*sin(f), d*sin(t)*sin(f), d*cos(f), q);
      float r1=(float)rand()/RAND_MAX*vmax*2-vmax;
      float r2=(float)rand()/RAND_MAX*vmax*2-vmax;
      float r3=(float)rand()/RAND_MAX*vmax*2-vmax;
      vp[i]=make_float3(r1,r2,r3);
      ap[i]=make_float3(0,0,0);
   }
   //positive particles (index >= Nneg): two polar caps of a shell
   for(int i=Nneg;i<Np;i++)
   {
      float d0=0.04f;
      float d=(float)rand()/RAND_MAX*0.001f+d0;
      float t=(float)rand()/RAND_MAX*2.f*MPI;
      float f1=(float)rand()/RAND_MAX*MPI/5;
      float f=(float)rand()/RAND_MAX*MPI/5+4*MPI/5;
      if(rand()%2>0) f=f1;
      xp[i]=make_float4(d*cos(t)*sin(f), d*sin(t)*sin(f), d*cos(f), (float)qpos);
   }
}

//Append the current particle positions to the trajectory buffer (PP-style:
//traj has Nsave frames of Np particles, 3 floats each).
void store_traj(std::vector<float>& traj, thrust::device_vector<float4>& xp_d, int frame, int Np)
{
   thrust::host_vector<float4> xp_h(xp_d);
   for(int i=0;i<Np;i++)
   {
      traj[(size_t)frame*Np*3+i*3+0]=xp_h[i].x;
      traj[(size_t)frame*Np*3+i*3+1]=xp_h[i].y;
      traj[(size_t)frame*Np*3+i*3+2]=xp_h[i].z;
   }
}

//Write the trajectory buffer in the same binary layout produced by plasma_pp.cpp
//(./gnuplot/results.bin can be visualised with the provided gnuplot scripts).
void save_traj_time(std::vector<float>& traj, int Nsave, int Np, const char* fname)
{
   ofstream ofile(fname, ios::out | ios::binary);
   for(int s=0;s<Nsave;s++)
      for(int i=0;i<Np;i++)
      {
         struct { float a,b,c; } x;
         x.a=traj[(size_t)s*Np*3+i*3+0];
         x.b=traj[(size_t)s*Np*3+i*3+1];
         x.c=traj[(size_t)s*Np*3+i*3+2];
         ofile.write(reinterpret_cast<char*>(&x), sizeof(x));
      }
   ofile.close();
}

//Stream one trajectory frame straight to disk (binary x,y,z per particle, the
//same layout as plasma_pp.cpp) AND accumulate the per-sign radial number-density
//profile. Writing each frame as it is produced avoids holding the whole
//Nsave*Np trajectory in host memory, which is what made large particle counts
//run out of RAM. Positive-charge particles (deuterons, sign w<0) go to dpos,
//negative-charge particles (electrons, w>0) to dneg; inert/parked particles
//(w==0) and particles beyond rmax are ignored.
void store_frame(std::ofstream& fout, std::vector<float>& dpos, std::vector<float>& dneg,
                 thrust::device_vector<float4>& xp_d, int frame, int Np, int Nbins, float rmax)
{
   thrust::host_vector<float4> xp_h(xp_d);
   std::vector<float> buf((size_t)Np*4);
   for(int i=0;i<Np;i++)
   {
      float x=xp_h[i].x, y=xp_h[i].y, z=xp_h[i].z, w=xp_h[i].w;
      buf[(size_t)i*4+0]=x; buf[(size_t)i*4+1]=y; buf[(size_t)i*4+2]=z; buf[(size_t)i*4+3]=w;
      if(w==0.f) continue;
      float r=sqrtf(x*x+y*y+z*z);
      if(r>=rmax) continue;
      int b=(int)(r/rmax*Nbins); if(b<0) b=0; if(b>=Nbins) b=Nbins-1;
      if(w>0.f) dpos[(size_t)frame*Nbins+b]+=1.f;   //positive-charge species
      else      dneg[(size_t)frame*Nbins+b]+=1.f;   //negative-charge species
   }
   fout.write(reinterpret_cast<char*>(buf.data()), (std::streamsize)Np*4*sizeof(float));
}

//Write the time-resolved radial number densities to a text file in gnuplot
//pm3d-friendly layout: columns "t r n_pos n_neg", one block per time frame
//separated by a blank line. The count in each radial shell is divided by the
//shell volume to give a number density (particles / m^3).
void save_density(std::vector<float>& dpos, std::vector<float>& dneg, int Nframes,
                  int Nbins, float rmax, double dt, int save_step, const char* fname)
{
   ofstream ofile(fname, ios::out);
   ofile<<"# t(s)  r(m)  n_pos(1/m^3)  n_neg(1/m^3)\n";
   double dr=(double)rmax/Nbins;
   for(int f=0;f<Nframes;f++)
   {
      double t=(double)f*save_step*dt;
      for(int b=0;b<Nbins;b++)
      {
         double r0=b*dr, r1=(b+1)*dr, rc=(b+0.5)*dr;
         double vol=4.0/3.0*MPI*(r1*r1*r1-r0*r0*r0);
         double np=dpos[(size_t)f*Nbins+b]/vol;
         double nn=dneg[(size_t)f*Nbins+b]/vol;
         ofile<<t<<" "<<rc<<" "<<np<<" "<<nn<<"\n";
      }
      ofile<<"\n";   //blank line between time blocks for gnuplot pm3d
   }
   ofile.close();
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


//cuFFT plan caches, keyed by the (doubled) grid dimensions. Creating a cuFFT
//plan is expensive and the same grid sizes recur every step (and across the
//multiscale clusters), so the plans are created once and reused for the lifetime
//of the program instead of being rebuilt on every fft()/fft_inv() call.
static cufftHandle getR2CPlan(int Nx, int Ny, int Nz)
{
   static std::map<std::tuple<int,int,int>, cufftHandle> cache;
   auto key=std::make_tuple(Nx,Ny,Nz);
   auto it=cache.find(key);
   if(it!=cache.end()) return it->second;
   cufftHandle p;
   int n[3]={Nx,Ny,Nz};
   int idist=Nx*Ny*Nz, odist=Nx*Ny*(Nz/2+1);
   int inembed[]={Nx,Ny,Nz}, onembed[]={Nx,Ny,Nz/2+1};
   cufftPlanMany(&p,3,n,inembed,1,idist,onembed,1,odist,CUFFT_R2C,1);
   cache[key]=p;
   return p;
}
static cufftHandle getC2RPlan(int Nx, int Ny, int Nz)
{
   static std::map<std::tuple<int,int,int>, cufftHandle> cache;
   auto key=std::make_tuple(Nx,Ny,Nz);
   auto it=cache.find(key);
   if(it!=cache.end()) return it->second;
   cufftHandle p;
   int n[3]={Nx,Ny,Nz};
   int idist=Nx*Ny*Nz, odist=Nx*Ny*(Nz/2+1);
   int inembed[]={Nx,Ny,Nz}, onembed[]={Nx,Ny,Nz/2+1};
   cufftPlanMany(&p,3,n,onembed,1,odist,inembed,1,idist,CUFFT_C2R,1);
   cache[key]=p;
   return p;
}

void fft(float *rho_D, cufftComplex *frho_D, cufftComplex *fker_D, float3 A, float3 B, int3 N)
{
   int Nx=2*N.x;
   int Ny=2*N.y;
   int Nz=2*N.z;
   int Ntot=Nx*Ny*Nz;

   // cuFFT 3D plan (cached by grid size; created once, reused thereafter)
   cufftHandle f_plan = getR2CPlan(Nx, Ny, Nz);
   cufftHandle f1_plan = f_plan;

   //input vectors
   int threads=N_THREADS;
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
   //f_plan is cached and reused; not destroyed here
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
//   int threads=N_THREADS;
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
   int threads=N_THREADS;
   int blocks=Ntot/threads+1;
   solver<<<blocks,threads>>>(frho_D, fker_D, Ntot);
}

void Poisson_Solver_thrust(thrust::device_vector<thrust::complex<float> >& frho_d, thrust::device_vector<thrust::complex<float> >& fker_d)
{
   thrust::transform(frho_d.begin(), frho_d.end(), fker_d.begin(), frho_d.begin(), 
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

   // cuFFT 3D plan (cached by grid size; created once, reused thereafter)
   cufftHandle i_plan = getC2RPlan(Nx, Ny, Nz);

   //input vectors
   cufftReal *in_D;
   cudaMalloc(&in_D, Ntot*sizeof(cufftReal));

   //Compute inverse FFT
   cufftExecC2R(i_plan, frho_D, in_D);
   int threads=N_THREADS;
   int blocks=(N.x*N.y*N.z)/threads+1;
   copy_out<<<blocks,threads>>>(rho_D, in_D, N);

   //i_plan is cached and reused; not destroyed here
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
         ofile.open("./grid.txt", ios::out | ios::out);
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
         ofile.open("./grid.txt", ios::out | ios::out);
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
   thrust::device_vector<int> IP_d(Np); 
   thrust::device_vector<int> PI_d(Np);
   thrust::sequence(IP_d.begin(), IP_d.end());
   thrust::copy(IP_d.begin(), IP_d.end(), PI_d.begin());
   sort_by_key(thrust::device, XCt_d.begin(), XCt_d.end(), IP_d.begin());
   copy(thrust::device, IP_d.begin(), IP_d.end(), CX_d.begin());
   //CX_d=IP_d;
   //XC_d
   thrust::device_vector<int> PIt_d(IP_d);
   thrust::copy(IP_d.begin(), IP_d.end(), PIt_d.begin());
   sort_by_key(thrust::device, PIt_d.begin(), PIt_d.end(), PI_d.begin());
   XC_d=PI_d;
   //compute number of particles per cell Npc and cells offset 
   thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> end;
   thrust::device_vector<int> offset_d(Np);
   thrust::device_vector<int> Npc_d(Np);
   thrust::device_vector<int> one_d(Np,1);
   end = reduce_by_key(thrust::device, XCt_d.begin(), XCt_d.end(), one_d.begin(), offset_d.begin(), Npc_d.begin());
   offset_d.resize(end.first- offset_d.begin());
   Npc_d.resize(end.second- Npc_d.begin());
   //compute Nc_d from Npc and offset.
   //offset_d holds the (sorted, unique) indices of the cells that contain at
   //least one particle and Npc_d the corresponding particle counts. We scatter
   //those counts into their cell slots, leaving empty cells at 0. (The previous
   //implementation used a gather with a map set to -1 for empty cells, which
   //read Npc_d[-1] out of bounds and filled empty cells with garbage. That was
   //harmless when most cells were populated but corrupted the per-cell counts
   //on the sparse fine grids produced at recursion depth >= 2, leading to a
   //runaway cluster size and an illegal memory access.)
   thrust::fill(thrust::device, Nc_d.begin(), Nc_d.end(), 0);
   thrust::scatter(thrust::device, Npc_d.begin(), Npc_d.end(), offset_d.begin(), Nc_d.begin());
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
   thrust::transform(fp_d.begin(), fp_d.end(), fpp_d.begin(), fp_d.begin(), thrust::plus<float>());
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
   int threads=N_THREADS;
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
   //cudaDeviceSynchronize();
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
   //cudaDeviceSynchronize();
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

//==================== GPU cell-multipole electric force ======================//
//A cell-multipole short-range solver for the FORCE, fully on the GPU. Each cell's
//charge is condensed onto its 8 corners (CellDensity_ker, reused), the electric
//force at the corners due to the corners of the neighbouring cells (within lc) is
//computed corner-to-corner, and finally interpolated back to the particles. This
//replaces the per-particle-pair sum of PPFieldF by a per-cell (8x8 corner) sum,
//so its cost is independent of how many particles a cell holds - much cheaper for
//densely populated cells. The result is in grid units and added to Ep, exactly
//like the mesh and the direct PP correction.

//Interpolate the 8 corner force vectors of a cell to a particle (CIC weights).
__device__ void CellToParticleF(float4 x, float3 &fp, const float3 fg[], int i_cell, int3 N, float3 A, float3 B)
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   int Nyc=N.y-1, Nzc=N.z-1;
   float den=Ax*Ay*Az;
   int3 ic;
   ic.x=(int)(i_cell)/(Nyc*Nzc);
   ic.y=(int)(i_cell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=(int)(i_cell-ic.x*Nyc*Nzc-ic.y*Nzc);
   float3 xc, xcp;
   xc.x=xI(ic.x,Ax,Bx,Nx); xc.y=xI(ic.y,Ay,By,Ny); xc.z=xI(ic.z,Az,Bz,Nz);
   xcp.x=x.x-xc.x; xcp.y=x.y-xc.y; xcp.z=x.z-xc.z;
   float a0,a1,a2,a3,a4,a5,a6,a7;
   a0=((Ax-abs(xcp.x))*(Ay-abs(xcp.y))*(Az-abs(xcp.z)))/den;
   a1=(abs(xcp.x)*(Ay-abs(xcp.y))*(Az-abs(xcp.z)))/den;
   a2=(abs(xcp.x)*abs(xcp.y)*(Az-abs(xcp.z)))/den;
   a3=(abs(xcp.x)*abs(xcp.y)*abs(xcp.z))/den;
   a4=((Ax-abs(xcp.x))*abs(xcp.y)*(Az-abs(xcp.z)))/den;
   a5=((Ax-abs(xcp.x))*abs(xcp.y)*abs(xcp.z))/den;
   a6=((Ax-abs(xcp.x))*(Ay-abs(xcp.y))*abs(xcp.z))/den;
   a7=(abs(xcp.x)*(Ay-abs(xcp.y))*abs(xcp.z))/den;
   fp.x=fg[0].x*a0+fg[1].x*a1+fg[2].x*a2+fg[3].x*a3+fg[4].x*a4+fg[5].x*a5+fg[6].x*a6+fg[7].x*a7;
   fp.y=fg[0].y*a0+fg[1].y*a1+fg[2].y*a2+fg[3].y*a3+fg[4].y*a4+fg[5].y*a5+fg[6].y*a6+fg[7].y*a7;
   fp.z=fg[0].z*a0+fg[1].z*a1+fg[2].z*a2+fg[3].z*a3+fg[4].z*a4+fg[5].z*a5+fg[6].z*a6+fg[7].z*a7;
}

//Electric force at the 8 corners of every cell, due to the corner charges of the
//neighbouring cells within +-lc. fgc_D holds 8 float3 per cell. Grid units:
//  F_corner = sum_m q_m (x_corner - x_m)/(4*pi*|x_corner - x_m|^3).
__global__ void PGFieldF_ker(const float4* __restrict__ rg0_D, const float4* __restrict__ rg1_D, float3* __restrict__ fgc_D, int Nc, int3 N, float3 A, float3 B, int lc)
{
   const int Ng=8;
   int Ncx=N.x-1, Ncy=N.y-1, Ncz=N.z-1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Nc)
   {
      int i_c=idx/(Ncy*Ncz);
      int j_c=(idx-i_c*(Ncy*Ncz))/Ncz;
      int k_c=(idx-i_c*(Ncy*Ncz)-j_c*Ncz);
      float3 xg[Ng]; int3 ig[Ng];
      CellVertex_ix(ig, xg, idx, N, A, B);
      float3 fg[Ng];
      #pragma unroll
      for(int n=0;n<Ng;n++) fg[n]=make_float3(0.f,0.f,0.f);
      for(int i=-lc;i<=lc;i++)
       for(int j=-lc;j<=lc;j++)
        for(int k=-lc;k<=lc;k++)
        {
           int i1=i_c+i, j1=j_c+j, k1=k_c+k;
           if(i1<0||i1>=Ncx||j1<0||j1>=Ncy||k1<0||k1>=Ncz) continue;
           int icc=i1*Ncy*Ncz+j1*Ncz+k1;
           float3 xg1[Ng]; int3 ig1[Ng];
           CellVertex_ix(ig1, xg1, icc, N, A, B);
           float4 r0=rg0_D[icc], r1=rg1_D[icc];
           float rgm[Ng]={r0.x,r0.y,r0.z,r0.w,r1.x,r1.y,r1.z,r1.w};
           #pragma unroll
           for(int n=0;n<Ng;n++)
              #pragma unroll
              for(int m=0;m<Ng;m++)
              {
                 float dx=xg[n].x-xg1[m].x, dy=xg[n].y-xg1[m].y, dz=xg[n].z-xg1[m].z;
                 float r2=dx*dx+dy*dy+dz*dz;
                 if(r2>0.f)
                 {
                    float inv=rgm[m]/(4.f*MPI*r2*sqrtf(r2));
                    fg[n].x+=dx*inv; fg[n].y+=dy*inv; fg[n].z+=dz*inv;
                 }
              }
        }
      #pragma unroll
      for(int n=0;n<Ng;n++) fgc_D[idx*Ng+n]=fg[n];
   }
}

//Interpolate the cell corner forces to the particles and add to Ep.
__global__ void CellFieldToParticleF_ker(const float4* __restrict__ xp_D, float3* __restrict__ Ep_D, const float3* __restrict__ fgc_D, const int* __restrict__ cellind_d, int Np, int3 N, float3 A, float3 B)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      int ic=cellind_d[idx];
      float3 fg[8];
      #pragma unroll
      for(int n=0;n<8;n++) fg[n]=fgc_D[ic*8+n];
      float3 fp; float4 xp=xp_D[idx];
      CellToParticleF(xp, fp, fg, ic, N, A, B);
      Ep_D[idx].x+=fp.x; Ep_D[idx].y+=fp.y; Ep_D[idx].z+=fp.z;
   }
}

//Host orchestration of the cell-multipole force, adding it (grid units) to Ep_D.
void MPFieldF(float4 *xp_D, float3 *Ep_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc)
{
   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   int threads=N_THREADS, blocks=Np/threads+1, blocksC=Nc/threads+1;
   int *offset_h, *cellind_h, *Nc_h, *offset_d, *cellind_d;
   cudaMallocHost(&offset_h, Nc*sizeof(int));
   cudaMallocHost(&cellind_h, Np*sizeof(int));
   cudaMallocHost(&Nc_h, Nc*sizeof(int));
   cudaMalloc(&offset_d, Nc*sizeof(int));
   cudaMalloc(&cellind_d, Np*sizeof(int));
   cudaMemcpy(Nc_h, Nc_D, Nc*sizeof(int), cudaMemcpyDeviceToHost);
   //cudaDeviceSynchronize();
   FillCell(offset_h, cellind_h, Nc_h, Nc, Np);
   cudaMemcpy(offset_d, offset_h, Nc*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cellind_d, cellind_h, Np*sizeof(int), cudaMemcpyHostToDevice);

   float4 *rg0_D, *rg1_D;
   cudaMalloc(&rg0_D, Nc*sizeof(float4));
   cudaMalloc(&rg1_D, Nc*sizeof(float4));
   cudaMemset(rg0_D, 0, Nc*sizeof(float4));
   cudaMemset(rg1_D, 0, Nc*sizeof(float4));
   float3 *fgc_D;
   cudaMalloc(&fgc_D, (size_t)Nc*8*sizeof(float3));

   CellDensity_ker<<<blocksC,threads>>>(xp_D, rg0_D, rg1_D, Nc_D, offset_d, Nc, N, A, B);
   PGFieldF_ker<<<blocksC,threads>>>(rg0_D, rg1_D, fgc_D, Nc, N, A, B, lc);
   CellFieldToParticleF_ker<<<blocks,threads>>>(xp_D, Ep_D, fgc_D, cellind_d, Np, N, A, B);
   //cudaDeviceSynchronize();

   cudaFree(rg0_D); cudaFree(rg1_D); cudaFree(fgc_D);
   cudaFree(offset_d); cudaFree(cellind_d);
   cudaFreeHost(offset_h); cudaFreeHost(cellind_h); cudaFreeHost(Nc_h);
}

//============ GPU cell-multipole for the retarded terms (F2 / F3) ============//
//The internal magnetic field (F3) and the inductive electric field (F2) are the
//velocity/acceleration-dependent sums computed all-pairs by BIntField/EIndField.
//Here they are accelerated with the same cell-multipole idea as the electrostatic
//force: each cell's current moments are condensed onto its 8 corners and the
//corner-to-corner fields are computed per cell pair. The required moments are
//  V_m = sum_j q_j v_j        (for B = q (r x V)/r^3)
//  A_m = sum_j q_j a_j  and   T_m = sum_j q_j v_j (x) v_j   (for E2 = q (A/r + T.r/r^3))
//all distributed to the corners with the CIC weights. q = -mu0|e|/(4 pi).

//CIC weights of a particle to its cell's 8 corners (corner order matches
//CellVertex_ix / ParticleToCell).
__device__ void cellWeights(float4 x, int i_cell, int3 N, float3 A, float3 B, float a[8])
{
   int Nx=N.x, Ny=N.y, Nz=N.z;
   float Ax=A.x, Ay=A.y, Az=A.z;
   float Bx=B.x, By=B.y, Bz=B.z;
   int Nyc=N.y-1, Nzc=N.z-1;
   float den=Ax*Ay*Az;
   int3 ic;
   ic.x=i_cell/(Nyc*Nzc);
   ic.y=(i_cell-ic.x*Nyc*Nzc)/Nzc;
   ic.z=i_cell-ic.x*Nyc*Nzc-ic.y*Nzc;
   float ax=fabsf(x.x-xI(ic.x,Ax,Bx,Nx));
   float ay=fabsf(x.y-xI(ic.y,Ay,By,Ny));
   float az=fabsf(x.z-xI(ic.z,Az,Bz,Nz));
   a[0]=(Ax-ax)*(Ay-ay)*(Az-az)/den; a[1]=ax*(Ay-ay)*(Az-az)/den;
   a[2]=ax*ay*(Az-az)/den;           a[3]=ax*ay*az/den;
   a[4]=(Ax-ax)*ay*(Az-az)/den;      a[5]=(Ax-ax)*ay*az/den;
   a[6]=(Ax-ax)*(Ay-ay)*az/den;      a[7]=ax*(Ay-ay)*az/den;
}

//Condense the per-cell current moments onto the 8 corners. Tgc0=(Txx,Txy,Txz),
//Tgc1=(Tyy,Tyz,Tzz) store the symmetric velocity tensor.
__global__ void CellMomentsRet_ker(const float4* __restrict__ xp, const float3* __restrict__ vp, const float3* __restrict__ ap, const int* __restrict__ Nc_D, const int* __restrict__ offset_d, int Nc, int3 N, float3 A, float3 B, float3* __restrict__ Vgc, float3* __restrict__ Agc, float3* __restrict__ Tgc0, float3* __restrict__ Tgc1)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Nc)
   {
      int Ncell=Nc_D[idx], offset=offset_d[idx];
      float3 Vg[8], Ag[8], T0[8], T1[8];
      #pragma unroll
      for(int k=0;k<8;k++){ Vg[k]=make_float3(0,0,0); Ag[k]=make_float3(0,0,0); T0[k]=make_float3(0,0,0); T1[k]=make_float3(0,0,0); }
      for(int p=offset;p<offset+Ncell;p++)
      {
         float4 x=xp[p]; float3 v=vp[p], a=ap[p]; float Z=x.w;
         float w[8]; cellWeights(x, idx, N, A, B, w);
         #pragma unroll
         for(int k=0;k<8;k++)
         {
            float zw=Z*w[k];
            Vg[k].x+=zw*v.x; Vg[k].y+=zw*v.y; Vg[k].z+=zw*v.z;
            Ag[k].x+=zw*a.x; Ag[k].y+=zw*a.y; Ag[k].z+=zw*a.z;
            T0[k].x+=zw*v.x*v.x; T0[k].y+=zw*v.x*v.y; T0[k].z+=zw*v.x*v.z;
            T1[k].x+=zw*v.y*v.y; T1[k].y+=zw*v.y*v.z; T1[k].z+=zw*v.z*v.z;
         }
      }
      #pragma unroll
      for(int k=0;k<8;k++){ Vgc[idx*8+k]=Vg[k]; Agc[idx*8+k]=Ag[k]; Tgc0[idx*8+k]=T0[k]; Tgc1[idx*8+k]=T1[k]; }
   }
}

//Corner B (Bgc) and corner E2 (Egc) from the neighbouring cells' corner moments.
__global__ void PGRetField_ker(const float3* __restrict__ Vgc, const float3* __restrict__ Agc, const float3* __restrict__ Tgc0, const float3* __restrict__ Tgc1, float3* __restrict__ Bgc, float3* __restrict__ Egc, int Nc, int3 N, float3 A, float3 B, int lc, float qcoef)
{
   const int Ng=8;
   int Ncx=N.x-1, Ncy=N.y-1, Ncz=N.z-1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Nc)
   {
      int i_c=idx/(Ncy*Ncz);
      int j_c=(idx-i_c*(Ncy*Ncz))/Ncz;
      int k_c=(idx-i_c*(Ncy*Ncz)-j_c*Ncz);
      float3 xg[Ng]; int3 ig[Ng];
      CellVertex_ix(ig, xg, idx, N, A, B);
      float3 Bn[Ng], En[Ng];
      #pragma unroll
      for(int n=0;n<Ng;n++){ Bn[n]=make_float3(0,0,0); En[n]=make_float3(0,0,0); }
      for(int i=-lc;i<=lc;i++)
       for(int j=-lc;j<=lc;j++)
        for(int k=-lc;k<=lc;k++)
        {
           int i1=i_c+i, j1=j_c+j, k1=k_c+k;
           if(i1<0||i1>=Ncx||j1<0||j1>=Ncy||k1<0||k1>=Ncz) continue;
           int icc=i1*Ncy*Ncz+j1*Ncz+k1;
           float3 xg1[Ng]; int3 ig1[Ng];
           CellVertex_ix(ig1, xg1, icc, N, A, B);
           #pragma unroll
           for(int n=0;n<Ng;n++)
              #pragma unroll
              for(int m=0;m<Ng;m++)
              {
                 float rx=xg[n].x-xg1[m].x, ry=xg[n].y-xg1[m].y, rz=xg[n].z-xg1[m].z;
                 float r2=rx*rx+ry*ry+rz*rz;
                 if(r2>0.f)
                 {
                    float rinv=rsqrtf(r2), r3inv=rinv/r2;
                    float3 Vm=Vgc[icc*8+m], Am=Agc[icc*8+m], T0=Tgc0[icc*8+m], T1=Tgc1[icc*8+m];
                    //B += qcoef (r x V)/r^3
                    Bn[n].x+=qcoef*(ry*Vm.z-rz*Vm.y)*r3inv;
                    Bn[n].y+=qcoef*(rz*Vm.x-rx*Vm.z)*r3inv;
                    Bn[n].z+=qcoef*(rx*Vm.y-ry*Vm.x)*r3inv;
                    //E += qcoef (A/r + (T.r)/r^3)
                    float Trx=T0.x*rx+T0.y*ry+T0.z*rz;
                    float Try=T0.y*rx+T1.x*ry+T1.y*rz;
                    float Trz=T0.z*rx+T1.y*ry+T1.z*rz;
                    En[n].x+=qcoef*(Am.x*rinv+Trx*r3inv);
                    En[n].y+=qcoef*(Am.y*rinv+Try*r3inv);
                    En[n].z+=qcoef*(Am.z*rinv+Trz*r3inv);
                 }
              }
        }
      #pragma unroll
      for(int n=0;n<Ng;n++){ Bgc[idx*8+n]=Bn[n]; Egc[idx*8+n]=En[n]; }
   }
}

//Interpolate the corner B / E2 to the particles, adding to Bp (physical B) and
//Eind (physical E). do_b / do_e select which fields are written.
__global__ void CellRetToParticle_ker(const float4* __restrict__ xp, float3* __restrict__ Bp, float3* __restrict__ Eind, const float3* __restrict__ Bgc, const float3* __restrict__ Egc, const int* __restrict__ cellind_d, int Np, int3 N, float3 A, float3 B, int do_b, int do_e)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      int ic=cellind_d[idx];
      float4 xp4=xp[idx]; float3 fp;
      if(do_b)
      {
         float3 fg[8];
         #pragma unroll
         for(int n=0;n<8;n++) fg[n]=Bgc[ic*8+n];
         CellToParticleF(xp4, fp, fg, ic, N, A, B);
         Bp[idx].x+=fp.x; Bp[idx].y+=fp.y; Bp[idx].z+=fp.z;
      }
      if(do_e)
      {
         float3 fg[8];
         #pragma unroll
         for(int n=0;n<8;n++) fg[n]=Egc[ic*8+n];
         CellToParticleF(xp4, fp, fg, ic, N, A, B);
         Eind[idx].x+=fp.x; Eind[idx].y+=fp.y; Eind[idx].z+=fp.z;
      }
   }
}

//Host orchestration: cell-multipole F3 (Bp_D) and/or F2 (Eind_D). Pass NULL to
//skip a field. lc is the cell range.
void RetardedMultipole(float4 *xp_D, float3 *vp_D, float3 *ap_D, float3 *Eind_D, float3 *Bp_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc)
{
   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   int threads=N_THREADS, blocks=Np/threads+1, blocksC=Nc/threads+1;
   int *offset_h, *cellind_h, *Nc_h, *offset_d, *cellind_d;
   cudaMallocHost(&offset_h, Nc*sizeof(int));
   cudaMallocHost(&cellind_h, Np*sizeof(int));
   cudaMallocHost(&Nc_h, Nc*sizeof(int));
   cudaMalloc(&offset_d, Nc*sizeof(int));
   cudaMalloc(&cellind_d, Np*sizeof(int));
   cudaMemcpy(Nc_h, Nc_D, Nc*sizeof(int), cudaMemcpyDeviceToHost);
   //cudaDeviceSynchronize();
   FillCell(offset_h, cellind_h, Nc_h, Nc, Np);
   cudaMemcpy(offset_d, offset_h, Nc*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cellind_d, cellind_h, Np*sizeof(int), cudaMemcpyHostToDevice);

   float3 *Vgc,*Agc,*Tgc0,*Tgc1,*Bgc,*Egc;
   size_t nc8=(size_t)Nc*8*sizeof(float3);
   cudaMalloc(&Vgc,nc8); cudaMalloc(&Agc,nc8); cudaMalloc(&Tgc0,nc8); cudaMalloc(&Tgc1,nc8);
   cudaMalloc(&Bgc,nc8); cudaMalloc(&Egc,nc8);

   CellMomentsRet_ker<<<blocksC,threads>>>(xp_D, vp_D, ap_D, Nc_D, offset_d, Nc, N, A, B, Vgc, Agc, Tgc0, Tgc1);
   float qcoef=-(float)(M0*ECHG/(4*MPI));
   PGRetField_ker<<<blocksC,threads>>>(Vgc, Agc, Tgc0, Tgc1, Bgc, Egc, Nc, N, A, B, lc, qcoef);
   CellRetToParticle_ker<<<blocks,threads>>>(xp_D, Bp_D, Eind_D, Bgc, Egc, cellind_d, Np, N, A, B, Bp_D!=NULL, Eind_D!=NULL);
   //cudaDeviceSynchronize();

   cudaFree(Vgc); cudaFree(Agc); cudaFree(Tgc0); cudaFree(Tgc1); cudaFree(Bgc); cudaFree(Egc);
   cudaFree(offset_d); cudaFree(cellind_d);
   cudaFreeHost(offset_h); cudaFreeHost(cellind_h); cudaFreeHost(Nc_h);
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
   int threads=N_THREADS;
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

   int threads=N_THREADS;
   int blocks=Nc/threads+1;
   int* Nc_D = raw_pointer_cast(&Nc_d[0]);
   threshold<<<blocks,threads>>>(Nc_D, cellind_D, Nc, thresh);
   //cudaDeviceSynchronize();
   
   thrust::device_vector<int> cellclust_d(cellind_d);
   int* cellclust_D = raw_pointer_cast(&cellclust_d[0]);
   thrust::device_vector<int>::iterator iter;
   iter = remove_if(thrust::device, cellind_d.begin(), cellind_d.end(), isMinus());
   //cudaDeviceSynchronize();
   cellind_d.resize(iter - cellind_d.begin());
   //cudaDeviceSynchronize();


//         auto start0 = std::chrono::high_resolution_clock::now(); 
   int Ncell=cellind_d.size();
//   threads=512;
//   blocks=Ncell/threads+1;
//   cluster<<<blocks,threads,sizeof(bool)>>>(cellclust_D, cellind_D, Ncell, N);
//   //cudaDeviceSynchronize();

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
   ofile.open("./grid.txt", ios::out | ios::binary | ios::app);
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

//================ float3 (vector force) helpers for the recursion ===========//
//These mirror FillCluster / ClusterToXp / OrderCellToFp but operate on the
//float3 electric force gathered at the particles, so the multiscale refinement
//corrects the FORCE rather than the (scalar) potential.

struct float3_plus
{
   __host__ __device__ float3 operator()(const float3 &a, const float3 &b) const
   { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
};

void FillClusterF(thrust::device_vector<float4>& xpc_d, thrust::device_vector<float4>& xp_d, thrust::device_vector<float3>& Epc_d, thrust::device_vector<float3>& Ep_d, thrust::host_vector<int>& Nc_h, thrust::host_vector<int>& offset_h, thrust::host_vector<int>& cellind_h, int celloff, int Ncell)
{
   int off=0;
   for(int i=celloff;i<celloff+Ncell;i++)
   {
      int icell = cellind_h[i];
      int start=offset_h[icell];
      int ende=start+Nc_h[icell];
      copy(thrust::device, xp_d.begin()+start, xp_d.begin()+ende, xpc_d.begin()+off);
      copy(thrust::device, Ep_d.begin()+start, Ep_d.begin()+ende, Epc_d.begin()+off);
      off+=ende-start;
   }
}

void ClusterToXpF(thrust::device_vector<float3>& Epc_d, thrust::device_vector<float3>& Ep_d, thrust::host_vector<int>& Nc_h, thrust::host_vector<int>& offset_h, thrust::host_vector<int>& cellind_h, int celloff, int Ncell)
{
   int off=0;
   for(int i=celloff;i<celloff+Ncell;i++)
   {
      int icell = cellind_h[i];
      int start=offset_h[icell];
      int ende=Nc_h[icell];
      copy(thrust::device, Epc_d.begin()+off, Epc_d.begin()+off+ende, Ep_d.begin()+start);
      off+=ende;
   }
}

//Ep += gather(XC, Epc): map the cluster-ordered force back to the input order.
void OrderCellToEp(thrust::device_vector<float3>& Ep_d, thrust::device_vector<float3>& Epc_d, thrust::device_vector<int>& XC_d)
{
   int Np=Ep_d.size();
   thrust::device_vector<float3> Epp_d(Np);
   thrust::gather(XC_d.begin(), XC_d.end(), Epc_d.begin(), Epp_d.begin());
   thrust::transform(Ep_d.begin(), Ep_d.end(), Epp_d.begin(), Ep_d.begin(), float3_plus());
}

//==================== Step C: PP short-range electric force ==================//
//Direct particle-particle electric field from particles in the (2*lc+1)^3 block
//of cells around each particle, accumulated (in grid units) into Ep. This is the
//short-range P^3M correction; it reuses the cell ordering already built for the
//mesh. The summed field is  E = sum_j q_j (xi-xj)/(4*pi*r^3).
__global__ void PPFieldF_ker(const float4* __restrict__ xp_D, float3* __restrict__ Ep_D, const int* __restrict__ Nc_D, const int* __restrict__ cellind_d, const int* __restrict__ offset_d, int Np, int3 N, int lc)
{
   int Ncx=N.x-1, Ncy=N.y-1, Ncz=N.z-1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < Np)
   {
      float4 x=xp_D[idx];
      float ex=0.f, ey=0.f, ez=0.f;
      int ic=cellind_d[idx];
      int i_c=ic/(Ncy*Ncz);
      int j_c=(ic-i_c*(Ncy*Ncz))/Ncz;
      int k_c=(ic-i_c*(Ncy*Ncz)-j_c*Ncz);
      for(int i=-lc;i<=lc;i++)
       for(int j=-lc;j<=lc;j++)
        for(int k=-lc;k<=lc;k++)
        {
           int i1=i_c+i, j1=j_c+j, k1=k_c+k;
           if(i1<0||i1>=Ncx||j1<0||j1>=Ncy||k1<0||k1>=Ncz) continue;
           int icc=i1*Ncy*Ncz+j1*Ncz+k1;
           int Ncell=Nc_D[icc];
           int offset=offset_d[icc];
           for(int n=offset;n<offset+Ncell;n++)
           {
              float4 xp=xp_D[n];
              float dx=x.x-xp.x, dy=x.y-xp.y, dz=x.z-xp.z;
              float r2=dx*dx+dy*dy+dz*dz;
              if(r2>0.f)
              {
                 float den=4.f*MPI*r2*sqrtf(r2);
                 float inv=xp.w/den;
                 ex+=dx*inv; ey+=dy*inv; ez+=dz*inv;
              }
           }
        }
      Ep_D[idx].x+=ex; Ep_D[idx].y+=ey; Ep_D[idx].z+=ez;
   }
}

void PPFieldF(float4 *xp_D, float3 *Ep_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc)
{
   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   int threads=N_THREADS, blocks=Np/threads+1;
   int *offset_h, *cellind_h, *Nc_h, *offset_d, *cellind_d;
   cudaMallocHost(&offset_h, Nc*sizeof(int));
   cudaMallocHost(&cellind_h, Np*sizeof(int));
   cudaMallocHost(&Nc_h, Nc*sizeof(int));
   cudaMalloc(&offset_d, Nc*sizeof(int));
   cudaMalloc(&cellind_d, Np*sizeof(int));
   cudaMemcpy(Nc_h, Nc_D, Nc*sizeof(int), cudaMemcpyDeviceToHost);
   //cudaDeviceSynchronize();
   FillCell(offset_h, cellind_h, Nc_h, Nc, Np);
   cudaMemcpy(offset_d, offset_h, Nc*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cellind_d, cellind_h, Np*sizeof(int), cudaMemcpyHostToDevice);
   PPFieldF_ker<<<blocks,threads>>>(xp_D, Ep_D, Nc_D, cellind_d, offset_d, Np, N, lc);
   gpuErrchk(cudaDeviceSynchronize());
   cudaFreeHost(offset_h); cudaFreeHost(cellind_h); cudaFreeHost(Nc_h);
   cudaFree(offset_d); cudaFree(cellind_d);
}

//==================== Step C: internal magnetic field =======================//
//Magnetic field generated at every particle by the motion of all the others
//(plasma_pp.cpp B_int), computed as a direct O(N^2) sum. Produces a PHYSICAL
//field (Tesla) added directly to B in the Boris push.
__global__ void BInt_ker(const float4* __restrict__ xp, const float3* __restrict__ vp, float3* __restrict__ Bp, int Np, float q)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      float4 xi=xp[idx];
      float bx=0.f, by=0.f, bz=0.f;
      for(int j=0;j<Np;j++)
      {
         float4 xj=xp[j];
         float3 vj=vp[j];
         float dx=xi.x-xj.x, dy=xi.y-xj.y, dz=xi.z-xj.z;
         float r2=dx*dx+dy*dy+dz*dz;
         if(r2>0.f)
         {
            float r3=r2*sqrtf(r2);
            float sj=xj.w/r3;                  //charge sign of source / r^3
            bx+=sj*(dy*vj.z-dz*vj.y);
            by+=sj*(dz*vj.x-dx*vj.z);
            bz+=sj*(dx*vj.y-dy*vj.x);
         }
      }
      Bp[idx].x+=q*bx; Bp[idx].y+=q*by; Bp[idx].z+=q*bz;
   }
}

void BIntField(float4 *xp_D, float3 *vp_D, float3 *Bp_D, int Np)
{
   int threads=N_THREADS, blocks=Np/threads+1;
   //q_j = w_j*|e|, so the coefficient carries |e| (=ECHG); w_j is read from xp.w
   float q=-(float)(M0*ECHG/(4*MPI));
   BInt_ker<<<blocks,threads>>>(xp_D, vp_D, Bp_D, Np, q);
}

//=========== Retarded internal electric field (F2 term of the notes) ========//
//The non-electrostatic part of the internal electric field, E = -dA/dt, from the
//motion of the other charges (plasma_equations.pdf eq.22, F2):
//   E2_i = q1 * sum_{j!=i} sign_j ( a_j |r|^2 + v_j (r . v_j) ) / |r|^3 ,
// with r = x_i - x_j and q1 = -mu0*qe/(4*pi). This is a PHYSICAL electric field
// (the electrostatic F1 part is supplied by the mesh) added directly to E in the
// Boris push. Direct O(N^2) sum, like the internal magnetic field.
__global__ void EInd_ker(const float4* __restrict__ xp, const float3* __restrict__ vp, const float3* __restrict__ ap, float3* __restrict__ Ep, int Np, float q1)
{
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<Np)
   {
      float4 xi=xp[idx];
      float ex=0.f, ey=0.f, ez=0.f;
      for(int j=0;j<Np;j++)
      {
         float4 xj=xp[j];
         float3 vj=vp[j], aj=ap[j];
         float dx=xi.x-xj.x, dy=xi.y-xj.y, dz=xi.z-xj.z;
         float r2=dx*dx+dy*dy+dz*dz;
         if(r2>0.f)
         {
            float r3=r2*sqrtf(r2);
            float prod=dx*vj.x+dy*vj.y+dz*vj.z;   //(x_i-x_j) . v_j
            float sj=xj.w/r3;                      //source charge sign / r^3
            ex+=sj*(aj.x*r2+vj.x*prod);
            ey+=sj*(aj.y*r2+vj.y*prod);
            ez+=sj*(aj.z*r2+vj.z*prod);
         }
      }
      Ep[idx].x+=q1*ex; Ep[idx].y+=q1*ey; Ep[idx].z+=q1*ez;
   }
}

void EIndField(float4 *xp_D, float3 *vp_D, float3 *ap_D, float3 *Eind_D, int Np)
{
   int threads=N_THREADS, blocks=Np/threads+1;
   //q_j = w_j*|e|, so the coefficient carries |e| (=ECHG); w_j is read from xp.w
   float q1=-(float)(M0*ECHG/(4*MPI));
   EInd_ker<<<blocks,threads>>>(xp_D, vp_D, ap_D, Eind_D, Np, q1);
}

//================= Multiscale recursion on the vector force ==================//
void RecursiveMeshField(thrust::device_vector<float4>& xp_d, thrust::device_vector<float3>& Ep_d, thrust::device_vector<int>& Nc_d, float3 A, float3 B, int3 N, int Nc, int fgrid, int depth, int depthmax)
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
      for(int ii=celloff;ii<celloff+Ncell;ii++)
      {
         int icell = cellind_h[ii];
         Npc+=Nc_h[icell];
      }
      thrust::device_vector<float4> xpc_d(Npc); //particle positions
      thrust::device_vector<float3> Epc_d(Npc, make_float3(0,0,0)); //cluster force
      //cast thrust vectors to pointers to be processed by kernels
      float4* xpc_D = raw_pointer_cast(&xpc_d[0]);
      //fill xpc with cluster particles (positions + current force)
      FillClusterF(xpc_d, xp_d, Epc_d, Ep_d, Nc_h, offset_h, cellind_h, celloff, Ncell);
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
         //Subtract the coarse-grid force (it will be replaced by the fine grid)
         MeshFieldToParticle(xpc_D, raw_pointer_cast(&Epc_d[0]), cfg_D, Npc, cN, cA, cB, -1);
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
         //Add the fine-grid force (gathered in the fine-cell order), then recurse
         thrust::device_vector<float3> Epco_d(Npc, make_float3(0,0,0));
         float3* Epco_D = raw_pointer_cast(&Epco_d[0]);
         MeshFieldToParticle(xpc_D, Epco_D, cfg_D, Npc, fN, fA, fB, 1);
         RecursiveMeshField(xpc_d, Epco_d, cNc_d, fA, fB, fN, cNc, fgrid, depth, depthmax);
         OrderCellToEp(Epc_d, Epco_d, XCc_d);
      }
      //Copy cluster force back to the global force array
      ClusterToXpF(Epc_d, Ep_d, Nc_h, offset_h, cellind_h, celloff, Ncell);
   }
}


