//Plasma simulation with Boris integration method 

//Compile: g++ plasma.cpp -o app
//Compile with openmap parallel: g++ plasma.cpp -o app -fopenmp
//Compile with openmap parallel and gsl: g++ plasma.cpp -o app -fopenmp -lgsl -lgslcblas
//Compile with openacc cuda: g++ plasma.cpp -o app -fopenacc
//Compile with openacc cuda (pgi compiler): /opt/pgi/linux86-64/17.4/bin/pgc++ plasma.cpp -o app -acc -ta=nvidia:maxwell -fast
//compile with openacc (verbose): /opt/pgi/linux86-64/17.4/bin/pgc++ plasma.cpp -o app -acc -ta=nvidia:maxwell,time -Minfo=accel -fast
//compile with openacc unified memory: /opt/pgi/linux86-64/17.4/bin/pgc++ plasma.cpp -o app -acc -ta=tesla:managed -fast


//#include <gsl/gsl_sf_hyperg.h>
//#include <omp.h>
#include <math.h>
#include <iostream>
#include <iomanip>
//#include <string>
#include <fstream>
#include <vector>
#include <ctime> 
#include <stdlib.h> 
#include <cstdlib>
//#include <algorithm>
//#include <functional>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <cufft.h>

#include "Poisson.h"

using namespace std;
//using namespace thrust;


//Main parameters
#define Np 100000 //number of particles
#define Nel 0 //number of electrons
#define Nstep 1000 //number of steps
#define Nsave 100 //number of trajectory points saved
#define dt (5*pow(10,-10)) //time step s

//Physical Constants
#define MPI 3.1415926535897932385 //greek pi
#define e0 (8.854187817*pow(10,-12))  //electric suscettivity C^2/(N*m^2)
#define m0 (4*MPI*pow(10,-7)) //magnetic permeability N/A^2
#define qe (-1.60217733*pow(10,-19)*10000) //electron charge C
#define me (9.1093897*pow(10,-31)*10000)   //electrom mass Kg
#define mp (2*1.67265*pow(10,-27)*10000)   //deuteron mass Kg

//System parameters
#define ar 0.05 //m charged-magnetic ring radius
#define dr 0.01 //m distance of the rings from center
#define ar2b (pow(ar-0.2*ar,2))
#define ar2t (pow(ar+0.2*ar,2))
#define drb (dr-0.2*dr)
#define drt (dr+0.2*dr)








//MAIN 
int main(int argc, char *argv[])
{

   //plasma particles host vectors
   thrust::host_vector<float4> xp_h(Np); //position & charge vector
   thrust::host_vector<float3> vp_h(Np); //velocity vector
   thrust::host_vector<float3> ap_h(Np); //acceleration vector
   //vector<float> traj(Nsave*Np*3); //vector of trajectories
   vector<float> lost(Np*3); //vector of lost particles

   //initialize random seed
//   srand((unsigned)time(0));
   srand (5);

   //initialize particles
   init_p(xp_h,vp_h,ap_h);

   //Device particles thrust vectors
   thrust::device_vector<float4> xp_d(xp_h); //particle positions
   //thrust::device_vector<float3> vp_d(vp_h), ap_d(ap_h);
   //clear host vectors
   xp_h.clear(); xp_h.shrink_to_fit();
   vp_h.clear(); vp_h.shrink_to_fit();
   ap_h.clear(); ap_h.shrink_to_fit();
   //particle-cell indices vectors
   thrust::device_vector<int> XC_d(Np); //cell indeces of particles
   thrust::device_vector<int> CX_d(Np); //particles indeces for cell
   //field at particle vectors
   thrust::device_vector<float> fp_d(Np, 0); //particle field

   //cast thrust vectors to pointers to be processed by kernels
   float4* xp_D = raw_pointer_cast(&xp_d[0]);
//   float4* xpc_D = raw_pointer_cast(&xpc_d[0]);
   float* fp_D = raw_pointer_cast(&fp_d[0]);
//   float* fpc_D = raw_pointer_cast(&fpc_d[0]);
   //float3* vp_D = raw_pointer_cast(&vp_d[0]);
   //float3* ap_D = raw_pointer_cast(&ap_d[0]);
   int* XC_D = raw_pointer_cast(&XC_d[0]);
   int* CX_D = raw_pointer_cast(&CX_d[0]);

   time_t start,end;
   time (&start);

   //int save_step=Nstep/Nsave;

   //number of lost particles
   int n_lost=0;

   //initialize hypergeometric function
   //hyper_func_calc(LH1,LH2,Np);

   float3 A, B;
   int3 N;

   float fgrid=1; //Ngrid=Np/fgrid


   for(int k=0;k<Nstep;k++)
   {
      //clear field vector fp
      fill(thrust::device, fp_d.begin(), fp_d.end(), 0);
      //find mesh tree structure 
      init_mesh_struct(xp_d, Np, A, B, N, fgrid);
      //define mesh depending vectors
      int Ntot2=2*N.x*2*N.y*(2*N.z/2+1);
      int Nc=(N.x-1)*(N.y-1)*(N.z-1);
      thrust::device_vector<float> rho_d(N.x*N.y*N.z, 0), fg_d(N.x*N.y*N.z, 0);
      thrust::device_vector<int> Nc_d(Nc, 0);
      float* rho_D = raw_pointer_cast(&rho_d[0]);
      float* fg_D = raw_pointer_cast(&fg_d[0]);
      int* Nc_D = raw_pointer_cast(&Nc_d[0]);

//print memory usage
//cudaDeviceSynchronize();
//size_t free_byte ;
//size_t total_byte ;
//cudaMemGetInfo( &free_byte, &total_byte ) ;
//double free_db = (double)free_byte ;
//double total_db = (double)total_byte ;
//double used_db = total_db - free_db ;
//printf("GPU memory: used = %f, free = %f MB, total = %f MB\n",
//            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);


      //Assign particles to mesh density
      ParticleToMesh(xp_D, rho_D, XC_D, Np, N, A, B);
//float* rho_H;
//cudaMallocHost(&rho_H, N.x*N.y*N.z*sizeof(float));
//cudaMemcpy(rho_H, rho_D, N.x*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost);
//thrust::host_vector<float> frho_h(rho_d);
//for(int i=0;i<frho_h.size();i++)
//{
//   //if(fg_h[i]==0 || fg_h[i]>Np) 
//   printf("%f, ", rho_H[i]);
//}
//cudaFreeHost(rho_H);

      if(true)
      {
         OrderCell(XC_d, CX_d, Nc_d, Np, N);
         thrust::device_vector<float4> xpc_d(Np); //particle positions ordered by cell
         OrderXpToCell(xp_d, xpc_d, CX_d);
         xp_d=xpc_d;
      }

   if(argc<3)
   {
      //Poisson solver in fourier space
      if(true)
      {
         thrust::device_vector<thrust::complex<float> > frho_d(Ntot2), fker_d(Ntot2);
         cufftComplex* frho_D = (cufftComplex*)raw_pointer_cast(&frho_d[0]);
         cufftComplex* fker_D = (cufftComplex*)raw_pointer_cast(&fker_d[0]);
         fft(rho_D, frho_D, fker_D, A, B, N);
         Poisson_Solver(frho_D, fker_D, N);
         //Poisson_Solver_thrust(frho_d, fker_d);
         fft_inv(frho_D, fg_D, N);

         //Assign mesh field to particles
         //MeshToParticle(xp_D, fp_D, rho_D, Np, N, A, B, 1);
         MeshToParticle(xp_D, fp_D, fg_D, Np, N, A, B, 1);
      }

      bool multgrid=true;
      if(multgrid)
      {
         int depth=0, depthmax=2;
         cudaDeviceSynchronize();
         RecursiveMeshField(xp_d, fp_d, Nc_d, A, B, N, Nc, fgrid, depth, depthmax);
      }

      if(argc>1)
      {
         int lc=atoi(argv[1]);
         //printf("PP interaction\n");
//         OrderCell(XC_d, CX_d, Nc_d, Np, N);
//         OrderXpToCell(xp_d, xpc_d, CX_d);
//         PPField(xpc_D, fpc_D, Nc_D, Np, N, A, B, lc);
         PPField(xp_D, fp_D, Nc_D, Np, N, A, B, lc);
         //PPField_Tot(xp_D, fp_D, Np);
//         OrderCellToFp(fp_d, fpc_d, XC_d);
      }

      if(k==Nstep-1) 
      {
          //save_traj(xp_d, fp_d, fg_d, Nc_d, A, B, N);
          save_traj_tot(xp_d, fp_d, A, B, N);
      }

   }
   else 
   {   
      PPField_Tot(xp_D, fp_D, Np);
      if(k==Nstep-1) save_traj_tot(xp_d, fp_d, A, B, N);
   }

      

      //loop over mesh tree levels
//      for(int l=0;l<Nm,l++)
//      {
//         mesh_h=Smesh_h[l];
//         ParticleToMesh<<<blocks,threads>>>(xp_D,mesh_D);
//         fft(mesh_D,freq_D);
//         fft_ker(mesh_D,ker_D);
//         Poisson_Solver();
//         PP_Int();
//         
//      }
      
      


//      if(n_lost>0.9*Np)
//        k=Nstep;

//      //update trajectories
//      if(k%save_step==0)
//      {
//         int ind=k/save_step;
//         for(int i=0;i<Np;i++)
//         {
//            traj[ind*Np*3+i*3+0]=xp[i*3+0];
//            traj[ind*Np*3+i*3+1]=xp[i*3+1];
//            traj[ind*Np*3+i*3+2]=xp[i*3+2];
//         }

//      }

      //print completion percentage
      float perc=(k*100.)/Nstep;
      if(floor(perc)==perc && perc>0)
      {
         time (&end);
         float dif = difftime (end,start);
         dif=dif*100./perc-dif;
         cout<<"\r"<<perc<<"% 	Remaining time: "<<(int)dif<<"s    "<<flush;
      }

   }//end for k (time steps)


   cout<<"\r"<<"100%                                  "<<endl;
   cout<<"Particles lost: "<<n_lost<<endl;
   time (&end);
   float dif = difftime (end,start);
   cout<<"Elapsed time: "<<dif<<endl;
   

   //save result on file
   //save_traj(traj);


return 0;

}


