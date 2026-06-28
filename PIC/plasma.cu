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
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
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
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <cufft.h>

#include "Poisson.h"

using namespace std;
//using namespace thrust;


//Default simulation parameters (mirroring plasma_pp.cpp so the codes can be
//compared). These are the fall-back values used when ./param.txt is absent or
//does not override a given key (see Config / read_config below).
#define DEF_NP    100000        //number of particles
#define DEF_NNEG  1000          //number of negative particles (index < Nneg)
#define DEF_NSTEP 2000          //number of steps
#define DEF_NSAVE 2000          //number of trajectory frames saved
#define DEF_DT    (5*pow(10,-10)) //time step s
#define DEF_SEED  5             //random seed

//Physical Constants
#define MPI 3.1415926535897932385 //greek pi
#define e0 (8.854187817*pow(10,-12))  //electric suscettivity C^2/(N*m^2)
#define m0 (4*MPI*pow(10,-7)) //magnetic permeability N/A^2
#define qe (-1.60217733*pow(10,-19)) //elementary charge C (magnitude)
#define me (9.1093897*pow(10,-31))   //electron mass Kg
#define mprot (1.67265*pow(10,-27))  //proton mass Kg
#define mp (2*mprot)                       //deuteron mass Kg

//System parameters (ring geometry; must match EXT_AR/EXT_DR in Poisson.cu)
#define ar 0.01 //m charged-magnetic ring radius
#define dr 0.01 //m distance of the rings from center
#define ar2b (pow(ar-0.2*ar,2))
#define ar2t (pow(ar+0.2*ar,2))
#define drb (dr-0.2*dr)
#define drt (dr+0.2*dr)








//Run-time configuration. Defaults come from the DEF_* macros above; they can be
//overridden by a ./param.txt file (key value per line) and finally by the
//command-line arguments.
struct Config
{
   int Np=DEF_NP, Nneg=DEF_NNEG, Nstep=DEF_NSTEP, Nsave=DEF_NSAVE;
   double dt=DEF_DT;
   unsigned seed=DEF_SEED;
   //two particle species. The charge is an integer multiple of the elementary
   //charge |e|; the mass is "value" times a base mass selected by the unit
   //("e"=electron, "p"=proton, "kg"=absolute kg). Defaults reproduce an
   //electron (-1 e, 1 me) / deuteron (+1 e, 2 mp) plasma.
   int qpos=1, qneg=-1;       //charge numbers (qpos>0, qneg<0)
   double mpos=2,  mneg=1;    //mass values
   std::string mpos_unit="p", mneg_unit="e";
   int bext=PIC_BEXT_LOOPS;   //external B: 0 none, 1 uniform, 2 current loops
   int eext=PIC_EEXT_RINGS;   //external E: 0 none, 1 charged rings
   int depth=2;               //multiscale recursion depth (1 = mesh only)
   int pp_lc=-1;              //short-range PP correction range in cells (<0 off)
   int mp_lc=-1;              //cell-multipole correction range in cells (<0 off)
   int rad_corr=0;            //Radiative Larmor losses (0/1)
   int bint=0;                //internal magnetic field F3 (0/1)
   int eind=0;                //retarded internal electric field F2 (0/1)
   int rp_lc=-1;              //cell-multipole range for F2/F3 (<0 = direct O(N^2))
   int nsub=1;                //temporal sub-cycling factor in dense cells (1 = off)
   int subthresh=100;         //cell particle count above which to sub-cycle
   int dens_bins=80;          //radial bins for the density output
   double dens_rmax=0.06;     //m, maximum radius of the density profile
};

//Resolve a species mass "value [unit]" to kg.
double mass_kg(double value, const std::string& unit)
{
   if(unit=="e"||unit=="me")        return value*me;
   if(unit=="p"||unit=="mp"||unit=="mprot") return value*mprot;
   return value;                    //"kg" or anything else: absolute value
}

//Parse a "key value" configuration file. Lines starting with '#' and blank
//lines are ignored; unknown keys are silently skipped. Returns false if the
//file could not be opened (in which case the defaults are kept).
bool read_config(const char* fname, Config& c)
{
   std::ifstream f(fname);
   if(!f.is_open()) return false;
   std::string line;
   while(std::getline(f,line))
   {
      size_t h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h);
      std::istringstream is(line);
      std::string key; if(!(is>>key)) continue;
      if     (key=="Np")         is>>c.Np;
      else if(key=="Nneg"||key=="Nel") is>>c.Nneg;   //Nel kept as alias
      else if(key=="Nstep")      is>>c.Nstep;
      else if(key=="Nsave")      is>>c.Nsave;
      else if(key=="dt")         is>>c.dt;
      else if(key=="seed")       is>>c.seed;
      else if(key=="qpos")       is>>c.qpos;
      else if(key=="qneg")       is>>c.qneg;
      else if(key=="mpos")       is>>c.mpos;
      else if(key=="mneg")       is>>c.mneg;
      else if(key=="mpos_unit")  is>>c.mpos_unit;
      else if(key=="mneg_unit")  is>>c.mneg_unit;
      else if(key=="bext")       is>>c.bext;
      else if(key=="eext")       is>>c.eext;
      else if(key=="depth")      is>>c.depth;
      else if(key=="pp_lc")      is>>c.pp_lc;
      else if(key=="mp_lc")      is>>c.mp_lc;
      else if(key=="rad_corr")   is>>c.rad_corr;
      else if(key=="bint")       is>>c.bint;
      else if(key=="eind")       is>>c.eind;
      else if(key=="rp_lc")      is>>c.rp_lc;
      else if(key=="nsub")       is>>c.nsub;
      else if(key=="subthresh")  is>>c.subthresh;
      else if(key=="dens_bins")  is>>c.dens_bins;
      else if(key=="dens_rmax")  is>>c.dens_rmax;
   }
   return true;
}

//Print the effective configuration before the run starts.
void print_config(const Config& c, bool from_file)
{
   const char* bname[]={"none","uniform Bz","current loops"};
   const char* ename[]={"none","charged rings"};
   int bi=(c.bext>=0&&c.bext<=2)?c.bext:0, ei=(c.eext>=0&&c.eext<=1)?c.eext:0;
   std::cout<<"========== PIC simulation configuration ==========\n";
   std::cout<<"  parameters from : "<<(from_file?"param.txt (+ defaults)":"built-in defaults")<<"\n";
   std::cout<<"  particles       : "<<c.Np<<"  (negative "<<c.Nneg
            <<", positive "<<c.Np-c.Nneg<<")\n";
   std::cout<<"  positive species : q = "<<c.qpos<<" e,  m = "<<c.mpos<<" "<<c.mpos_unit
            <<" ("<<mass_kg(c.mpos,c.mpos_unit)<<" kg)\n";
   std::cout<<"  negative species : q = "<<c.qneg<<" e,  m = "<<c.mneg<<" "<<c.mneg_unit
            <<" ("<<mass_kg(c.mneg,c.mneg_unit)<<" kg)\n";
   std::cout<<"  steps / frames  : "<<c.Nstep<<" / "<<c.Nsave<<"\n";
   std::cout<<"  time step dt    : "<<c.dt<<" s\n";
   std::cout<<"  random seed     : "<<c.seed<<"\n";
   std::cout<<"  external E       : "<<ename[ei]<<"\n";
   std::cout<<"  external B       : "<<bname[bi]<<"\n";
   std::cout<<"  multiscale depth : "<<c.depth<<(c.depth>1?"":"  (mesh only)")<<"\n";
   std::cout<<"  temporal sub-cyc : "<<(c.nsub>1?(std::to_string(c.nsub)+"x in cells > "+std::to_string(c.subthresh)):std::string("off"))<<"\n";
   std::cout<<"  PP correction lc : "<<(c.pp_lc>=0?std::to_string(c.pp_lc):std::string("off"))<<"\n";
   std::cout<<"  cell-multipole lc: "<<(c.mp_lc>=0?std::to_string(c.mp_lc):std::string("off"))<<"\n";
   std::cout<<"  internal B (F3)  : "<<(c.bint?"on":"off")<<"\n";
   std::cout<<"  retarded E (F2)  : "<<(c.eind?"on":"off")<<"\n";
   if(c.bint||c.eind)
      std::cout<<"  F2/F3 method     : "<<(c.rp_lc>=0?("cell-multipole lc="+std::to_string(c.rp_lc)):std::string("direct O(N^2)"))<<"\n";
   std::cout<<"  Radiative losses : "<<(c.rad_corr?"on":"off")<<"\n";
   std::cout<<"  ring geometry    : a="<<ar<<" m, d="<<dr<<" m  (compile-time)\n";
   std::cout<<"  density profile  : "<<c.dens_bins<<" radial bins, rmax="<<c.dens_rmax<<" m\n";
   std::cout<<"==================================================\n"<<std::flush;
}

//MAIN
int main(int argc, char *argv[])
{

   //----- configuration: built-in defaults -> param.txt -> command line ------
   // Optional command-line overrides (highest priority):
   //   argv[1] external B (0 none,1 uniform,2 loops)  argv[2] external E (0,1)
   //   argv[3] multiscale depth   argv[4] PP range lc   argv[5] internal B (0/1)
   //   argv[6] retarded internal E F2 (0/1)   argv[7] cell-multipole range lc
   Config cfg;
   bool from_file = read_config("param.txt", cfg);
   if(argc>1) cfg.bext  = atoi(argv[1]);
   if(argc>2) cfg.eext  = atoi(argv[2]);
   if(argc>3) cfg.depth = atoi(argv[3]);
   if(argc>4) cfg.pp_lc = atoi(argv[4]);
   if(argc>5) cfg.bint  = atoi(argv[5]);
   if(argc>6) cfg.eind  = atoi(argv[6]);
   if(argc>7) cfg.mp_lc = atoi(argv[7]);
   if(argc>8) cfg.rad_corr = atoi(argv[8]);
   print_config(cfg, from_file);

   const int Np=cfg.Np, Nneg=cfg.Nneg, Nstep=cfg.Nstep, Nsave=cfg.Nsave;
   const double dt=cfg.dt;
   const int qpos=cfg.qpos, qneg=cfg.qneg;
   const float mpos_kg=(float)mass_kg(cfg.mpos,cfg.mpos_unit);
   const float mneg_kg=(float)mass_kg(cfg.mneg,cfg.mneg_unit);
   int bext_mode=cfg.bext, eext_mode=cfg.eext, depthmax=cfg.depth, pp_lc=cfg.pp_lc, mp_lc=cfg.mp_lc;
   int bint_on=cfg.bint, eind_on=cfg.eind, rp_lc=cfg.rp_lc;
   int nsub=cfg.nsub, subthresh=cfg.subthresh, rad_corr=cfg.rad_corr;
   const int dens_bins=cfg.dens_bins;
   const float dens_rmax=(float)cfg.dens_rmax;

   //plasma particles host vectors
   thrust::host_vector<float4> xp_h(Np); //position & charge-number vector
   thrust::host_vector<float3> vp_h(Np); //velocity vector
   thrust::host_vector<float3> ap_h(Np); //acceleration vector
   //The trajectory is streamed to disk frame by frame (see below) instead of
   //being held in RAM, so that large particle counts do not run out of memory.
   vector<float> dpos((size_t)Nsave*dens_bins, 0);       //positive-charge radial density
   vector<float> dneg((size_t)Nsave*dens_bins, 0);       //negative-charge radial density
   ofstream results("./results.bin", ios::out | ios::binary); //trajectory stream

   //initialize random seed
//   srand((unsigned)time(0));
   srand (cfg.seed);

   //initialize particles (positions, velocities, charge sign in w)
   init_p(xp_h,vp_h,ap_h,Nneg,qpos,qneg);

   //load the hypergeometric tables for the current-loop magnetic field
   load_ext_tables("LH1.bin","LH2.bin");

   //Device particles thrust vectors
   thrust::device_vector<float4> xp_d(xp_h); //positions (+charge sign)
   thrust::device_vector<float3> vp_d(vp_h); //velocities
   thrust::device_vector<float3> ap_d(ap_h); //accelerations
   thrust::device_vector<float4> xpc_d(Np);
   thrust::device_vector<float3> vpc_d(Np), apc_d(Np);
   xp_h.clear(); xp_h.shrink_to_fit();
   vp_h.clear(); vp_h.shrink_to_fit();
   ap_h.clear(); ap_h.shrink_to_fit();
   //particle-cell indices vectors
   thrust::device_vector<int> XC_d(Np); //cell indeces of particles
   thrust::device_vector<int> CX_d(Np); //particles indeces for cell
   //internal electric field (grid units, -grad phi) at each particle
   thrust::device_vector<float3> Ep_d(Np); //particle field (float3)
   //device counter of lost particles (accumulated across steps)
   thrust::device_vector<int> nlost_d(1, 0);
   //electric F2 and magnetic F3 internal terms
   thrust::device_vector<float3> Bp_d, Eind_d;
   float3 *Bp_D=NULL, *Eind_D=NULL;
  
   //cast thrust vectors to pointers to be processed by kernels
   float4* xp_D = raw_pointer_cast(&xp_d[0]);
   float3* vp_D = raw_pointer_cast(&vp_d[0]);
   float3* ap_D = raw_pointer_cast(&ap_d[0]);
   float3* Ep_D = raw_pointer_cast(&Ep_d[0]);
   int* XC_D = raw_pointer_cast(&XC_d[0]);
   int* CX_D = raw_pointer_cast(&CX_d[0]);
   int* nlost_D = raw_pointer_cast(&nlost_d[0]);

   time_t start,end;
   time (&start);

   int save_step=Nstep/Nsave; if(save_step<1) save_step=1;

   //number of lost particles
   int n_lost=0;

   float3 A, B;
   int3 N=make_int3(10, 10, 10);
   int Nc=(N.x-1)*(N.y-1)*(N.z-1);
   
   //density vector on the mesh
   //thrust::device_vector<float> rho_d(N.x*N.y*N.z, 0), fg_d(N.x*N.y*N.z, 0);
   //thrust::device_vector<int> Nc_d(Nc, 0);
   thrust::device_vector<float> rho_d, fg_d;
   thrust::device_vector<int> Nc_d;
   rho_d.reserve(N.x*N.y*N.z);
   fg_d.reserve(N.x*N.y*N.z);
   Nc_d.reserve(Nc);


   float fgrid=1; //Ngrid=Np/fgrid

   for(int k=0;k<Nstep;k++)
   {
      //clear the internal-field vector
      thrust::fill(thrust::device, Ep_d.begin(), Ep_d.end(), make_float3(0,0,0));

      //----- adaptive mesh from the current particle distribution -----
      init_mesh_struct(xp_d, Np, A, B, N, fgrid);
      int Ntot2=2*N.x*2*N.y*(2*N.z/2+1);
      Nc=(N.x-1)*(N.y-1)*(N.z-1);
      rho_d.assign(N.x*N.y*N.z, 0);
      fg_d.assign(N.x*N.y*N.z, 0);
      Nc_d.assign(Nc,0);
      //thrust::device_vector<float> rho_d(N.x*N.y*N.z, 0), fg_d(N.x*N.y*N.z, 0);
      //thrust::device_vector<int> Nc_d(Nc, 0);
      float* rho_D = raw_pointer_cast(&rho_d[0]);
      float* fg_D = raw_pointer_cast(&fg_d[0]);
      int* Nc_D = raw_pointer_cast(&Nc_d[0]);

      //----- deposit charge density and order particles by cell -----
      ParticleToMesh(xp_D, rho_D, XC_D, Np, N, A, B);
      OrderCell(XC_d, CX_d, Nc_d, Np, N);
      {
         //thrust::device_vector<float4> xpc_d(Np);
         //thrust::device_vector<float3> vpc_d(Np), apc_d(Np);
         OrderXpToCell(xp_d, xpc_d, CX_d);              //sort positions by cell
         thrust::gather(CX_d.begin(), CX_d.end(), vp_d.begin(), vpc_d.begin());
         thrust::gather(CX_d.begin(), CX_d.end(), ap_d.begin(), apc_d.begin());
         xp_d=xpc_d; vp_d=vpc_d; ap_d=apc_d;            //keep v,a,x in the same order
      }

      //----- particle-mesh electrostatic field: solve Poisson then E=-grad(phi) -----
      {
         thrust::device_vector<thrust::complex<float> > frho_d(Ntot2), fker_d(Ntot2);
         cufftComplex* frho_D = (cufftComplex*)raw_pointer_cast(&frho_d[0]);
         cufftComplex* fker_D = (cufftComplex*)raw_pointer_cast(&fker_d[0]);
         fft(rho_D, frho_D, fker_D, A, B, N);
         Poisson_Solver(frho_D, fker_D, N);
         fft_inv(frho_D, fg_D, N);
         MeshFieldToParticle(xp_D, Ep_D, fg_D, Np, N, A, B, 1);
      }

      //----- multiscale refinement of the internal force (depthmax>1) -----
      if(depthmax>1)
      {
         //cudaDeviceSynchronize();
         RecursiveMeshField(xp_d, Ep_d, Nc_d, A, B, N, Nc, fgrid, 0, depthmax);
      }

      //----- short-range electric correction (P^3M): direct PP and/or cell-multipole -----
      if(pp_lc>=0)
         PPFieldF(xp_D, Ep_D, Nc_D, Np, N, A, B, pp_lc);
      if(mp_lc>=0)
         MPFieldF(xp_D, Ep_D, Nc_D, Np, N, A, B, mp_lc);

      //----- retarded internal terms: F3 magnetic (bint) and F2 electric (eind) -----
      // Computed as direct O(N^2) sums, or with the GPU cell-multipole when rp_lc>=0.
      //thrust::device_vector<float3> Bp_d, Eind_d;
      //float3 *Bp_D=NULL, *Eind_D=NULL;
      if(bint_on)
      {
         Bp_d.resize(Np);
         thrust::fill(thrust::device, Bp_d.begin(), Bp_d.end(), make_float3(0,0,0));
         Bp_D=raw_pointer_cast(&Bp_d[0]);
      }
      if(eind_on)
      {
         Eind_d.resize(Np);
         thrust::fill(thrust::device, Eind_d.begin(), Eind_d.end(), make_float3(0,0,0));
         Eind_D=raw_pointer_cast(&Eind_d[0]);
      }
      if(bint_on || eind_on)
      {
         if(rp_lc>=0)
            RetardedMultipole(xp_D, vp_D, ap_D, Eind_D, Bp_D, Nc_D, Np, N, A, B, rp_lc);
         else
         {
            if(bint_on) BIntField(xp_D, vp_D, Bp_D, Np);
            if(eind_on) EIndField(xp_D, vp_D, ap_D, Eind_D, Np);
         }
      }

      //----- Boris push (external E/B added analytically per particle) -----
      // With temporal sub-cycling (nsub>1) particles in dense cells take nsub
      // sub-steps of dt/nsub, re-evaluating the external field each sub-step.
      if(nsub>1)
         BorisSubPush(xp_D, vp_D, ap_D, Ep_D, Bp_D, Eind_D, Nc_D, Np, N, A, B, (float)dt, nsub, subthresh, bext_mode, eext_mode, mpos_kg, mneg_kg, rad_corr);
      else
         BorisPush(xp_D, vp_D, ap_D, Ep_D, Bp_D, Eind_D, Np, (float)dt, bext_mode, eext_mode, mpos_kg, mneg_kg, rad_corr);

      //----- remove particles that hit the magnet rings or escape (on GPU) -----
      // A lost particle (crosses a ring, leaves the box |x|>Rmax, or becomes
      // non-finite) is parked inertly at the origin with charge number 0, so it
      // no longer contributes to the density/field and the mesh stays well
      // conditioned. Done on the device so large-Np runs are not bottlenecked by
      // a per-step host copy + CPU scan.
      HandleLoss(xp_D, vp_D, ap_D, Np, nlost_D, 0.5f, (float)drb, (float)drt, (float)ar2b, (float)ar2t);
      n_lost=nlost_d[0];

      //----- stream the trajectory frame and accumulate the density profile -----
      if(k%save_step==0)
      {
         int f=k/save_step;
         if(f<Nsave) store_frame(results, dpos, dneg, xp_d, f, Np, dens_bins, dens_rmax);
      }

      if(n_lost>0.9*Np) { cout<<"\nMost particles lost, stopping."<<endl; break; }

      //print completion percentage
      float perc=(k*100.)/Nstep;
      if(floor(perc)==perc && perc>0)
      {
         time (&end);
         float dif = difftime (end,start);
         int t_remain=(int)(dif*100./perc-dif);
         cout<<"\r"<<perc<<"% 	Remaining time: "<<t_remain<<"s    Elapsed time: "<<dif<<"s    "<<flush;
      }

   }//end for k (time steps)


   cout<<"\r"<<"100%                                  "<<endl;
   cout<<"Particles lost: "<<n_lost<<endl;
   time (&end);
   float dif = difftime (end,start);
   cout<<"Elapsed time: "<<dif<<"s       "<<endl;

   //finalize the streamed trajectory, write the densities and the final state
   results.close();
   //save_density(dpos, dneg, Nsave, dens_bins, dens_rmax, dt, save_step, "./density.txt");
   thrust::device_vector<float> fpdummy(Np,0);
   save_traj_tot(xp_d, fpdummy, A, B, N);   //writes ./xp.bin and ./grid.txt
   free_ext_tables();

return 0;

}


