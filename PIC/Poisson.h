#ifndef POISSON_H
#define POISSON_H

#include <vector>
#include <fstream>

//extern __global__ void ParticleToMesh(float4 *xp, int *cp, float *rho, int Np, int3 N, float3 A, float3 B);
void ParticleToMesh(float4 *xp, float *rho, int *cell, int Np, int3 N, float3 A, float3 B);

void MeshToParticle(float4 *xp, float *fp, float *fg, int Np, int3 N, float3 A, float3 B, int sgn);

void init_mesh_struct(thrust::device_vector<float4>& xp_d,int Np,float3& A,float3& B, int3& N, float fgrid);

void init_mesh_struct_clust(thrust::host_vector<int>& cellind_h, int celloff, int Ncell, int Npc, float3 A, float3 B, int3 N, float3& cA, float3& cB, int3& cN, float3& fA, float3& fB, int3& fN, float fgrid);

void init_p(thrust::host_vector<float4>& xp, thrust::host_vector<float3>& vp, thrust::host_vector<float3>& ap, int Nneg, int qpos, int qneg);

//External fields, mesh electric field, Boris push and trajectory I/O (added to
//turn the static-potential solver into a time-evolving PIC; see Poisson.cu).
void load_ext_tables(const char* f1, const char* f2);
void free_ext_tables();
void GridField(float* fg_D, float3* eg_D, int3 N, float3 A);
void MeshToParticle3(float4* xp, float3* Ep, float3* eg, int Np, int3 N, float3 A, float3 B, int sgn);
void MeshFieldToParticle(float4* xp_D, float3* Ep_D, float* fg_D, int Np, int3 N, float3 A, float3 B, int sgn);
void BorisPush(float4* xp_D, float3* vp_D, float3* ap_D, float3* Eg_D, float3* Bint_D, float3* Eind_D, int Np, float dt, int bext_mode, int eext_mode, float mpos, float mneg, int radiative_corr);
void HandleLoss(float4* xp_D, float3* vp_D, float3* ap_D, int Np, int* nlost_D, float Rmax, float drb, float drt, float ar2b, float ar2t);
void BorisSubPush(float4* xp_D, float3* vp_D, float3* ap_D, float3* Eg_D, float3* Bint_D, float3* Eind_D, int* Nc_D, int Np, int3 N, float3 A, float3 B, float dt, int nsub, int thresh, int bext_mode, int eext_mode, float mpos, float mneg, int radiative_corr);
void store_traj(std::vector<float>& traj, thrust::device_vector<float4>& xp_d, int frame, int Np);
void save_traj_time(std::vector<float>& traj, int Nsave, int Np, const char* fname);
void store_frame(std::ofstream& fout, std::vector<float>& dpos, std::vector<float>& dneg, thrust::device_vector<float4>& xp_d, int frame, int Np, int Nbins, float rmax);
void save_density(std::vector<float>& dpos, std::vector<float>& dneg, int Nframes, int Nbins, float rmax, double dt, int save_step, const char* fname);
//selector values must match the enums in Poisson.cu
#define PIC_BEXT_NONE    0
#define PIC_BEXT_UNIFORM 1
#define PIC_BEXT_LOOPS   2
#define PIC_EEXT_NONE    0
#define PIC_EEXT_RINGS   1

void fft(float *rho_D, cufftComplex *frho_D, cufftComplex *fker_D, float3 A, float3 B, int3 N);

void fft_ker(cufftComplex *fker_D, float3 A, float3 B, int3 N);

void Poisson_Solver(cufftComplex *frho_D, cufftComplex *fker_D, int3 N);

void Poisson_Solver_thrust(thrust::device_vector<thrust::complex<float> >& frho_d, thrust::device_vector<thrust::complex<float> >& fker_d);

void fft_inv(cufftComplex *frho_D, float *rho_D, int3 N);

void save_traj(thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fp_d,  thrust::device_vector<float>& rho_d, thrust::device_vector<int>& Nc_d, float3 A, float3 B, int3 N);

void save_traj_tot(thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fp_d, float3 A, float3 B, int3 N);

void OrderCell(thrust::device_vector<int>& XC_d, thrust::device_vector<int>& CX_d, thrust::device_vector<int>& Nc_d, int Np, int3 N);

void PPField(float4 *xp_D, float *fp_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc);

void PPField_Tot(float4 *xp_D, float *fp_D, int Np);

void OrderXpToCell(thrust::device_vector<float4>& xp_d, thrust::device_vector<float4>& xpc_d, thrust::device_vector<int>& XC_d);

void OrderCellToFp(thrust::device_vector<float>& fp_d, thrust::device_vector<float>& fpc_d, thrust::device_vector<int>& CX_d);

void ParticleCluster(thrust::device_vector<int>& cellind_d, thrust::device_vector<int>& iclus_d, thrust::device_vector<int>& nclus_d, thrust::device_vector<int>& Nc_d,  int3 N, int thresh);

void FillCluster(thrust::device_vector<float4>& xpc_d, thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fpc_d, thrust::device_vector<float>& fp_d, thrust::host_vector<int>& Nc_h, thrust::host_vector<int>& offset_h, thrust::host_vector<int>& cellind_h, int celloff, int Ncell);

void ClusterToXp(thrust::device_vector<float>& fpc_d, thrust::device_vector<float>& fp_d, thrust::host_vector<int>& Nc_h, thrust::host_vector<int>& offset_h, thrust::host_vector<int>& cellind_h, int celloff, int Ncell);

void RecursiveMeshField(thrust::device_vector<float4>& xp_d, thrust::device_vector<float3>& Ep_d, thrust::device_vector<int>& Nc_d, float3 A, float3 B, int3 N, int Nc, int fgrid, int depth, int depthmax);

//Short-range particle-particle electric force correction (float3) and internal
//magnetic field from the moving charges.
void PPFieldF(float4 *xp_D, float3 *Ep_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc);
void MPFieldF(float4 *xp_D, float3 *Ep_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc);
void BIntField(float4 *xp_D, float3 *vp_D, float3 *Bp_D, int Np);
void EIndField(float4 *xp_D, float3 *vp_D, float3 *ap_D, float3 *Eind_D, int Np);
void RetardedMultipole(float4 *xp_D, float3 *vp_D, float3 *ap_D, float3 *Eind_D, float3 *Bp_D, int *Nc_D, int Np, int3 N, float3 A, float3 B, int lc);

void save_clust(thrust::device_vector<float4> xpc_d, int i, int Np, float3 A, float3 B, int3 N, float3 cA, float3 cB, int3 cN, float3 fA, float3 fB, int3 fN);

void save_fft(cufftComplex *frho_D, cufftComplex *fker_D, int3 N);


#endif
