#ifndef POISSON_H
#define POISSON_H

//extern __global__ void ParticleToMesh(float4 *xp, int *cp, float *rho, int Np, int3 N, float3 A, float3 B);
void ParticleToMesh(float4 *xp, float *rho, int *cell, int Np, int3 N, float3 A, float3 B);

void MeshToParticle(float4 *xp, float *fp, float *fg, int Np, int3 N, float3 A, float3 B, int sgn);

void init_mesh_struct(thrust::device_vector<float4>& xp_d,int Np,float3& A,float3& B, int3& N, float fgrid);

void init_mesh_struct_clust(thrust::host_vector<int>& cellind_h, int celloff, int Ncell, int Npc, float3 A, float3 B, int3 N, float3& cA, float3& cB, int3& cN, float3& fA, float3& fB, int3& fN, float fgrid);

void init_p(thrust::host_vector<float4>& xp, thrust::host_vector<float3>& vp, thrust::host_vector<float3>& ap);

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

void RecursiveMeshField(thrust::device_vector<float4>& xp_d, thrust::device_vector<float>& fp_d, thrust::device_vector<int>& Nc_d, float3 A, float3 B, int3 N, int Nc, int fgrid, int depth, int depthmax);

void save_clust(thrust::device_vector<float4> xpc_d, int i, int Np, float3 A, float3 B, int3 N, float3 cA, float3 cB, int3 cN, float3 fA, float3 fB, int3 fN);

void save_fft(cufftComplex *frho_D, cufftComplex *fker_D, int3 N);


#endif
