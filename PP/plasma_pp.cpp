//Plasma simulation with Boris integration method 

//Compile: g++ plasma.cpp -o app
//Compile with openmap parallel: g++ plasma.cpp -o app -fopenmp
//Compile with openmap parallel and gsl: g++ plasma.cpp -o app -fopenmp -lgsl -lgslcblas
//Compile with openacc cuda: g++ plasma.cpp -o app -fopenacc
//Compile with openacc cuda (pgi compiler): /opt/pgi/linux86-64/17.4/bin/pgc++ plasma.cpp -o app -acc -ta=nvidia:maxwell -fast
//compile with openacc (verbose): /opt/pgi/linux86-64/17.4/bin/pgc++ plasma.cpp -o app -acc -ta=nvidia:maxwell,time -Minfo=accel -fast
//compile with openacc unified memory: /opt/pgi/linux86-64/17.4/bin/pgc++ plasma.cpp -o app -acc -ta=tesla:managed -fast


//#include <gsl/gsl_sf_hyperg.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <ctime> 
#include <stdlib.h> 
#include <math.h> 
#include <algorithm>
#include <functional>


using namespace std;


//Main parameters
#define Ne 1800 //number of particles
#define Nel 1000 //number of electrons
#define Nstep 2000 //number of steps
#define Nsave 2000 //number of trajectory points saved
#define dt (5*pow(10,-10)) //time step s
//#define dt (5*pow(10,-10)) //time step s

//Physical Constants
#define MPI 3.1415926535897932385 //greek pi
#define e0 (8.854187817*pow(10,-12))  //electric suscettivity C^2/(N*m^2)
#define m0 (4*MPI*pow(10,-7)) //magnetic permeability N/A^2
#define qe (-1.60217733*pow(10,-19)*10000) //electron charge C
#define me (9.1093897*pow(10,-31)*10000)   //electrom mass Kg
#define mp (2*1.67265*pow(10,-27)*10000)   //deuteron mass Kg

//System parameters
#define ar 0.01 //m charged-magnetic ring radius
#define dr 0.02 //m distance of the rings from center
#define ar2b (pow(ar-0.2*ar,2))
#define ar2t (pow(ar+0.2*ar,2))
#define drb (dr-0.2*dr)
#define drt (dr+0.2*dr)


//plasma particles vectors
//position vector
float xe[Ne][3];
//velocity vector
float ve[Ne][3];
//acceleration vector
float ae[Ne][3];
//vector of trajectories
float traj[Nsave][Ne][3];
//vector of lost particles
float lost[Ne][3];

//Hypergeometric functions approximation (needed for gpu loop)
#define Np 200000
float LH1[Np];
float LH2[Np];
void hyper_func_calc(float LH1[Np],float LH2[Np], int np)
{
   ifstream ifile;
   ifile.open("LH1.bin", ios::in | ios::binary);
   if(ifile.is_open())
   {
      for(int i=0;i<np;i++)
      {
         ifile.read(reinterpret_cast<char *>(&LH1[i]), sizeof(LH1[i]));
      }
      ifile.close();
   }
   else
      cout<<"Hypergeometric data file not found!"<<endl;
   ifile.open("LH2.bin", ios::in | ios::binary);
   if(ifile.is_open())
   {
      for(int i=0;i<np;i++)
      {
         ifile.read(reinterpret_cast<char *>(&LH2[i]), sizeof(LH2[i]));
      }
      ifile.close();
   }
   else
      cout<<"Hypergeometric data file not found!"<<endl;
}


//External electric field
void E_ext(float x[3],float E[3]) //electric field from charged ring
{
   float Q=-2*pow(10,-9)/10; //C (charges )
   float a=ar; //m   //ring radius
   float d=dr; //m distance from center
   float q=Q/(4*MPI*e0);
   int Na=100;
   float e[3];
   e[0]=0.;
   e[1]=0.;
   e[2]=0.;
   //field 1
   for(int i=0;i<Na;i++)
   {
      float t=i*2.*MPI/Na;
      float norm=pow(pow(x[0]+a*cos(t),2)+pow(x[1]+a*sin(t),2)+pow(x[2]+d,2),1.5);
      e[0]=e[0]+q*(x[0]+a*cos(t))/norm;
      e[1]=e[1]+q*(x[1]+a*sin(t))/norm;
      e[2]=e[2]+q*(x[2]+d)/norm;
   }
   //field 2
   d=-d;
   for(int i=0;i<Na;i++)
   {
      float t=i*2.*MPI/Na;
      float norm=pow(pow(x[0]+a*cos(t),2)+pow(x[1]+a*sin(t),2)+pow(x[2]+d,2),1.5);
      e[0]=e[0]+q*(x[0]+a*cos(t))/norm;
      e[1]=e[1]+q*(x[1]+a*sin(t))/norm;
      e[2]=e[2]+q*(x[2]+d)/norm;
   }
   E[0]=e[0];
   E[1]=e[1];
   E[2]=e[2];
}

void E_ext_charge(float x[3],float E[3]) //electric field from 2 negative charges 
{
   float Q=-2*pow(10,-9)*1000; //C (charges )
   float dy=0.;//2.1*0.01; //m   //distance between charges y
   float d=0.05; //m distance from center z
   float q=Q/(4*MPI*e0);
   float norm0=pow(pow(x[0],2)+pow(x[1],2)+pow(x[2]+d,2),1.5);
   float norm1=pow(pow(x[0],2)+pow(x[1]+dy,2)+pow(x[2]-d,2),1.5);
   E[0]=q*(x[0]/norm0+x[0]/norm1);
   E[1]=q*(x[1]/norm0+(x[1]+dy)/norm1);
   E[2]=q*((x[2]+d)/norm0+(x[2]-d)/norm1);
}

void E_ext_null(float x[3],float E[3]) //electric field null
{
   E[0]=0.;
   E[1]=0.;
   E[2]=0.;
}

//Internal electric field
void E_int(float x[Ne][3], float v[Ne][3], float a[Ne][3], int i, float E[3])
{
   float e[3];
   e[0]=0.;
   e[1]=0.;
   e[2]=0.;
   float q0=qe/(4*MPI*e0);
   float q1=-m0*qe/(4*MPI);
   int sign=1; //charge sign
   for(int j=0;j<Ne;j++)
   {
      if(j>=Nel)
        sign=-1;
      if(j!=i)
      {
	 float norm2 = pow(x[i][0]-x[j][0],2)+pow(x[i][1]-x[j][1],2)+pow(x[i][2]-x[j][2],2);
	 float norm3 = pow(norm2,1.5);
         float prod_xv=(x[i][0]-x[j][0])*v[j][0]+(x[i][1]-x[j][1])*v[j][1]+(x[i][2]-x[j][2])*v[j][2];
         e[0]=e[0]+(sign*q0*(x[i][0]-x[j][0])+sign*q1*(a[j][0]*norm2+v[j][0]*prod_xv))/norm3;
         e[1]=e[1]+(sign*q0*(x[i][1]-x[j][1])+sign*q1*(a[j][1]*norm2+v[j][1]*prod_xv))/norm3;
         e[2]=e[2]+(sign*q0*(x[i][2]-x[j][2])+sign*q1*(a[j][2]*norm2+v[j][2]*prod_xv))/norm3;
      }
   }
   E[0]=e[0];
   E[1]=e[1];
   E[2]=e[2];
}

//External magnetic field

void B_ext(float x[3], float B[3], float LH1[Np], float LH2[Np]) //static
{
   B[0]=0.;
   B[1]=0.;
   B[2]=2.;
}
void B_ext_null(float x[3], float B[3], float LH1[Np], float LH2[Np]) //magnetic cusp
{
   //float J=3.5*pow(10,8); //current A
   float J=3.5*pow(10,4)*1000; //current
   float a=ar; //m radius
   float d=dr; //m distance from center
   //field 1
   float xyz=pow(pow(a,2)+pow(x[0],2)+pow(x[1],2)+pow(d+x[2],2),2);
   float C=pow(a,2)*J*m0/(8*pow(xyz,2.25));
   float p=4*pow(a,2)*(pow(x[0],2)+pow(x[1],2))/xyz;
   float H1=LH1[(int)floor((p+1.)*Np/2)];
   float H2=LH2[(int)floor((p+1.)*Np/2)];
   float bxy=3*C*(d+x[2])*(2*xyz*H1+5*pow(a,2)*(pow(x[0],2)+pow(x[1],2))*H2);
   B[0]=bxy*x[0];
   B[1]=bxy*x[1];
   B[2]=0.5*C*(-4*(-2*pow(a,2)+pow(x[0],2)+pow(x[1],2)-2*pow(d+x[2],2))*xyz*H1+15*pow(a,2)*(pow(x[0],2)+pow(x[1],2))*(pow(a,2)-pow(x[0],2)-pow(x[1],2)+pow(d+x[2],2))*H2);
   //field 2
   J=J; //current
   d=-d; //distance from center
   xyz=pow(pow(a,2)+pow(x[0],2)+pow(x[1],2)+pow(d+x[2],2),2);
   C=pow(a,2)*J*m0/(8*pow(xyz,2.25));
   p=4*pow(a,2)*(pow(x[0],2)+pow(x[1],2))/xyz;
   H1=LH1[(int)floor((p+1.)*Np/2)];
   H2=LH2[(int)floor((p+1.)*Np/2)];
   bxy=3*C*(d+x[2])*(2*xyz*H1+5*pow(a,2)*(pow(x[0],2)+pow(x[1],2))*H2);
   B[0]=B[0]+bxy*x[0];
   B[1]=B[1]+bxy*x[1];
   B[2]=B[2]+0.5*C*(-4*(-2*pow(a,2)+pow(x[0],2)+pow(x[1],2)-2*pow(d+x[2],2))*xyz*H1+15*pow(a,2)*(pow(x[0],2)+pow(x[1],2))*(pow(a,2)-pow(x[0],2)-pow(x[1],2)+pow(d+x[2],2))*H2);
//   //field 3
//   J=3.5*pow(10,9); //current
//   a=0.01; //radius
//   d=0.01; //distance from center
//   float dy=0;//2.1*a;
//   xyz=pow(pow(a,2)+pow(x[0],2)+pow(x[1]+dy,2)+pow(d+x[2],2),2);
//   C=pow(a,2)*J*m0/(8*pow(xyz,2.25));
//   p=4*pow(a,2)*(pow(x[0],2)+pow(x[1]+dy,2))/xyz;
//   H1=LH1[(int)floor((p+1.)*Np/2)];
//   H2=LH2[(int)floor((p+1.)*Np/2)];
//   bxy=3*C*(d+x[2])*(2*xyz*H1+5*pow(a,2)*(pow(x[0],2)+pow(x[1]+dy,2))*H2);
//   B[0]=B[0]+bxy*x[0];
//   B[1]=B[1]+bxy*(x[1]+dy);
//   B[2]=B[2]+0.5*C*(-4*(-2*pow(a,2)+pow(x[0],2)+pow(x[1]+dy,2)-2*pow(d+x[2],2))*xyz*H1+15*pow(a,2)*(pow(x[0],2)+pow(x[1]+dy,2))*(pow(a,2)-pow(x[0],2)-pow(x[1]+dy,2)+pow(d+x[2],2))*H2);
//   //field 4
//   J=-J; //current
//   d=-d; //distance from center
//   xyz=pow(pow(a,2)+pow(x[0],2)+pow(x[1]+dy,2)+pow(d+x[2],2),2);
//   C=pow(a,2)*J*m0/(8*pow(xyz,2.25));
//   p=4*pow(a,2)*(pow(x[0],2)+pow(x[1]+dy,2))/xyz;
//   H1=LH1[(int)floor((p+1.)*Np/2)];
//   H2=LH2[(int)floor((p+1.)*Np/2)];
//   bxy=3*C*(d+x[2])*(2*xyz*H1+5*pow(a,2)*(pow(x[0],2)+pow(x[1]+dy,2))*H2);
//   B[0]=B[0]+bxy*x[0];
//   B[1]=B[1]+bxy*(x[1]+dy);
//   B[2]=B[2]+0.5*C*(-4*(-2*pow(a,2)+pow(x[0],2)+pow(x[1]+dy,2)-2*pow(d+x[2],2))*xyz*H1+15*pow(a,2)*(pow(x[0],2)+pow(x[1]+dy,2))*(pow(a,2)-pow(x[0],2)-pow(x[1]+dy,2)+pow(d+x[2],2))*H2);
}

//Internal magnetic field
#pragma acc routine seq
void B_int(float x[Ne][3], float v[Ne][3], int i, float B[3])
{
   float bi[3];
   bi[0]=0.;
   bi[1]=0.;
   bi[2]=0.;
   float q=-m0*qe/(4*MPI);
   int sign=1; //charge sign
   #pragma acc loop seq
   for(int j=0;j<Ne;j++)
   {
      if(j>=Nel)
        sign=-1;
      if(j!=i)
      {
	 float norm = pow(pow(x[i][0]-x[j][0],2)+pow(x[i][1]-x[j][1],2)+pow(x[i][2]-x[j][2],2),1.5);
         bi[0]=bi[0]+sign*((x[i][1]-x[j][1])*v[j][2]-(x[i][2]-x[j][2])*v[j][1])/norm;
         bi[1]=bi[1]+sign*((x[i][2]-x[j][2])*v[j][0]-(x[i][0]-x[j][0])*v[j][2])/norm;
         bi[2]=bi[2]+sign*((x[i][0]-x[j][0])*v[j][1]-(x[i][1]-x[j][1])*v[j][0])/norm;         
      }
   }
   bi[0]=bi[0]*q;
   bi[1]=bi[1]*q;
   bi[2]=bi[2]*q;

   B[0]=bi[0];
   B[1]=bi[1];
   B[2]=bi[2];
}






//MAIN 
int main()
{

  time_t start,end;
  time (&start);

  int save_step=Nstep/Nsave;

  //number of lost particles
  int n_lost=0;

  //initialize random seed
//  srand((unsigned)time(0));

  //initialize hypergeometric function
  hyper_func_calc(LH1,LH2,Np);

  //initialize initial positions
  for(int i=0;i<Ne;i++)
  {
     //float xmin=-0.01;
     float xmax=0.003;
     //xe[i][0]=(float)rand()/RAND_MAX*(xmax-xmin)*2+xmin*2;
     //xe[i][1]=(float)rand()/RAND_MAX*(xmax-xmin)*2+xmin*2;
     //xe[i][2]=(float)rand()/RAND_MAX*(xmax-xmin)+xmin;
     float d=(float)rand()/RAND_MAX*xmax;
     float t=(float)rand()/RAND_MAX*2.*MPI;
     float f=(float)rand()/RAND_MAX*MPI;
     xe[i][0]=d*cos(t)*sin(f);
     xe[i][1]=d*sin(t)*sin(f);
     xe[i][2]=d*cos(f); 

  }
  for(int i=0;i<Ne;i++)
  {
     traj[0][i][0]=xe[i][0];
     traj[0][i][1]=xe[i][1];
     traj[0][i][2]=xe[i][2];
  }

  //initialize initial velocities and accelerations
  for(int i=0;i<Ne;i++)
  {
     float vmax=10000;
     ve[i][0]=(float)rand()/RAND_MAX*vmax*2-vmax;
     ve[i][1]=(float)rand()/RAND_MAX*vmax*2-vmax;
     ve[i][2]=(float)rand()/RAND_MAX*vmax*2-vmax;
     ae[i][0]=0.;
     ae[i][1]=0.;
     ae[i][2]=0.;
  }

  //initial protons positions
  for(int i=Nel;i<Ne;i++)
  //for(int i=0;i<0;i++)
  {
     //float d=0.15;
     float d=(float)rand()/RAND_MAX*0.004+0.04;
     float t=(float)rand()/RAND_MAX*2.*MPI;
     //float f=(float)rand()/RAND_MAX*MPI/2.+MPI/4.;
     float f1=(float)rand()/RAND_MAX*MPI/128;
     float f=(float)rand()/RAND_MAX*MPI/128+127*MPI/128;
     if (rand()%2>0)
        f=f1;
     xe[i][0]=d*cos(t)*sin(f);
     xe[i][1]=d*sin(t)*sin(f);
     xe[i][2]=d*cos(f); 
     //ve[i][0]=0.;
     //ve[i][1]=0.;
     //ve[i][2]=0.;
  }

  #pragma acc data copyin(LH1,LH2)
  //#pragma acc kernels loop seq copyin(xe,ve,ae) copyout(traj)
  for(int k=0;k<Nstep;k++)
  {
     float vep[Ne][3];
     //#pragma acc kernels loop independent copyin(xe,ve,ae) copyout(vep) gang(32) vector(32)
     #pragma acc kernels copyin(xe,ve,ae) copyout(vep)
     #pragma acc loop independent device_type(NVIDIA) gang worker vector
     for(int i=0;i<Ne;i++)
     {
        //calculate electric and magnetic field
        float e_int[3];
        float e_ext[3];
        float b_int[3];
        float b_ext[3];
        E_int(xe,ve,ae,i,e_int);
        E_ext(xe[i],e_ext);
        B_int(xe,ve,i,b_int);
        B_ext(xe[i],b_ext,LH1,LH2);
        float ef[3];
        float t[3];
        float s[3];
        int sign=1;
        float mass=me;
        if(i>=Nel)
        {
           sign=-1;
           mass=mp;
        }
        ef[0]=sign*qe/mass*dt/2.*(e_ext[0]+e_int[0]);
        ef[1]=sign*qe/mass*dt/2.*(e_ext[1]+e_int[1]);
        ef[2]=sign*qe/mass*dt/2.*(e_ext[2]+e_int[2]);
        t[0]=sign*qe/mass*dt/2.*(b_ext[0]+b_int[0]);
        t[1]=sign*qe/mass*dt/2.*(b_ext[1]+b_int[1]);
        t[2]=sign*qe/mass*dt/2.*(b_ext[2]+b_int[2]);
        float norm2_t=pow(t[0],2)+pow(t[1],2)+pow(t[2],2);
        s[0]=2./(1.+norm2_t)*t[0];
        s[1]=2./(1.+norm2_t)*t[1];
        s[2]=2./(1.+norm2_t)*t[2];
        //calculate new velocities ve->vep
        float ve1[3];
        float ve2[3];
        float ve3[3];
        for(int n=0;n<3;n++)
        {
           ve1[n]=0.;
           ve2[n]=0.;
           ve3[n]=0.;
        }
        ve1[0]=ve[i][0]+ef[0];
        ve1[1]=ve[i][1]+ef[1];
        ve1[2]=ve[i][2]+ef[2];
        ve2[0]=ve1[0]+ve1[1]*t[2]-ve1[2]*t[1];
        ve2[1]=ve1[1]+ve1[2]*t[0]-ve1[0]*t[2];
        ve2[2]=ve1[2]+ve1[0]*t[1]-ve1[1]*t[0];
        ve3[0]=ve1[0]+ve2[1]*s[2]-ve2[2]*s[1];
        ve3[1]=ve1[1]+ve2[2]*s[0]-ve2[0]*s[2];
        ve3[2]=ve1[2]+ve2[0]*s[1]-ve2[1]*s[0];
        vep[i][0]=ve3[0]+ef[0];
        vep[i][1]=ve3[1]+ef[1];
        vep[i][2]=ve3[2]+ef[2];

     }//end for i (particles)

     //update positions accelerations and velocities
     for(int i=0;i<Ne;i++)
     {
        xe[i][0]=xe[i][0]+vep[i][0]*dt;
        xe[i][1]=xe[i][1]+vep[i][1]*dt;
        xe[i][2]=xe[i][2]+vep[i][2]*dt;

        ae[i][0]=(vep[i][0]-ve[i][0])/dt;
        ae[i][1]=(vep[i][1]-ve[i][1])/dt;
        ae[i][2]=(vep[i][2]-ve[i][2])/dt;

        ve[i][0]=vep[i][0];
        ve[i][1]=vep[i][1];
        ve[i][2]=vep[i][2];

        //delete particles which escape or hit magnets
        float dist2=pow(xe[i][0],2)+pow(xe[i][1],2);
        //float dist3=dist2+pow(xe[i][2],2);
        if((((xe[i][2]>drb && xe[i][2]<drt)||(xe[i][2]<-drb && xe[i][2]>-drt)) && dist2>ar2b && dist2<ar2t))
        //if((((xe[i][2]>0.0045 && xe[i][2]<0.0055)||(xe[i][2]<-0.0045 && xe[i][2]>-0.0055)) && dist2>0.000081 && dist2<0.000121))
//||(dist3>0.04 && dist3<pow(10,2)))
        //if(dist3>0.04 && dist3<pow(10,2))
        {
           n_lost=n_lost+1;
           lost[i][0]=xe[i][0];
           lost[i][1]=xe[i][1];
           lost[i][2]=xe[i][2];
           float d=n_lost*pow(10,3);
           float t=(float)rand()/RAND_MAX*2*MPI;
           float f=(float)rand()/RAND_MAX*MPI;
           xe[i][0]=d*cos(t)*sin(f);
           xe[i][1]=d*sin(t)*sin(f);
           xe[i][2]=d*cos(f); 
           ve[i][0]=ae[i][0]=0.;
           ve[i][1]=ae[i][1]=0.;
           ve[i][2]=ae[i][2]=0.; 
        }
     }

     if(n_lost>0.9*Ne)
        k=Nstep;

     //update trajectories
     if(k%save_step==0)
     {
        int ind=k/save_step;
        for(int i=0;i<Ne;i++)
        {
           traj[ind][i][0]=xe[i][0];
           traj[ind][i][1]=xe[i][1];
           traj[ind][i][2]=xe[i][2];
        }

     }

     //print percentage
     float perc=(k*100.)/Nstep;
     if(floor(perc)==perc && perc>0)
     {
        time (&end);
        float dif = difftime (end,start);
        dif=dif*100./perc-dif;
        cout<<"\r"<<perc<<"% 	Remaining time: "<<(int)dif<<"s    "<<flush;
        //if (perc==2)
        //{
        //   cout<<"\r Remaining time: "<<dif<<"s"<<flush;
        //}
     }

  }//end for k (time steps)

  cout<<"\r"<<"100%                                  "<<endl;
  cout<<"Particles lost: "<<n_lost<<endl;


  time (&end);
  float dif = difftime (end,start);
  cout<<"Elapsed time: "<<dif<<endl;
   

   //save result on file
   bool save=true;
   bool gnuplot=true;
   if(save)
   {
   ofstream ofile;
   if(gnuplot)  //format output for gnuplot
   {
      ofile.open("./gnuplot/results.bin", ios::out | ios::binary);
      for(int i=0;i<Nsave;i++)
      {
         for(int j=0;j<Ne;j++)
         {
            struct X
            {
                float a, b, c;
            } x;
            x.a = traj[i][j][0];
            x.b = traj[i][j][1];
            x.c = traj[i][j][2];
            ofile.write(reinterpret_cast<char *>(&x), sizeof(x));
         }
      }
      ofile.close();

      ofile.open("./gnuplot/lost.bin", ios::out | ios::binary);
      for(int j=0;j<Ne;j++)
      {
         if(lost[j][0]!=0 || lost[j][1]!=0 || lost[j][2]!=0)
         {
            struct X
            {
                float a, b, c;
            } x;
            x.a = lost[j][0];
            x.b = lost[j][1];
            x.c = lost[j][2];
            ofile.write(reinterpret_cast<char *>(&x), sizeof(x));
         }
      }
      ofile.close();
   }
   else //format output for Mathematica
   {
      ofile.open("results.txt");
      ofile<<setprecision(8);
      ofile<<"{";
      for(int i=0;i<Nsave;i++)
      {
         ofile<<"{";
         for(int j=0;j<Ne-1;j++)
         {
            ofile<<fixed<<"{"<<traj[i][j][0]<<","<<traj[i][j][1]<<","<<traj[i][j][2]<<"},";
         }
         ofile<<fixed<<"{"<<traj[i][Ne-1][0]<<","<<traj[i][Ne-1][1]<<","<<traj[i][Ne-1][2]<<"}";
         if(i<Nsave-1)
            ofile<<"},"<<endl;
         else
            ofile<<"}"<<endl;
      }
      ofile <<"}"<<endl;
   }
   }


return 0;

}


