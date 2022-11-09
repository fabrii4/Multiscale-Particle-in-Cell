//Numerical evaluation of hypergeometric function

//Compile with openmap parallel and gsl: g++ hypergeometric_function_gen.cpp -o app -fopenmp -lgsl -lgslcblas

#include <gsl/gsl_sf_hyperg.h>
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

//Hypergeometric functions approximation (needed for gpu loop)
#define Np 200000 //this number must be equal to the one in plasma.cpp
float LH1[Np];
float LH2[Np];
void hyper_func_calc(float LH1[Np],float LH2[Np], int np)
{
   #pragma omp parallel for
   for(int i=0;i<np;i++)
   {
      LH1[i]=gsl_sf_hyperg_2F1(0.75, 1.25, 2., i*2./np-1.);
      LH2[i]=gsl_sf_hyperg_2F1(1.75, 2.25, 3., i*2./np-1.);
   }
}

//MAIN 
int main()
{

  //initialize random seed
  srand((unsigned)time(0));

  //initialize hypergeometric function
  hyper_func_calc(LH1,LH2,Np);
  //check approximation
  for(int i=0;i<10;i++)
  {
     float z=(float)rand()/RAND_MAX*2.+-1.;
     float h1e=gsl_sf_hyperg_2F1(0.75, 1.25, 2., z);
     float h2e=gsl_sf_hyperg_2F1(1.75, 2.25, 3., z);
     float h1n=LH1[(int)floor((z+1.)*Np/2)];
     float h2n=LH2[(int)floor((z+1.)*Np/2)];
     cout<<z<<", "<<h1e<<", "<<h1n<<", "<<h2e<<", "<<h2n<<endl;
  }

   

   //save result on file LH1
//   ofstream ofile;
//   ofile.open("LH1.txt");
//   ofile<<setprecision(8);
//   for(int i=0;i<Np;i++)
//   {
//      ofile<<fixed<<LH1[i]<<" ";
//      if((i+1)%10==0)
//         ofile<<endl;
//   }
//   ofile.close();

   //save result on file LH2
//   ofile.open("LH2.txt");
//   ofile<<setprecision(8);
//   for(int i=0;i<Np;i++)
//   {
//      ofile<<fixed<<LH2[i]<<" ";
//      if((i+1)%10==0)
//         ofile<<endl;
//   }
//   ofile.close();

   //save result on file LH1
   ofstream ofile;
   ofile.open("LH1.bin", ios::out | ios::binary);
   for(int i=0;i<Np;i++)
   {
      ofile.write(reinterpret_cast<char *>(&LH1[i]), sizeof(LH1[i]));
   }
   ofile.close();

   //save result on file LH2
   ofile.open("LH2.bin", ios::out | ios::binary);
   for(int i=0;i<Np;i++)
   {
      ofile.write(reinterpret_cast<char *>(&LH2[i]), sizeof(LH2[i]));
   }
   ofile.close();



return 0;

}


