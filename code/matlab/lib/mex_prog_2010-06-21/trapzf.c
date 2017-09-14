/*  z = trapzf(x,y)
   Trapezoid sum computed with vector multiply.
   z = sum( (x(2:N) - x(1:N-1)) .* (y(1:N-1) + y(2:N)) )/2;
*/

#include "mex.h"

static double trapzf(double *area, double *x, double *y, int N)
{
  int i;

  if(N==1)   
    {*area = 0;
      return *area;
    }

  /* integration */ 
  *area = 0;
  
  for(i=1; i<N; i++)
  {*area = *area + (y[i]+y[i-1]) * (x[i] - x[i-1]);
  }
  *area = *area/2;
  
  return *area;
} 


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
int N, N1, T, T1, st, j;
double *area;
double *x, *y ;
/*  z = trapzf(x,y) */

/*  Get the dimensions of the input*/
N  = mxGetM(prhs[0]);
T  = mxGetN(prhs[0]);

N1 = mxGetM(prhs[1]);
T1 = mxGetN(prhs[1]);

if(N1!=N)
   mexErrMsgTxt("Input x and y must have the same length.");
if(T1!=T)
   mexErrMsgTxt("Input x and y must have the same length.");

if(N==1){
	N=T; T=1;
}
/* Create an mxArray of real values for the output */
plhs[0] = mxCreateDoubleMatrix(1,T,mxREAL);

/* Get the data */
x  = mxGetPr(prhs[0]);
y  = mxGetPr(prhs[1]);

/* the output pointer */
area = mxGetPr(plhs[0]);
   
/* actual computations in a subroutine */
st = 0;

for (j=0;j<T;j++){
   trapzf(area + j, x + st, y + st,N);
   st += N;
}
   
return;
}
