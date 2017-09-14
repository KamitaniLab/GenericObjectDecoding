#include "mex.h"
#include "mexutil.h"

double erfnc(double x)
{
/*	Œë·ŠÖ” y = erfnc(x); Hastings ‚Ì‹ß—® 
   erf(x) = 2/sqrt(pi) int_dt exp(-t^2)  0‚©‚çx‚Ü‚Å‚ÌÏ•ª
  erfc(x) =  1 - erf(x).

*/

	int i;
	double w, flg;
	static double a[6] = {	 1.0,           0.0705230784,  0.0422820123,
							 0.0092705272,	0.0001520143,  0.0002765672};
	flg = 1.0;
	if(x < 0.)
	{
		x = -x;
		flg = - 1.0;
	}
	
	if(x == 0.)	return 0.;
	if(x >= 10.)	return flg;
	
	w = 0.0000430638;
	for(i = 5; i >= 0; i--)	w = w * x + a[i];
	for(i = 0; i < 4; i++)	w *= w;
	w = (1. - 1. / w) * flg;
	return w;
}

/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	double *x, *y;
	int    m, n, i;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=1) 
	  mexErrMsgTxt("One input required.");
	if(nlhs!=1) 
	  mexErrMsgTxt("One output required.");
	
	
	/*  Create a pointer to the input */
	x = mxGetPr(prhs[0]);
	
	/*  Get the dimensions of the matrix input */
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);

	/*  Set the output pointer to the output matrix. 
		plhs[0] = mxCreateDoubleMatrix(my,ny, mxREAL);
		Create uninitialized matrix for speed up
	*/
	plhs[0] = mxCreateDoubleMatrixE(m,n,mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	y = mxGetPr(plhs[0]);
	
	for (i=0; i < n*m; i++) {
		*y = 1. - erfnc( *x );
		x++;
		y++;
	}
}
