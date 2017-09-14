#include "mex.h"
#include "mexutil.h"

/*
   Correlation of dY and X in time embeded space
   dYX = error_corr_delay(X,dY,D,tau)
   dYX = dY * X';
   
 * Written by Masa-aki Sato 2008/05/01
 */

/* 
   error correlation
   dYX(n) = dY(n,:) * X(m,:)';  (m:fix)
   Z   = sum_t( Y(t) * X(t))
 */
void dyx_corr(double *x, double *y, double *z, int m, int n, int t)
{
	double *yt, *xt;
	int j;
	
	yt = y;
	xt = x;
	
  	for (j=0; j<t; j++) {
      	*z = *z + *xt * *yt;
      	xt += m;
      	yt += n;
    }
}

/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	/*   [dYX] = error_corr_delay(X,dY,D,tau)  */
	double *x,*w,*dy,*dyx,*dYX;
	int  m,M,n,N,T,Tx,D,j,k,NX,NY,NW,Nr,tau,Nall,ndims;
	int  wid,tid,yid,xid,xst,yst;
    int  *dims;
	mxArray *dyx_ptr;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=4) 
	  mexErrMsgTxt("4 inputs required.");
	if(nlhs!=1) 
	  mexErrMsgTxt("One output required.");
	
	/* x = mxGetPr(prhs[0]) Create a pointer to the inputs*/
	/* mxGetNumberOfElements: 配列の要素数 */
    /* mxGetNumberOfDimensions: 配列の次元数 */
	/* mxGetDimensions: 各次元内の要素数 */

	/*   [dYX] = error_corr_delay(X,Y,D,tau)  */
	x  = mxGetPr(prhs[0]);
	dy = mxGetPr(prhs[1]);
	D  = (int) mxGetScalar(prhs[2]);
	tau = (int) mxGetScalar(prhs[3]);
	
	/*  Get the dimensions of the matrix W 
	X   : M x Tx x Nr
	dY  : N x T  x Nr
	dYX : N x (M*D)
	*/

	dims = mxGetDimensions(prhs[0]);
	M  = dims[0];
	Tx = dims[1];
	
	ndims = mxGetNumberOfDimensions(prhs[1]);
	dims  = mxGetDimensions(prhs[1]);

	N = dims[0];
	T = dims[1];
	if( ndims==3 ){
		Nr=dims[2];
	}else{
		Nr=1;
	}

	NW = D*M;
	NY = N*T;
	NX = M*Tx;
	
	/*  Set the output pointer to the output matrix. */
	plhs[0] = mxCreateDoubleMatrix(N,NW,mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	dYX = mxGetPr(plhs[0]);
	
	dyx_ptr = mxCreateDoubleMatrix(1,1,mxREAL);
	dyx = mxGetPr(dyx_ptr);
/* 
	Xdelay(t) = [X(t+tau*(D-1)); ... ; X(t+tau); X(t)]
*/
/*			
printf("N= %d , M=%d , T=%d , D=%d , Tx=%d ,tau=%d\n", N,M,T,D,Tx,tau); 
*/

wid = 0;
tid = tau*(D-1);

/* dY = Y 
copy_data(y, dy, Nall);
*/

for (j=0;j<D;j++){
	for (m=0;m<M;m++){
		for (n = 0;n<N;n++){
			yid = 0;
			xid = m;
			*dyx = 0;
			
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;
				
				/* error correlation
				   dYX = dY(n,:) * X(m,:)';  
				*/
	 			dyx_corr(x + xst, dy + yst, dyx, M, N, T);
				
				xid += NX;
				yid += NY;
				/* END of k < Nr (Trial) */
			}
			
			dYX[wid + n] = *dyx;
			
			/* END of n < N (Y dim) */
		}
		wid += N;
		/* END of m < M (X dim) */
	}
	tid += -tau;
	/* END of j < D */
}

/* END */
}
