#include "mex.h"
#include "mexutil.h"

/*
    error calculation in time embeded space
    dY = dY - W * X
    
 * Written by Masa-aki Sato 2008/05/01
*/
/* 
   copy data
 */
void copy_data(double *x, double *y, int t)
{
	double *yt, *xt;
	int j;
	
	yt = y;
	xt = x;
	
  	for (j=0; j<t; j++) {
      	*yt = *xt;
      	xt++;
      	yt++;
    }
}

/* 
    % error update
    (m :fix)
    dY(n,t) = dY(n,t) - dW(n) * X(m,t);
 */
void dy_err(double *x, double w, double *y, int m, int n, int t)
{
	double *yt, *xt;
	int j;
	
	yt = y;
	xt = x;
	
	for (j=0; j<t; j++) {
      	*yt = *yt - *xt * w;
      	xt += m;
      	yt += n;
    }
}

/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	/*   dY = error_delay_time(X,Y,W,tau,Ntr);  */
	double *x,*y,*w,*dy,*Ntr;
	int  m,M,n,N,t,T,Tx,D,j,k,NX,NY,Nr,tau,Nall,ndims;
	int  wid,tid,yid,xid,xst,yst,wst;
    int  *dims;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=5) 
	  mexErrMsgTxt("5 inputs required.");
	if(nlhs!=1) 
	  mexErrMsgTxt("One output required.");
	
	/* x = mxGetPr(prhs[0]) Create a pointer to the inputs*/
	/* mxGetNumberOfElements: 配列の要素数 */
    /* mxGetNumberOfDimensions: 配列の次元数 */
	/* mxGetDimensions: 各次元内の要素数 */

	/*   dY = error_delay_time(X,Y,W,tau);  */
	x  = mxGetPr(prhs[0]);
	y  = mxGetPr(prhs[1]);
	w  = mxGetPr(prhs[2]);
	tau = (int) mxGetScalar(prhs[3]);
	Ntr = mxGetPr(prhs[4]);
	
	/*  Get the dimensions of the matrix W 
	X : M x Tx x Nr
	Y : N x T  x Nr
	W : N x (M*D)
	*/

	dims = mxGetDimensions(prhs[0]);
	M  = dims[0];
	Tx = dims[1];
	
	Nall  = mxGetNumberOfElements(prhs[1]);
	ndims = mxGetNumberOfDimensions(prhs[1]);
	dims  = mxGetDimensions(prhs[1]);

	N = dims[0];
	T = dims[1];
	if( ndims==3 ){
		Nr=dims[2];
	}else{
		Nr=1;
	}

	D = mxGetN(prhs[2])/M;
	NY = N*T;
	NX = M*Tx;
	
	/*  Set the output pointer to the output matrix. */
	plhs[0] = mxCreateNumericArray(ndims,dims,mxDOUBLE_CLASS,mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	dy = mxGetPr(plhs[0]);
	
/* 
	dY = Y - W * X
	Y : N x T  x Nr
	X : M x Tx x Nr
	W : N x (M*D)
	Xdelay(t) = [X(t+tau*(D-1)); ... ; X(t+tau); X(t)]
*/
/*			
printf("N= %d , M=%d , T=%d , D=%d , Tx=%d ,t=%d\n", N,M,T,D,Tx,tau); 
printf("W= [%e , %e , %e] (%d) \n", Wout[w_cnt],w[w_cnt],dw,m); 
*/

wid = 0;
tid = tau*(D-1);

/* dY = Y */
copy_data(y, dy, Nall);

for (j=0;j<D;j++){
	for (m=0;m<M;m++){
		for (n = 0;n<N;n++){
			wst = wid + n;

			yid = 0;
			xid = m;
			
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;
				
	 			dy_err(x + xst, *(w + wst), dy + yst, M, N, *(Ntr+k));
				/* dY(n,yid:yend) = dY(n,yid:yend) - W(n,wid) * X(xid:M:xend);*/
				
				xid += NX;
				yid += NY;
				/* END of k < Nr (Trial) */
			}
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
