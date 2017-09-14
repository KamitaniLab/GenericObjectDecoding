#include "mex.h"
#include "mexutil.h"

/*
  Output calculation in time embeded space
  	Y = weight_out_delay_time(X,W,T,tau);  
  	Y = W * X
	Y(n,t) = sum_{m,j} W(n,m,j) * X(m,t - j*tau)
  
 * Written by Masa-aki Sato 2008/05/01
*/

/* 
    % Output
    (m :fix)
    Y(n,t) = Y(n,t) + W(n) * X(m,t);
 */
void y_wx(double *x, double w, double *y, int m, int n, int t)
{
	double *yt, *xt;
	int j;
	
	yt = y;
	xt = x;
	
	for (j=0; j<t; j++) {
      	*yt = *yt + *xt * w;
      	xt += m;
      	yt += n;
    }
}

/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	/*   Y = weight_out_delay_time(X,W,T,tau);  */
	double *x,*y,*w,*dy;
	int  m,M,n,N,t,T,Tx,D,j,k,NX,NY,Nr,tau,Nall,ndims;
	int  wid,tid,yid,xid,xst,yst,wst;
    int  *dims;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=4) 
	  mexErrMsgTxt("4 inputs required.");
	if(nlhs!=1) 
	  mexErrMsgTxt("One output required.");
	
	/* x = mxGetPr(prhs[0]) Create a pointer to the inputs*/
	/* mxGetNumberOfElements: 配列の要素数 */
    /* mxGetNumberOfDimensions: 配列の次元数 */
	/* mxGetDimensions: 各次元内の要素数 */

	/*   Y = weight_out_delay_time(X,W,T,tau);  */
	x  = mxGetPr(prhs[0]);
	w  = mxGetPr(prhs[1]);
	T  = (int) mxGetScalar(prhs[2]);
	tau = (int) mxGetScalar(prhs[3]);
	
	/*  Get the dimensions of the matrix W 
	Y : N x T  x Nr
	X : M x Tx x Nr
	W : N x (M*D)
	*/

	ndims = mxGetNumberOfDimensions(prhs[0]);
	dims  = mxGetDimensions(prhs[0]);

	M  = dims[0];
	Tx = dims[1];
	if( ndims==3 ){
		Nr=dims[2];
	}else{
		Nr=1;
	}

	N = mxGetM(prhs[1]);
	D = mxGetN(prhs[1])/M;
	NY = N*T;
	NX = M*Tx;
	
	dims[0] = N;
	dims[1] = T;
	
	/*  Set the output pointer to the output matrix. */
	plhs[0] = mxCreateNumericArray(ndims,dims,mxDOUBLE_CLASS,mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	y = mxGetPr(plhs[0]);
	
/* 
	Y = W * X
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

for (j=0;j<D;j++){
	for (m=0;m<M;m++){
		for (n = 0;n<N;n++){
			wst = wid + n;

			yid = 0;
			xid = m;
			
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;
				
	 			y_wx(x + xst, *(w + wst), y + yst, M, N, T);
				/* Y(n,yid:yend) = Y(n,yid:yend) + W(n,wid) * X(xid:M:xend);*/
				
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
