#include "mex.h"
#include "mexutil.h"

/*
   stepwise weight update in time embeded space
	Wout = weight_update_embed(X,dY,W,XX,A,tau,Ntr) 
   
   Data have variable trial length
   Sample number of n-th trial is given by Ntr(n)
   X  : Xdim x Time x trial 
   dY : Ydim x Time x trial 
   
	  Error function
       E = (Y - W*X)^2 + W*A*W
	   dW(n,m) = (dYX(n,m) - W(n,m)*A(m))./ (XX(m) + A(m));
	   A should have the dimension of X without embedding

 * Written by Masa-aki Sato 2009-11-5
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
	/*   Wout = weight_update_embed(X,dY,W,XX,A,tau,Ntr)  */
	double *x,*w,*dy,*xx,*a,*Wout,*dyx,dw,*Ntr;
	int  m,M,n,N,T,Tx,D,j,k,NX,NY,NW,Nr,tau,Nall,ndims;
	int  wid,tid,yid,xid,aid,xst,yst,wst;
    int  *dims;
	mxArray *dyx_ptr;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=7) 
	  mexErrMsgTxt("7 inputs required.");
	if(nlhs!=1) 
	  mexErrMsgTxt("One output required.");
	
	/* x = mxGetPr(prhs[0]) Create a pointer to the inputs*/
	/* mxGetNumberOfElements: 配列の要素数 */
    /* mxGetNumberOfDimensions: 配列の次元数 */
	/* mxGetDimensions: 各次元内の要素数 */

	/*   Wout = weight_update_embed(X,dY,W,XX,A,tau,Ntr)  */
	x  = mxGetPr(prhs[0]);
	dy = mxGetPr(prhs[1]);
	w  = mxGetPr(prhs[2]);
	xx = mxGetPr(prhs[3]);
	a  = mxGetPr(prhs[4]);
	tau = (int) mxGetScalar(prhs[5]);
	Ntr = mxGetPr(prhs[6]);
	
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

	NW = mxGetN(prhs[2]);
	D  = NW/M;
	NY = N*T;
	NX = M*Tx;
	
	/*  Set the output pointer to the output matrix. */
	plhs[0] = mxCreateDoubleMatrix(N,NW,mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	Wout = mxGetPr(plhs[0]);

	dyx_ptr = mxCreateDoubleMatrix(1,1,mxREAL);
	dyx = mxGetPr(dyx_ptr);
	
/* 
	dY = Y - W * X
	Y : N x T  x Nr
	X : M x Tx x Nr
	W : N x (M*D)
	Xdelay(t) = [X(t+tau*(D-1)); ... ; X(t+tau); X(t)]
*/
/*			
printf("N= %d , M=%d , T=%d , D=%d , Tx=%d ,tau=%d\n", N,M,T,D,Tx,tau); 
*/

wid = 0;
aid = 0;
tid = tau*(D-1);

for (j=0;j<D;j++){
	for (m=0;m<M;m++){
		for (n = 0;n<N;n++){
			wst = wid + n;

			yid = 0;
			xid = m;
			*dyx = 0;
			
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;
				
				/* error correlation
				   dYX = dY(n,:) * X(m,:)';  
				*/
	 			dyx_corr(x + xst, dy + yst, dyx, M, N, *(Ntr+k));
				
				xid += NX;
				yid += NY;
				/* END of k < Nr (Trial) */
			}
			
			/*  Weight update
			    dW(n)  = (dYX(n) - W(n,m)*A(m))./ (XX(m) + A(m));
			*/
			dw = (*dyx - w[wst] * a[aid])/(xx[aid] + a[aid]);
			Wout[wst] = w[wst] + dw;
/*			
printf("W= [%e , %e , %e] (n=%d,m=%d,d=%d) \n", Wout[wst],*dyx,dw,n,m,tid); 
*/

			yid = 0;
			xid = m;
			
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;

				/*  error update
				    dY(n,t) = dY(n,t) - dW(n) * X(m,t);
				*/
	 			dy_err(x + xst, dw, dy + yst, M, N, *(Ntr+k));
				
				xid += NX;
				yid += NY;
				/* END of k < Nr (Trial) */
			}
			/* END of n < N (Y dim) */
		}
		wid += N;
		aid++;
		/* END of m < M (X dim) */
	}
	tid += -tau;
	/* END of j < D */
}

/* END */
}
