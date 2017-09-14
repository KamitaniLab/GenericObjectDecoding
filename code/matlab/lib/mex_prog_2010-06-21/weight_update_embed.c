#include "mex.h"
#include "mexutil.h"

/*
   stepwise weight update in time embeded space
   Wout = weight_update_embed(X,dY,W,XX,A,tau)
   
	  Error function
	    E = (Y - W*X)^2 + W*A*W
	   dW(n,m) = (dYX(n,m) - W(n,m)*A(m))./ (XX(m) + A(m));
	   A should have the dimension of X without embedding

 * Written by Masa-aki Sato 2009-11-1
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
	/*   Wout = weight_update_embed(X,dY,W,XX,A,tau)  */
	double *x,*w,*dy,*xx,*a,*Wout,*dyx,dw;
	int  m,M,n,N,T,Tx,D,j,k,NX,NY,NW,Nr,tau,Nall,ndims;
	int  wid,tid,yid,xid,xxi,xst,yst,wst;
    int  *dims;
	mxArray *dyx_ptr;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=6) 
	  mexErrMsgTxt("4 inputs required.");
	if(nlhs!=1) 
	  mexErrMsgTxt("One output required.");
	
	/* x = mxGetPr(prhs[0]) Create a pointer to the inputs*/
	/* mxGetNumberOfElements: 配列の要素数 */
    /* mxGetNumberOfDimensions: 配列の次元数 */
	/* mxGetDimensions: 各次元内の要素数 */

	/*  Error function
	    E = (Y - W*X)^2 + W*A*W
	*/
	/*   Wout = weight_update_embed(X,dY,W,XX,A,tau)  */
	x  = mxGetPr(prhs[0]);
	dy = mxGetPr(prhs[1]);
	w  = mxGetPr(prhs[2]);
	xx = mxGetPr(prhs[3]);
	a  = mxGetPr(prhs[4]);
	
	/*  time delay value */
	tau = (int) mxGetScalar(prhs[5]);
	
	/*  Get the dimensions of the matrix 
	X  : M x Tx x Nr
	XX : M x 1
	Y  : N x T  x Nr
	dY : N x T  x Nr
	W  : N x (M*D)
	A  : 1 x (M)
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
xxi = 0;
tid = tau*(D-1);

/* Loop for time delay j < D */
for (j=0;j<D;j++){
	/* Loop for X-dim m < M  */
	for (m=0;m<M;m++){
		/* Loop for Y-dim n < N  */
		for (n = 0;n<N;n++){
			/* Weight upgate for W(n,m,j)
			   Y(n,t) = sum W(n,m,j) * X(m,t - j*tau)
			*/
			wst = wid + n;

			yid = 0;
			xid = m;
			*dyx = 0;
			
			/* Trial loop k < Nr */
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;
				
				/* error correlation
				   dYX(n,m) = dY(n,:) * X(m,:)';  
				*/
	 			dyx_corr(x + xst, dy + yst, dyx, M, N, T);
				
				xid += NX;
				yid += NY;
				/* END of k < Nr (Trial) */
			}
			
			/*  Weight update
			  - (Y - sum_k (W_o(k) * X(k)) - W(m)*X(m) )*X(m) + W(m)*A(m) = 0
			    W(m)*(XX(m) + A(m)) = (Y - W_o * X)*X(m) + W_o(m)*XX(m)
			    W(m) = W_o(m) + (dYX(m) - W_o(m)*A(m))./ (XX(m) + A(m))
			   dW(n,m) = (dYX(n,m) - W(n,m)*A(m))./ (XX(m) + A(m));
			*/
			dw = (*dyx - w[wst] * a[xxi])/(xx[xxi] + a[xxi]);
			Wout[wst]  = w[wst] + dw;
			
			/* Update error dY  */
			yid = 0;
			xid = m;
			
			/* Trial loop k < Nr */
			for (k = 0;k<Nr;k++){
				xst = xid + tid*M ;
				yst = yid + n;

				/*  error update
				    dY(n,t) = dY(n,t) - dW(n) * X(m,t);
				*/
	 			dy_err(x + xst, dw, dy + yst, M, N, T);
				
				xid += NX;
				yid += NY;
				/* END of k < Nr (Trial) */
			}
			/* END of n < N (Y dim) */
		}
		wid += N;
		xxi++;
		/* END of m < M (X dim) */
	}
	tid += -tau;
	/* END of j < D */
}

/* END */
}
