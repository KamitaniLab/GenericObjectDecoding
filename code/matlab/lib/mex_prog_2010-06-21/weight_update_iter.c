#include "mex.h"
#include "mexutil.h"

/*
   component-wise weight update with component-wise precision parameter
                No time embedding is done
   Wout = weight_update_iter(X,dY,W,XX,A,SY) 

	  Error function
	  E = (Y - W*X)^2/SY + A*W^2
	dW(n,m) = (dYX(n,m) - W(n,m)*A(n,m)*S(m))./ (XX(m) + A(n,m)*S(m));
	   A has the same size as W

   Written by Masa-aki Sato 2009-11-1
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
	/*  Error function
	    E = (Y - W*X)^2/SY + A*W^2
	*/
	/*   Wout = weight_update_iter(X,dY,W,XX,A,SY)  */
	double *x,*w,*dy,*xx,*sy,*a,*Wout,*dyx,dw;
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
	    E = (Y - W*X)^2/SY + A*W^2
	*/
	/*   Wout = weight_update_embed(X,dY,W,XX,A,SY)  */
	x  = mxGetPr(prhs[0]);
	dy = mxGetPr(prhs[1]);
	w  = mxGetPr(prhs[2]);
	xx = mxGetPr(prhs[3]);
	a  = mxGetPr(prhs[4]);
	sy = mxGetPr(prhs[5]);
	
	/*  Get the dimensions of the matrix 
	X  : M x T x Nr
	XX : M x 1
	Y  : N x T  x Nr
	dY : N x T  x Nr
	W  : N x (M)
	A  : N x (M)
	SY : N x 1
	*/

	dims = mxGetDimensions(prhs[0]);
	M  = dims[0];
	Tx = dims[1];
	
	Nall  = mxGetNumberOfElements(prhs[1]);
	ndims = mxGetNumberOfDimensions(prhs[1]);
	dims  = mxGetDimensions(prhs[1]);

	N = dims[0];
	T = dims[1];
	if(T!=Tx) 
	  mexErrMsgTxt("Time length mismatch of X and Y .");
	
	if( ndims==3 ){
		Nr=dims[2];
	}else{
		Nr=1;
	}

	NW = mxGetN(prhs[2]);
	if(NW!=M) 
	  mexErrMsgTxt("mismatch of X and W .");
	  
	NY = N*T;
	NX = M*T;
	
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

/* Loop for X-dim m < M  */
for (m=0;m<M;m++){
	/* Loop for Y-dim n < N  */
	for (n = 0;n<N;n++){
		/* Weight upgate for W(n,m,j)
		   Y(n,t) = sum W(n,m,j) * X(m,t - j*tau)
		*/
		wst = wid + n; /* wst : (n,m) , wid : (0,m) */

		yid = n; /* yid : (n,0,k) */
		xid = m; /* xid : (m,0,k) */
		*dyx = 0;
		
		/* Trial loop k < Nr */
		for (k = 0;k<Nr;k++){
			
			/* error correlation
			   dYX(n,m) = dY(n,:) * X(m,:)';  
			*/
 			dyx_corr(x + xid, dy + yid, dyx, M, N, T);
			
			xid += NX;
			yid += NY;
			/* END of k < Nr (Trial) */
		}
		
		/*  Weight update
		- (Y - sum_k (W_o(k) * X(k)) - W(m)*X(m) )*X(m) + W(m)*A(m)*S(m) = 0
		 W(m)*(XX(m) + A(m)*S(m)) = (Y - W_o * X)*X(m) + W_o(m)*XX(m)
		 W(m) = W_o(m) + (dYX(m) - W_o(m)*A(m))./ (XX(m) + A(m))
		dW(n,m) = (dYX(n,m) - W(n,m)*A(n,m)*S(m))./ (XX(m) + A(n,m)*S(m));
		  wst : (n,m) , 
		*/
		dw = (*dyx - w[wst] * a[wst] * sy[n])/(xx[m] + a[wst] * sy[n]);
		Wout[wst]  = w[wst] + dw;
		
		/* Update error dY  */
		yid = n; /* yid : (n,0,k) */
		xid = m; /* xid : (m,0,k) */
		
		/* Trial loop k < Nr */
		for (k = 0;k<Nr;k++){

			/*  error update
			    dY(n,t) = dY(n,t) - dW(n) * X(m,t);
			*/
 			dy_err(x + xid, dw, dy + yid, M, N, T);
			
			xid += NX;
			yid += NY;
			/* END of k < Nr (Trial) */
		}
		/* END of n < N (Y dim) */
	}
	wid += N;
	/* END of m < M (X dim) */
}

/* END */
}
