function [Model, Info] = linear_sparse_seq(X,Y,Model,parm)
%  Estimate linear weight matrix for input-output mapping
%     Automatic Relevance Prior for each input dimension
%     is imposed to get sparse weight matrix
%   Sequential iterative algorithm to increase parameters one by one
%
% Notice !!
%     Delay embedding is done in this module,
%     then input vector should be original input data without embedding
%
%   [Model, Info] = linear_sparse_seq(X,Y,Model,parm)
%
% --- Input
%  Y  : Output data ( N x T x Ntrial )
%  X  : Input data  ( M x (T + (Dtau-1)*Tau)  x Ntrial)
%  N  =  # of output
%  M  =  # of input (original input space dimension)
%  T  =  # of time sample
%
%  Estimate the following model
%    Y(t) = W * [X(:, t + (Dtau-1)*Tau ); ...; X(:, t)]
%
%  Model : Structure for estimated model
%  if Model is empty, initialization is done before training
%  if Model is previous training result, re-training us done
%
%  parm  : Structure for learning parameter
%  parm.Npre_train :  # of VB-update in initial training
%  parm.Ntrain :  # of training
%  parm.Nskip  :  skip # for print
%  parm.a_min  :  Min value for pruning small variance component
%  parm.Prune  :  = 1 : Prune small variance & irrelevant input dimension
%
%  parm.Tau       = Lag time
%  parm.Dtau      = Number of embedding dimension
%    Total input dimension in embedding space is (M * parm.Dtau)
% --- Output
%  Model : Structure for estimated model
%  Model.SY  :  Noise variance         ( 1 x 1 )
%  Model.W   :  Weight matrix          ( N x M*D ) , D = parm.Dtau
%  Model.A   :  Prior weight variance  ( 1 x M*D ) 
%  Model.ix_act : Active index for W after pruning
%
%  Info  : Structure for learning process history
%  Info.FE  = LP + H : Free energy
%  Info.LP  = Log likelihood
%  Info.H   = - Model entropy
%
% 2009-10-18 Made by M. Sato

global a_min;

% Constants
MINVAL = 1.0e-15;

% Dimension
[N ,T  ,Ntrial]  = size(Y); % N = # of output
[M ,Tx ,Ntrialx] = size(X); % M = # of input without embedding

Tall = T * Ntrial;

if Ntrial~=Ntrialx, error('# of trial is different in X and Y'); end;

% # of embedding dimension
if isfield(parm,'Dtau')
	D    = parm.Dtau; 
	tau  = parm.Tau;
else
	D    = 1;
	tau  = 1;
end

% # of total training iteration
Ntrain = parm.Ntrain;

Nskip  = 100;   % skip steps for display info
a_min  = 1e-10; % Minimum value for weight pruning
Fdiff  = 1e-10; % Threshold for convergence
Ncheck = 100;   % Minimum number of training iteration
Fstep  = 5;     % Free energy convergence check step
Prune  = 1;     % Prune mode
Nupdate = 100;

if isfield(parm,'Nskip'), Nskip  = parm.Nskip; end;
if isfield(parm,'Fdiff'), Fdiff   = parm.Fdiff; end;
if isfield(parm,'a_min'), a_min   = parm.a_min ; end;
if isfield(parm,'Prune'), Prune = parm.Prune; end;
if isfield(parm,'Ncheck'), Ncheck = parm.Ncheck; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end
if isfield(parm,'Nupdate'),	Nupdate = parm.Nupdate;end

Xdim    = M;
M_all   = M*D;			% Input dimension

fprintf('Sequential linear sparse regression start\n')
fprintf('--- Output Dimension  = %d\n',N)
fprintf('--- Input  Dimension  = %d\n',M)
fprintf('--- Embedding  Dimension  = %d\n',D)
fprintf('--- Number of trials  = %d\n',Ntrial)
fprintf('--- Number of training sample = %d\n',T)
fprintf('--- Total update iteration    = %d\n',Ntrain)

% Free energy histry
FE  = zeros(Ntrain,1);
LP  = zeros(Ntrain,1);
H   = zeros(Ntrain,1);
Err = zeros(Ntrain,1);

% Incremental Free energy histry
dF  = zeros(Ntrain,1);% Free energy increase at each step
dID = zeros(Ntrain,1);% update/delete/add ID
dIX = zeros(Ntrain,1);% component index for update/delete/add
dIM = zeros(Ntrain,1);% Number of component 

% Free energy increase list for each component
DFup  = zeros(M_all,1);% Free energy increase for update
DFdel = zeros(M_all,1);% Free energy increase for delete
DFadd = zeros(M_all,1);% Free energy increase for add

%  
% --- Initialization
%  

% XX = X(ix_act,:)*X'  [M_act x M_all]
% YX = Y * X'          [N x M_all]
% xx = sum(X.^2,2)'    [1 x M_all]
% YY = sum(Y.^2,2)     [N x 1]

% Input variance
% Input covariance (Spatial dimension)
% xx = diag(XX)'	% 1 x M_all
xx = zeros(1,M_all);

Tdelay = tau*(D-1);
T  = Tx - Tdelay;
t  = (1:T) + Tdelay ;
ix = 1:M;

for j=1:D
	xx(ix) = sum(reshape(X(:,t,:), [M, T*Ntrial] ).^2 ,2)';
	
	ix = ix + M;
	t  = t - tau;
end
% xx = (reshape(X, [M, Tx*Ntrial]) 
%     * reshape(X, [M, Tx*Ntrial])') 

% XX, YX, YY, dYX are not normalized
YX = error_corr_delay(X,Y,D,tau);
YY = sum(Y(:).^2);    % N x 1

SY0 = YY/(Tall*N);

if ~isfield(Model, 'ix_act')
	%
	% --- Initialization: Find input component with maximum correlation
	% 
	[SY,ix_act] = compo_init_index(xx,YX,YY,Tall);

	% XX correlation matrix = X(ix_act,:)*X'
	XX = embed_cor_cov(X,D,tau,ix_act);
	
	% Set initial alpha
	[A,W,SW,S,Q] = compo_init_alpha(XX,YX,xx,ix_act);
else
	%
	% --- Initialization by previous training result
	% 
	fprintf('Old result is used as initial value\n')
	fprintf('Old method = %s\n', Model.method)

	ix_act = Model.ix_act;
	Ainv   = Model.A;
	
	if isfield(Model, 'mode') &  strcmp(Model.mode,'scalar')==1,
		%A  = (1./SY)' * (W.^2) + 1./(T*diag(XX)');	% 1 x M
		Ainv  = sum(Ainv ,1)./(sum(Model.SY));
	end

	A  = zeros(1,M_all);
	A(ix_act)  = 1./Ainv ;	% 1 x M
	
	% XX correlation matrix = X(ix_act,:)*X'
	XX = embed_cor_cov(X,D,tau,ix_act);

	if isfield(Model, 'S')
		SY = 1./Model.SY;	% 1 x 1
		W  = Model.W ;	% N x M
		SW = Model.SW;	% M x M
		S  = Model.S;	% 1 x M_all
		Q  = Model.Q;	% 1 x M_all
	else
	    % Residual Error & correlation
		[W, SW]   = initial_weight_estimation(XX,YX,A,ix_act);
		
		[ix_dim,ix_tau,jx_act] = embed_trans_index(ix_act,Xdim,D,1);
		W0  = zeros(N, length(ix_dim)*D);
		W0(:,jx_act) = W;
		
		dY  = error_delay_time(X(ix_dim,:,:),Y,W0,tau); 
		dYX = error_corr_delay(X,dY,D,tau);
		dYY = sum(sum(dY.^2, 2),3);  % N x 1
		
		[SY,Q,S]  = component_statics(xx,XX,dYY,dYX,W,SW,A,ix_act,Tall);
	end

end

fprintf('a_min = %g\n', a_min)
fprintf('SY0 = %g\n', SY0)
fprintf('SY  = %g\n', SY)

M  = length(ix_act);
fprintf('Initial dim  = %d\n',M)

ix_non = find( A == 0 );

k_save = 0;
k_up   = 0;
k_pr   = 0;

%%%%%% Learning Loop %%%%%%
for k=1:Ntrain
	%
	% --- Find component that gives maximal Free energy increase 
	%
	[dFup,ix_up,dFdel,ix_del,ix_bad,dmax] = ...
		fincrease_update_act(A,S,Q,SY,ix_act);
	
	[dFadd,ix_add] = fincrease_add_act(S,Q,SY,ix_non);
	
	% Free energy increase list for all component
	DFup  = zeros(M_all,1);
	DFdel = zeros(M_all,1);
	DFadd = zeros(M_all,1);
	DFup(ix_up)   = dFup;
	DFdel(ix_del) = dFdel;
	DFadd(ix_add) = dFadd;

	if isempty(ix_bad),
		% Find max increase of each case
		if ~isempty(ix_up)
			[Fup,i_up] = max(dFup);
			i_up = ix_up(i_up(1));
		else
			Fup  = -realmax;
			i_up = [];
		end
		if ~isempty(ix_del)
			[Fdel,i_del] = max(dFdel);
			i_del = ix_del(i_del(1));
		else
			Fdel  = -realmax;
			i_del = [];
		end
			
		if ~isempty(ix_add)
			[Fadd,i_add] = max(dFadd);
			i_add = ix_add(i_add(1));
		else
			Fadd  = -realmax;
			i_add = [];
		end
		
		% Find max increase
		[dFmax, id] = max([Fup, Fdel, Fadd]);
		
		dF(k)  = dFmax;
		dID(k) = id;
		
		% Check convergence
		if isempty(ix_add) & isempty(ix_del) & isempty(ix_up),
			% No change in alpha
			fprintf('Converged: Max alpha change = %g\n',dmax)
			break;
		end
		
		%
		% --- Model change & parameter update
		%
		switch	id
		case	1
			% update index within 'ix_act'
			jx = trans_index(i_up,ix_act,M_all);
			dIX(k) = i_up;
			
			% Update component
			[A,W,SW,S,Q,ix_act] = ...
				compo_update_act(XX,A,W,SW,S,Q,SY,i_up,jx,ix_act);
		case	2
			% deletion index within 'ix_act'
			jx = trans_index(i_del,ix_act,M_all);
			dIX(k) = i_del;
			
			% Delete component
			[A,W,SW,S,Q,ix_act] = ...
				compo_delete_act(XX,A,W,SW,S,Q,SY,i_del,jx,ix_act);
			% Delete component from XX correlation matrix
			M  = size(XX,1);
			XX = XX([1:(jx-1), (jx+1):M],:);
		case	3
			% Add component
			Xadd = embed_cor_cov(X,D,tau,i_add);
			
			[A,W,SW,S,Q,ix_tmp] = ...
				compo_add_act(XX,Xadd,A,W,SW,S,Q,SY,i_add,ix_act);
			dIX(k) = i_add;
			
			% change order of 'ix_tmp' to 'ix_act'
			ix_act = find( A > 0 );
			jx = trans_index(ix_act, ix_tmp, M_all);
			SW = SW(jx,jx);
			W  = W(:,jx);
	
			% Add component to XX correlation matrix
			XX = [XX; Xadd];
			XX = XX(jx,:);
		end
	else
		% There are bad component
		id = 4;
		dIX(k) = ix_bad(1);
		dID(k) = id;
		dF(k)  = 0;
	end

	% Active & Non-active index
	ix_non = find( A == 0 );
	ix_act = find( A > 0  );
    M      = length(ix_act);
    dIM(k) = M;

	k_up = k_up + 1;
	k_pr = k_pr + 1;
	
	if k_up >= min( M-1, Nupdate ) | id == 4 | any(diag(SW) < eps),
		k_up = 0;
		k_save = k_save + 1;

		% Pruning
		Amin = min( A( A>0 ) );
		ix_tmp = ix_act;
		ix_act = find( A > 0  & (A/Amin) <  1/a_min);
		AA = A(ix_act);
		A  = zeros(1,M_all);
		A(ix_act) = AA;
		ix_non = find( A == 0);

		jx = trans_index(ix_act, ix_tmp, M_all);
		XX = XX(jx,:);

		% Update weight 'W' & covariance matrix 'SW' 
		SW = pinv( diag(AA) + XX(:,ix_act) );
		W  = YX(:,ix_act) * SW;

	    % Residual Error & correlation
		[ix_dim,ix_tau,jx_act] = embed_trans_index(ix_act,Xdim,D,1);
		
		W0  = zeros(N, length(ix_dim)*D);
		W0(:,jx_act) = W;
		
		dY  = error_delay_time(X(ix_dim,:,:),Y,W0,tau); 
		dYX = error_corr_delay(X,dY,D,tau);
		dYY = sum(sum(dY.^2, 2),3);  % N x 1

		% Noise variance update
		SY  = (N*Tall)./sum( dYY + W.^2 * AA' );
		
	    % Update residual correlation
		Q   = dYX;
		S   = abs(xx - sum(XX.*(SW*XX),1));
		
		% Free energy per data : Fenergy/(N*T)
		LP(k_save)  = log(SY);
		H(k_save)   = ( log_det(SW) + sum(log(AA)) )/Tall;
		FE(k_save)  = LP(k_save) + H(k_save);
		Err(k_save) = sum(dYY)/(SY0*N*Tall);
		
		if k_save > Ncheck,
			Fdif = (FE(k_save) - FE(k_save-Fstep))/(abs(FE(k_save))+eps);
		else
			Fdif = Fdiff + 1;
		end
		
		if (Fdiff > abs(Fdif)), 
			fprintf('Converged : Free energy change = %g\n',Fdif)
			break; 
		end;
		
		if k_pr >= Nskip
	        fprintf('Iter = %4d, M = %4d, err = %g, SY = %g, F = %g\n', ...
	                 k, M, Err(k_save), 1/SY, FE(k_save));
			k_pr = 0;
		end
		
	end

end

% Active index
Model.ix_act = ix_act;
Model.M_all  = M_all ;

% Save trained variable
Model.A  = 1./A(ix_act) ;	% 1 x M
Model.SY = 1./SY;	% 1 x 1

Model.W  = W ;	% N x M
Model.SW = SW;	% M x M
Model.S  = S;	% 1 x M_all
Model.Q  = Q;	% 1 x M_all
Model.DFup  = DFup ;
Model.DFdel = DFdel;
Model.DFadd = DFadd;

Model.method = 'linear_sparse_seq';
Model.mode   = 'cov';
Model.sparse = 'sparse';

% Save history
Info.dF  = dF(1:k);
Info.dID = dID(1:k);
Info.dIX = dIX(1:k);
Info.dIM = dIM(1:k);

Info.FE  = FE(1:k_save);
Info.LP  = LP(1:k_save);
Info.H   = H(1:k_save) ;
Info.Err = Err(1:k_save);

return

function	[W,SW] = initial_weight_estimation(XX,YX,A,ix_act)
% Update weight 'W' & covariance matrix 'SW' 

% XX correlation matrix = X(ix_act,:)*X'
AA = A(ix_act);
SW = pinv( diag(AA) + XX(:,ix_act) );
W  = YX(:,ix_act) * SW;

function	[SY,Q,S] = component_statics(xx,XX,dYY,dYX,W,SW,A,ix_act,Tall)

N = length(dYY);

% Noise variance update
AA = A(ix_act);
SY = (N*Tall)./sum( dYY + W.^2 * AA' );

% Update residual correlation
Q   = dYX;
S   = abs(xx - sum(XX.*(SW*XX),1));

return

%%% ---- Index transformation from old active_index to current active_index
function	jx = trans_index(ix,ix_old,M)

N = length(ix_old);
Itrans = zeros(M,1);
Itrans(ix_old) = 1:N;

jx = Itrans(ix);
