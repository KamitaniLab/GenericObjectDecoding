function [Model, Info] = linear_stepwise_reg(X,Y,Model,parm)
%  Estimate linear weight matrix for input-output mapping
%     regularization version
%
% Notice !!
%     Delay embedding is done in this module,
%     then input vector should be original input data without embedding
%
%   [Model, Info] = linear_stepwise_reg(X,Y,Model,parm)
%     Scalar iteration version
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

MINVAL = 1.0e-15;

fprintf('linear_stepwise (regularization) start\n')

% Dimension
[N ,T  ,Ntrial]  = size(Y); % N = # of output
[M ,Tx ,Ntrialx] = size(X); % M = # of input without embedding

Tall = T * Ntrial;

if Ntrial~=Ntrialx, error('# of trial is different in X and Y'); end;

Nskip  = 100;   % skip steps for display info
a_min = 1e-14;	% Minimum value for weight pruning
Fdiff = 1e-12; % Threshold for convergence
Ncheck = 100;   % Minimum number of training iteration
Fstep  = 5;     % Free energy convergence check step

if isfield(parm,'Nskip'), Nskip  = parm.Nskip; end;
if isfield(parm,'Fdiff'), Fdiff   = parm.Fdiff; end;
if isfield(parm,'a_min'), a_min   = parm.a_min ; end;
if isfield(parm,'Ncheck'), Ncheck = parm.Ncheck; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end

% # of embedding dimension
if isfield(parm,'Dtau')
	D    = parm.Dtau; 
	tau  = parm.Tau;
else
	D    = 1;
	tau  = 1;
end
% Parameters
Ntrain = parm.Ntrain;

if Ntrain < 1, Info = []; return; end;

fprintf('--- Output Dimension  = %d\n',N)
fprintf('--- Input  Dimension  = %d\n',M)
fprintf('--- Embedding  Dimension  = %d\n',D)
fprintf('--- Number of trials  = %d\n',Ntrial)
fprintf('--- Number of training sample = %d\n',T)
fprintf('--- Total update iteration    = %d \n',Ntrain)

% original input dimension
Xdim  = M;
M = Xdim*D;
M_ALL = M;

%  
% --- Initialization
%

% Input/Output variance
sx = mean((X(:) - mean(X(:))).^2);
sy = mean((Y(:) - mean(Y(:))).^2);

SY0 = mean(sy);
A0  = mean(sx);

% Input covariance
XX  = sum(sum(X.^2,3),2)/(Tx*Ntrial);
XX  = repmat(XX', [1 D]);% 1 x M

if isfield(Model,'A')
	A = Model.A;
	A = mean(A);
else
	A = A0;
	W = zeros(N,M_ALL);
end

if size(A,1)==Xdim && size(A,2)==1
	A   = repmat(A',[1 D]);
elseif size(A,1)==1 && size(A,2)==Xdim
	A   = repmat(A,[1 D]);
elseif size(A,1)==1 && size(A,2)==1
	A   = repmat(A,[1 M_ALL]);
end

W = zeros(N,M_ALL);
SY  = SY0;

% Save original input for pruning
IX_act  = 1:M_ALL;
IX_dim  = 1:Xdim;
ix_act  = 1:M;
ix_dim  = 1:Xdim;
M_all   = M;

fprintf('SY0 = %g\n', SY0)
fprintf('SY  = %g\n', SY)

% Free energy histry
FE  = zeros(Ntrain,1);
LP  = zeros(Ntrain,1);
H   = zeros(Ntrain,1);
Err = zeros(Ntrain,1);
Mhist = zeros(Ntrain,1);

[dY] = error_delay_time(X,Y,W,tau);

k_save  = 0;
% Ainv = alpha , A = 1/alpha
Ainv    = A;	

%%%%%% Learning Loop %%%%%%
for k=1:Ntrain
	
	% E = (Y-W*X)^2/SY +  W^2/A
	%   = ( (Y-W*X)^2 +  W^2 * (SY/A) )/SY
    [W]  = weight_update_embed(X, dY, W, Tall*XX, Ainv, tau);
	[dY] = error_delay_time(X, Y, W, tau);

    dYY = sum(dY(:).^2)/(N*Tall); 
	WW  = sum(W.^2,1);
    
    % Noise variance update
    SY  = dYY + sum(WW .* Ainv)/(N*Tall);
    % Prevent zero variance
    SY  = max( SY, MINVAL);

    % Log variance
    log_sy  = N*( log(SY) );
	
    % Free energy
    LP(k)  = - 0.5 * log_sy ;
    FE(k)  = LP(k) ;
    Err(k) = sum(dYY)/(SY0);

    if mod(k, Nskip)==0
        fprintf('Iter = %4d, err = %g, WW=%g, \n', ...
               k, Err(k), mean(WW));
    end
    
    % Convergence check
	if k > Ncheck,
		Fdif = (FE(k) - FE(k-Fstep))/(abs(FE(k))+eps);
	else
		Fdif = Fdiff + 1;
	end
	
	if (Fdiff > abs(Fdif)), 
		fprintf('Converged : Free energy change = %g\n',Fdif)
		break; 
	end;
end

ix_act = IX_act(ix_act);

% Active index
%Model.ix_act = ix_act;
Model.M_all  = M_ALL ;

% Save output variable
Model.A    = A ;
Model.W    = W ;
Model.SY   = SY;
Model.method = 'linear_stepwise';
Model.mode   = 'scalar';
Model.sparse = 'reg';

% Save history
Info.FE  = FE(1:k);
Info.LP  = LP(1:k);
Info.H   = H(1:k) ;
Info.Err = Err(1:k);

