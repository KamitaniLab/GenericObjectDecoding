function [Model, Info] = linear_fit(X,Y,Model,parm)
%  Estimate linear weight matrix for input-output mapping
%
%   [Model, Info] = linear_fit(X,Y,Model,parm)
%
% --- Input
%
%  X  : Input data  ( M x T )
%  Y  : Output data ( N x T )
%  N  =  # of output
%  M  =  # of input
%  T  =  # of data
%
%  parm  : Structure for learning parameter
%  parm.Ntrain :  # of training
%  parm.Nskip  :  skip # for print
%
%  Model : Structure for estimated model
%
% --- Output
%  Model : Structure for estimated model
%  Model.SY  :  Noise variance         ( N x 1 )
%  Model.W   :  Weight matrix          ( N x M )
%
%  Info  : Structure for learning process history
%  Info.FE  = LP : Free energy
%  Info.LP  = - (Log error)
%
% 2008-5-18 Made by M. Sato

MAXVAL = 1.0e+15;
MINVAL = 1.0e-15;

% Dimension
% Dimension
[N ,T  ,Ntrial]  = size(Y); % N = # of output
[M ,Tx ,Ntrialx] = size(X); % M = # of input without embedding

if Ntrial~=Ntrialx, error('# of trial is different in X and Y'); end;

Tall = T * Ntrial;

% # of embedding dimension
if isfield(parm,'Dtau')
	D    = parm.Dtau; 
	tau  = parm.Tau;
else
	D    = 1;
	tau  = 1;
end

% original input dimension
Xdim  = M;
M = Xdim*D;

% Parameters
Ntrain = parm.Ntrain;

Nskip  = 100;   % skip steps for display info
a_min  = 1e-10; % Minimum value for weight pruning
Fdiff  = 1e-10; % Threshold for convergence
Ncheck = 100;   % Minimum number of training iteration
Fstep  = 5;     % Free energy convergence check step

if isfield(parm,'Nskip'), Nskip  = parm.Nskip; end;
if isfield(parm,'Fdiff'), Fdiff   = parm.Fdiff; end;
if isfield(parm,'a_min'), a_min   = parm.a_min ; end;
if isfield(parm,'Ncheck'), Ncheck = parm.Ncheck; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end

if Ntrain < 1, Info = []; return; end;

fprintf('linear fit start\n')
fprintf('--- Total update iteration      = %d\n',Ntrain)

% Histry
FE    = zeros(Ntrain,1);
LP    = zeros(Ntrain,1);

%  
% --- Initialization
%  
% Input/Output variance
sx = mean((X(:) - mean(X(:))).^2);
sy = mean((Y(:) - mean(Y(:))).^2);

SY0 = mean(sy);
A0  = 1./mean(sx);
SY  = SY0;  % N x 1
rx  = 1/(M+1);

W   = zeros(N,M);

% Working variable
% Input covariance
XX  = sum(sum(X.^2,3),2)/(Tx*Ntrial);
XX  = repmat(XX', [1 D]);

fprintf('XXmin = %g\n', min(XX))
fprintf('XXmax = %g\n', max(XX))
fprintf('SY0 = %g\n', SY0)
fprintf('SY  = %g\n', SY)

%%%%%% Learning Loop %%%%%%
for k=1:Ntrain
%    dY  = Y - W * X
%    dYX = (dY * X')/T;
	dY  = error_delay_time(X, Y, W, tau);
    dYX = error_corr_delay(X,dY,D,tau);
    
    dYY = sum(sum(dY.^2,2),3)/(Tall); 
    dYX = dYX/(Tall);   
    
    SY  = dYY;
    SY  = max( SY, MINVAL);

    % Weight update
    dW  = rx .* repmultiply(dYX , 1./ XX) ;    % ( N x M )
    W   = W + dW;

    % Log variance
    log_sy  = sum( log(SY) );

    % Free energy
    LP(k) = sum(dYY)/(N*SY0);
    FE(k) = - (0.5*T) * ( log_sy );

    if mod(k, Nskip)==0
      fprintf('Iteration = %4d, err = %g, F = %g\n', ...
               k,LP(k),FE(k));
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

Model.M_all  = M;
Model.W  = W ;
Model.SY = SY;
Model.A  = W.^2;
Model.method = 'linear_fit';
Model.mode   = 'scalar';
Model.sparse = 'no';

Model.ix_act = 1:M;

% Save expectation variable
Model.XX  = XX ;  % <<x^2>>   M x 1
Model.dYX = dYX;  % <<dy*x'>> N x M
Model.dYY = dYY;  % <<dy^2>>  N x 1

Info.FE  = FE;
Info.LP  = LP;
