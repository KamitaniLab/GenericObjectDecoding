% Batch script for sparse prediction
clear all

% --- training method for variable sample size data
method_id = 1;
%case	1 linear_sparse_space_sw
%case	2 linear_sparse_space_vec_sw
%case	3 linear_sparse_stepwise_sw
%case	4 linear_sparse_stepwise_em_sw

% --- Basic Learning Parameters
parm.Ntrain = 2000; % # of total training iteration
parm.Nskip  = 100;	% skip steps for display info

% --- Time delay embedding parameter
parm.Tau   = 1 ;   % Lag time steps
parm.Dtau  = 3 ;   % Number of embedding dimension
parm.Tpred = 0 ;   % Prediction time step : y(t+Tpred) = W * x(t)

% --- Normalization parameter
parm.data_norm  = 1;	% Normalize input and output

% File name of old training result for retraining 
old_file = [];

% File name for test data
datafile  = ['./test/test.mat'];
modelfile = ['./test/model'];
newdata   = 1; % = 1: make new data

%parm.Trial  = [1000 200 100];% # of samples in each trial
parm.Trial  = [1000 600];% # of samples in each trial

% --- Make or Load training & test data
if newdata == 0 && exist(datafile,'file')
	load(datafile, ...
		'xdata','ydata','xtest','ytest')
else
	% Training & test data setting
	Tdata  = 1000 ; % # of training data
	Ttest  = 1000 ; % # of test data
	
	parm.Ydim = 3;
	parm.Xdim = 200;   % Input dim
	parm.Meff = 30;    % Effective Input
	parm.Sdim = 10; % number of sinusoidal input
	parm.Tmin = 50; % minimum period
	parm.Tmax = 200;% maximum period
	parm.SY   = [0.3; 0.1; 0.5];% output moise variance
	parm.Ntrial = length(parm.Trial);

	parm.sy = parm.SY;
	[xdata,ydata,Wout,Weff,Wrdn,Win,Wirr] = ...
		make_train_data(parm, Tdata);
	
	parm.sy = 0.0;
	[xtest,ytest] = ...
		make_train_data(parm, Ttest, Wout,Wrdn,Win,Wirr);
	
	if ~isempty(datafile)
		save(datafile, ...
			'xdata','ydata','xtest','ytest','Wout','Weff','parm')
	end
end

T = size(ydata,2);
if T~=size(xdata,2), error('Time of X and Y is not match'); end

if isfield(parm,'data_norm') && parm.data_norm==2
	% Add Bias term
	[M,T,K] = size(xdata);
	xdata = [xdata; ones(1,T,K)];
end;

% Time alignment for prediction using embedding input
[tx,ty] = pred_time_index(xdata,parm);
xdata = xdata(:,tx,:);
ydata = ydata(:,ty,:);
% Adjust sample length for each trial by time embedding
if isfield(parm,'Trial'), parm.Trial = parm.Trial - (T - length(ty)); end;

% Normalize input data
[X,nparm] = normalize_data(xdata, parm.data_norm);
parm.xmean = nparm.xmean;
parm.xnorm = nparm.xnorm;

% Normalize output data
[Y,nparm] = normalize_data(ydata, parm.data_norm);
parm.ymean = nparm.xmean;
parm.ynorm = nparm.xnorm;

% --- Initialization of Model Parameters
if ~isempty(old_file)
	% Start from old result
	load([old_file], 'Model')
else
	Model = [];
end

%profile_on = 1;
%profile_start(profile_on);

%
% --- Sparse estimation
%
switch	method_id
case	1
	[Model, Info] = linear_sparse_space_sw(X, Y, Model, parm);
case	2
	[Model, Info] = linear_sparse_space_vec_sw(X, Y, Model, parm);
case	3
	[Model, Info] = linear_sparse_stepwise_sw(X, Y, Model, parm);
case	4
	[Model, Info] = linear_sparse_stepwise_em_sw(X, Y, Model, parm);
end

%
% --- Estimate prediction error for test data
%

% Time alignment for prediction using embedding input
[tx,ty] = pred_time_index(xtest,parm);
xtest = xtest(:,tx,:);
ytest = ytest(:,ty,:);


if isfield(parm,'data_norm') && parm.data_norm==2
	% Add Bias term
	[M,T,K] = size(xtest);
	xtest = [xtest; ones(1,T,K)];
end;

% Use normalization constant calculated by training data
xtest = normalize_data(xtest, parm.data_norm, parm);

% --- Prediction for test data
ypred = predict_output(xtest, Model, parm);
err   = sum((ytest(:)-ypred(:)).^2)/sum(ytest(:).^2)

if ~isempty(modelfile)
	fsave = [modelfile sprintf('_id%d.mat',method_id)];
	save(fsave, 'Model', 'Info', 'parm');
end

plot_predict

%profile_end(profile_on)
return
