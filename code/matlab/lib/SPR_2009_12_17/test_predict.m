function	[ypred,ytest] = test_predict(xtest,ytest,modelfile)
% --- Prediction for test data
%    [ypred,ytest] = test_predict(xtest,ytest,modelfile)
%  xtest : test input  data ( Xdim x Timesample x Ntrial)
%  ytest : test output data ( Ydim x Timesample x Ntrial)
%  ypred : predicted output from xtest
%  if ytest is empty, empty ytest is returned
%  if ytest is not empty, ytest is time aligned with ypred

load(modelfile, 'Model', 'parm');
%
% --- Estimate prediction for test data
%

% Time alignment for prediction using embedding input
[tx,ty] = pred_time_index(xtest,parm);
xtest = xtest(:,tx,:);

if ~isempty(ytest),
	ytest = ytest(:,ty,:);
end

if isfield(parm,'data_norm') && parm.data_norm==2
	% Add Bias term
	[M,T,K] = size(xtest);
	xtest = [xtest; ones(1,T,K)];
end;

% Use normalization constant calculated by training data
xtest = normalize_data(xtest, parm.data_norm, parm);

% --- Prediction for test data
ypred = predict_output(xtest, Model, parm);
