function	[xdata,parm] = normalize_data(xdata,norm_mode,parm)
% Normalize data
% --- Usage
%  1. Normalize data and set mean & norm to 'parm'
%  [xdata, parm] = normalize_data(xdata)
%  [xdata, parm] = normalize_data(xdata, norm_mode)
%
%  2.  Normalize data using 'parm'
%  [xdata] = normalize_data(xdata, norm_mode, parm)
%
% --- Input
% xdata : [Xdim x Nsample x Ntrial]
% --- Optional input
% norm_mode : normalization mode
%           = 0: zero mean & std is calculated
%           = 1: zero mean & normalized [Default]
%           = 2: normalized (mean is unchanged)
%
% parm  : if the following fields exist, these values are used for mean & std 
%    parm.xmean = mean of xdata 
%    parm.xnorm = std (root of variance) of xdata
%
% --- Output
%    parm.xmean = mean of xdata                    [Xdim x 1]
%    parm.xnorm = std (root of variance) of xdata  [Xdim x 1]
%  if norm_mode == 0,
%     xdata : zero mean
%  if norm_mode = 1
%     xdata : zero mean & normalized
%  if norm_mode = 2
%     xdata : normalized  (mean is unchanged)
%
%  if 'parm.xmean' & 'parm.xnorm' are given, 
%     these values are used as mean & std
% 
% 2007/1/27 M. Sato
% 2007-4-8 Modified by M. Sato
% 2009-11-2 Modified by M. Sato, norm_mode = 2 is added

if ~exist('norm_mode','var'),
	norm_mode = 1;
end
if ~exist('parm','var'),
	parm = [];
end

[Xdim,Ndata,Ntrial] = size(xdata);

xdata = reshape(xdata, [Xdim, Ndata*Ntrial]);

% Mean
if ~isfield(parm,'xmean') | isempty(parm.xmean)
	parm.xmean = sum(xdata,2)/(Ndata*Ntrial);
end

xdim = length(parm.xmean);

if xdim > Xdim,
	parm.xmean = parm.xmean(1:Xdim);
elseif xdim < Xdim,
	parm.xmean = repmat(parm.xmean,[Xdim/xdim 1]);
end

% Zero mean
if norm_mode < 2
% 	xdata = xdata - repmat(parm.xmean, [1 Ndata]);
	xdata = repadd( double(xdata) , - double(parm.xmean) );
end

% Variance
if ~isfield(parm,'xnorm') | isempty(parm.xnorm)
	parm.xnorm = sqrt(sum(xdata.^2,2)/((Ndata-1)*Ntrial)); % modified by TH: use unbiased variance
end

xdim = length(parm.xnorm);

if xdim > Xdim,
	parm.xnorm = parm.xnorm(1:Xdim);
elseif xdim < Xdim,
	parm.xnorm = repmat(parm.xnorm,[Xdim/xdim 1]);
end

% Normalize amplitude
if norm_mode > 0
% 	xdata = xdata./repmat(parm.xnorm, [1 Ndata]);
	xdata = repmultiply( double(xdata) , 1./double(parm.xnorm) );
end

xdata = reshape(xdata, [Xdim, Ndata, Ntrial]);
