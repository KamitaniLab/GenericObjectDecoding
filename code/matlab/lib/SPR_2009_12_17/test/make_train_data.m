function	[xdata,ydata,Wout,Weff,Wrdn,Win,Wirr] = ...
			make_train_data(parm,T,Wout,Wrdn,Win,Wirr)
%  Make training and test data set
%   [xdata,ydata,Wout,Weff,Wrdn,Win,Wirr] = ...
%			make_train_data(parm,T,Wout,Wrdn,Win,Wirr)
% - Output
% xdata : Xdim x Tsample x Ntrial
% ydata : Ydim x Tsample x Ntrial
% Wout  : Output weight for time embedded input (Ydim,Xdim*Dtau)
% Weff  : nonzero output weight for (Ydim,Meff*Dtau)
%         effective input dimension : [Meff x Dtau]
% Win   : Effective input mixing matrix
% Wrdn  : Redundant input mixing matrix
% Wirr  : Irrelevant input mixing matrix
% - Input
% parm.Ntrial =  Number of trials
% parm.Ydim   =  Output dim
% parm.Xdim   =  Input dim
% parm.Meff   =  Effective Input dim
% parm.Mrdn   =  Redundant Input dim [= fix(Meff/2)]
% parm.sy     =  Output noise variance
% parm.Tau    =  Lag time steps
% parm.Dtau   =  Number of embedding dimension
%
% *** sinusoidal input (this is part of effective input)
% parm.Sdim = 	number of sinusoidal input
% parm.Tmin = 	minimum period
% parm.Tmax =   maximum period
% *** Correlation between effective input and irrelevant input
% data_id : Correlation structure of input
% = 0 : No correlation between input
% = 1 : Correlation within effective input 
%     + Correlation within irrelevant input
% = 2 : Correlation within effective input
%     + Correlation within irrelevant input
%     + Redundant input which have correlation with effctive input
% = 3 : Correlation between effective input and irrelevant input
%
% *** Output weight
% parm.WXmax = maximum weight of X-dim
% parm.WYmax = bias weight for Y-dim
% parm.Wtau  = effctive time delay length of weight

if ~exist('T','var'), T = 2000; end;

% Input dim
if isfield(parm,'Xdim')
	Xdim = parm.Xdim ;
else
	Xdim = 100;
end
% Output dim
if isfield(parm,'Ydim')
	Ydim = parm.Ydim ;
else
	Ydim = 1;
end
% Effective Input dim
if isfield(parm,'Meff')
	Meff = parm.Meff ;
else
	Meff = 10; ;
end
% Sinusoidal input dimension
if isfield(parm,'Sdim')
	Sdim = parm.Sdim ;
else
	Sdim = 0 ;
end
% Redundant input dimension
if isfield(parm,'Mrdn')
	Mrdn = parm.Mrdn ;
else
	Mrdn = fix(Meff/2);
end

if Meff > Xdim, Meff = Xdim; end;
if Meff < Sdim, Sdim = Meff; end;
if (Meff + Mrdn) > Xdim, Mrdn = Xdim - Meff; end;

% data_id : Correlation structure of input
% = 0 : No correlation between input
% = 1 : Correlation within effective input 
%     + Correlation within irrelevant input
% = 2 : Correlation within effective input
%     + Correlation within irrelevant input
%     + Redundant input which have correlation with effctive input
% = 3 : Correlation between effective input and irrelevant input

if isfield(parm,'data_id')
	data_id = parm.data_id;
else
	data_id = 0;
end
% Input correlation parameter 
% # of principal dimension in mixing matrix
if isfield(parm,'Rcor')
	Rcor = parm.Rcor ;   
else
	Rcor = 10 ;   
end

if isfield(parm,'Ntrial')
	Ntrial = parm.Ntrial;
else
	Ntrial = 1;
end

if isfield(parm,'sx')
	% Sinusoidal input noise variance
	sx = parm.sx   ;	
else
	sx = 0.0;
end
if isfield(parm,'sy')
	% output noise variance
	sy = parm.sy   ;	
else
	sy = 0.1;
end

if length(sy) == 1 && Ydim > 1, sy = repmat(sy, [Ydim,1]); end;

if isfield(parm,'Tpred')
	Tpred = parm.Tpred ;
else
	Tpred = 0;
end
if isfield(parm,'Dtau')
	Tau   = parm.Tau   ;
	Dtau  = parm.Dtau  ;
else
	Tau   = 1;
	Dtau  = 1;
end

TPRED = Tpred + Tau*(Dtau-1);
Ndata = T + TPRED;

xdata = zeros(Meff,Ndata,Ntrial);

% Sinusoidal signal
if Sdim > 0
	xsin   = make_sin_out(Ndata, Sdim , parm);
	xdata(1:Sdim,:,:) = repmat(xsin, [1 1 Ntrial]) ...
	                  + sx*randn(Sdim,Ndata,Ntrial);
end

% Gaussian random input for effctive dimension
if Meff > Sdim
	% Effctive input
	XSdim = (Meff-Sdim);
	
	x = randn(XSdim,Ndata*Ntrial) ;
	
	% Input variable (make correlation among input)
	switch	data_id
	case	0
		% No correlation
		xdata((Sdim+1):Meff,:,:) = reshape(x ,[XSdim,Ndata,Ntrial]); 
		Win = [];
	case	{1,2}
		% Mixing matrix
		if ~exist('Win','var')
			Win = make_nn_diag(XSdim , Rcor);
		end
		xdata((Sdim+1):Meff,:,:) = reshape(Win * x ,[XSdim,Ndata,Ntrial]); 
	case	{3}
		Xrest = (Xdim-Sdim);
		Nrest = (Xdim-Meff);
		% Mixing matrix
		if ~exist('Win','var')
			Win = make_nn_diag(Xrest , Rcor);
		end
		% Correlation between effctive input and irrelevant input
		x = Win * randn(Xrest,Ndata*Ntrial) ;
		
		xdata((Sdim+1):Meff,:,:) = ...
			reshape( x(1:XSdim,:), [XSdim,Ndata,Ntrial]);
		xres  = ...
			reshape( x((XSdim+1):Xrest,:), [Nrest,Ndata,Ntrial]);
	end
end

xdata  = normalize_data(xdata);
xembed = embed_data(xdata,[],parm);

% Output weight
if exist('Wout','var')
	Wout = reshape(Wout, [Ydim, Xdim, Dtau]);
	Weff = Wout(:,1:Meff,:);
	Weff = reshape(Weff, [Ydim, Meff * Dtau]);
else
	[Wout,Weff] = make_test_weight(parm);
end

y = Weff * reshape(xembed ,[Meff*Dtau, T*Ntrial]); 

ymean = mean(y(:));
yvar  = sqrt( mean((y(:) - ymean).^2) );
noise = repmultiply( randn(Ydim,T*Ntrial), sy*yvar);

ydata = zeros(Ydim,Ndata,Ntrial);
ydata(:,(TPRED+1):Ndata,:) = reshape(y + noise ,[Ydim, T, Ntrial]);

% Irrelevant input
XRdim = (Xdim-Meff);

if Xdim > Meff
	switch	data_id
	case	{0}
		% No correlation with effctive input
		% Gaussian random input
		xres = randn(XRdim,Ndata,Ntrial) ;
		Wrdn = [];
		Wirr = [];
	case	{1}
		% No correlation with effctive input
		% Correlation within irrelevant input (correlation with neighbor)
		if ~exist('Wirr','var')
			% mixing matrix for irrelevant component
			Wirr = make_nn_diag(XRdim,Rcor);
		end
		xres = randn(XRdim,Ndata*Ntrial) ;
		xres = reshape(Wirr * xres ,[XRdim,Ndata,Ntrial]); 
		Wrdn = [];
	case	{2}
		% Redundant input: Linear correlation with effctive input
		if ~exist('Wrdn','var')
			% projection matrix for redundant component
			Wrdn = make_rand_orth(Mrdn,Meff);
		end
		
		% Irrelevant inputs
		Nrest = XRdim - Mrdn;
		if Nrest > 0
			% Correlation within irrelevant input
			if ~exist('Wirr','var')
				% mixing matrix for irrelevant component
				Wirr = make_nn_diag(Nrest,Rcor);
			end
			xres2 = randn(Nrest,Ndata*Ntrial) ;
			xres2 = reshape(Wirr * xres2 ,[Nrest,Ndata,Ntrial]); 
			xres = [Wrdn * xdata; xres2];
		else
			% Redundant input: Linear correlation with effctive input
			xres  = Wrdn * xdata;
		end
	case	{3}
		Wrdn = [];
		Wirr = [];
	end
	
	xres  = normalize_data(xres);
else
	xres = [];
	Wrdn = [];
	Wirr = [];
end


xdata = [xdata; xres];

if Ntrial==1,
	xdata = xdata(:,:,1);
end

%save(parm.datafile,'ydata','xdata','Wout')
