  % 1 x 1  % 1 x 1% plot_info
% plot Free energy and other infomation
a_min = 1e-5;
NX = 2; 
NY = 2;
alpha_plot = 0;

FE  = Info.FE;

tend = length(FE);

% Start and End of iteration number for plot
tmode = 2;
t0 = 10;
t3 = 600;
switch	tmode
case	1
	t1 = t0;	
	t2 = tend;	
case	2
	t1 = t3;	
	t2 = tend;	
case	3
	t1 = t0;	
	t2 = t3;
end

Hmode = 1;
Fmode = 2;
% Fmode = 1 :Remove sudden drop of free energy due to weight pruning

if Fmode == 1
	% Remove sudden drop of free energy due to weight pruning
	Fdif = diff(FE);
	
	ixz  = find( Fdif < 0);
	ixz  = [ixz, ixz + 1, ixz + 2];
	ixz  = ixz(:);
	
	ix = setdiff2(1:tend,ixz);
else
	ix = 1:tend;
end

% Free energy
nfig = 1;
subplot(NX,NY,nfig)
%hold on
plot(ix,FE(ix))

title('Free energy')
xlim([t1 t2])

%return

% Model entropy

if	Hmode < 2 &isfield(Info,'H')
	nfig = nfig + 1;
	subplot(NX,NY,nfig)
	
	if isfield(parm,'T') 
		switch	Hmode 
		case 0
			H   = -Info.H * parm.T/log(parm.T);
		case 1
			H   = -Info.H /log(parm.T);
		end
	else
		H   = -Info.H ;
	end
	
	plot(ix,H(ix))
	hold on
	title('Effective parameter number')
	xlim([t1 t2])
end

if	Hmode == 2 & isfield(Info,'M')
	nfig = nfig + 1;
	subplot(NX,NY,nfig)

	plot(ix,Info.M(ix), '-b')
	title('Number of input dimension')
	xlim([t1 t2])
end

if	isfield(Info,'LP')
	nfig = nfig + 1;
	subplot(NX,NY,nfig)
	LP  = Info.LP;
	plot(ix,LP(ix))
	title('Expected log-likelihood')
	xlim([t1 t2])
end

if	isfield(Info,'Err')
	nfig = nfig + 1;
	subplot(NX,NY,nfig)
	Err  = Info.Err;
	plot(ix,Err(ix))
	title('Error')
	xlim([t1 t2])
end



if alpha_plot == 0, return; end;

% Hyper variance parameter
if isfield(Model,'A')
	if isfield(Model,'ix_act')
		N = size(Model.A ,1);
		A = zeros(N,Model.M_all);
		A(:,Model.ix_act) = Model.A;
	else
		A =  Model.A;
	end

    nfig = nfig + 1;
	subplot(NX,NY,nfig)
	plot(A')
	title('Hyper variance parameter')

    nfig = nfig + 1;
	subplot(NX,NY,nfig)
	hist(Model.A, 100)
	title('Hyper variance parameter')
	
	M=length(Model.A);
	K=sum(Model.A < a_min);
	fprintf('# of active dim = %d, # of near zero = %d\n', M, K)
end


if alpha_plot == 1, return; end;

if isfield(Info, 'A')
    AA  = Info.A ;
    nfig = nfig + 1;
	subplot(NX,NY,nfig)
	plot(AA')
	title('Hyper variance histry')
end
