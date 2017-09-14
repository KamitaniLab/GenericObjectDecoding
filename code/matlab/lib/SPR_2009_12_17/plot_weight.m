% plot estimated weight

plot_mode = 1;
% 	= 1: Weight in embedded space
% 	= 2: Weight in original input space

% Subplot fig number
NX   = 1; 
NY   = 1;
nfig = 0;

%
% --- Recover Estimated Weight matrix
%

[W ,Ydim, Xdim, Dtau] = get_estimated_weight(Model,parm);

nfig = nfig + 1;
subplot(NY,NX,nfig)

% Estimated weight
switch	plot_mode
case	1
	% Weight in embedded space
	if exist('Wout','var'), plot(Wout'); hold on; end;
	plot(W','-r')
case	2
	% Weight in original input space
	Wd = reshape(W, [Ydim, Xdim, Dtau]);
	Wd = sum(abs(Wd),3);
	plot(Wd','-r')
end

title('Estimated Weight')

%title(['Weight of ' parm.jobname],'Interpreter','none')

return
