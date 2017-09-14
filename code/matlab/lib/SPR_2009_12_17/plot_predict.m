% plot_predict(ytest,ypred)


% time range for plot
%Tlim  = [200 500]; 

[N,T,Ntr] = size(ytest);
yt = reshape(ytest,[N,T*Ntr]);
yp = reshape(ypred,[N,T*Ntr]);

% Plot prediction for test data
NY = 2; NX = 1;
nfig = 0;

for n=1:N
	nfig = nfig+1;
	if nfig > NX*NY, figure; nfig=1; end
	subplot(NY,NX,nfig)
	plot(yt(n,:)')
	hold on
	plot(yp(n,:)','-.r')
	%xlim(Tlim)
end

%plot(Info.dID,'o')
return
