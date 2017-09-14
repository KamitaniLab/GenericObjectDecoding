%clear all

%fname = 'D:\MatData\umeda data2\modified_motion01_03.mat';
%%load(fname)
%
%ydata = ydata(2:3,:,:);
%
%xdata(:,:,ind) = [];
%ydata(:,:,ind) = [];

%fsave = 'test_spike.mat';
%
%load(fsave)

ypred = predict_output(xtest, Model, parm);
ydev  = predict_variance(xtest, Model, parm);

ymin = min(ydev(:));
ydev = ydev - ymin;

yerr = abs(ypred - ytest);

[err,rcor] = mean_sq_error(yerr,ydev)

NY = 2; NX = 1;
nfig = 1;
subplot(NY,NX,nfig)
%plot(yerr(1,:,1))
plot(yerr(1,:,1)/max(yerr(1,:,1)))
hold on

%nfig = nfig+1;
subplot(NY,NX,nfig)
%plot(ydev(1,:,1),'-r')
hold on
plot(ydev(1,:,1)/max(ydev(1,:,1)),'-r')

return

[M,T,Ntr] = size(xdata);
[N,T,Ntr] = size(ydata);

x = reshape(xdata,[M,T*Ntr]);
y = reshape(ydata,[N,T*Ntr]);

NY = 5; NX = 2;
nfig = 0;

for n=1:N
	nfig = nfig+1;
	if nfig > NX*NY, figure; nfig=1; end
	subplot(NY,NX,nfig)
	plot(y(n,:))
end

figure

NY = 6; NX = 2;

for n=1:M
	subplot(NY,NX,n)
	plot(x(n,:))
end

%save(fsave,'xdata','ydata')
