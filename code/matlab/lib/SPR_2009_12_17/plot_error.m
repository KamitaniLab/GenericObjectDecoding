% plot_error

% Data dir
rootdir   = [getenv('MATHOME') '/umeda_data/'];
datadir   = [rootdir 'motion_unit_rate/'];
modeldir  = [rootdir 'model_spr3/'];
errordir  = [rootdir 'model_err/'];

dataname  = 'motion';
modelbase = 'neuron_to_motion';
%modelbase = 'motion_to_neuron';

xy_mode = [1 3];

Nxyz = 6;   % 6 angle
Nxyz = 3*7; % 7 xyz
Nneuron = 23;
Nmotion = 16;
motion_id  = [1:Nmotion];
method_id  = 1;
taumode_id = 1;

taumode{1} = ['_at'  ];
taumode{2} = ['_pred'];
taumode{3} = ['_back'];
taumode{4} = ['_both'];
mtype = {'w','r','b','c'};

Nxymode = length(xy_mode);
gof_all = zeros(Nxyz,Nmotion,Nxymode);

m=taumode_id;

for k=1:Nxymode
	modelname = [modelbase taumode{m} sprintf('_xy%d',xy_mode(k))];

	errorfile = [errordir  modelname '_err.mat'];
	load(errorfile,'gof_list')
	gof_all(:,:,k) = gof_list;

end

gofmin = 0.4;

gof  = zeros(Nxyz,Nmotion);
gof1 = gof_all(:,:,1);
gof2 = gof_all(:,:,2);

ix = find( gof1 < gofmin );
gof1(ix) = 0;
ix = find( gof2 < gofmin );
gof2(ix) = 0;

gof = gof1 - gof2;

subplot(2,1,1)
plot(gof1(:))
hold on
plot(gof2(:),'-r')

subplot(2,1,2)
plot(gof(:))

hold on

return
%gof_list = zeros(Nxyz,Nneuron,Nmotion);
gofmin = 0.6;

ix = find( gof_list(:) < gofmin );

gof_list(ix) = 0;

mode = 2;

switch	mode
case	1
	% xyz-neuron
	str1 = 'xyz';
	str2 = 'neuron';
	gof = sum(gof_list,3);
case	2
	% neuron-motion
	str1 = 'neuron';
	str2 = 'motion';
	gof = squeeze(sum(gof_list,1));
case	3
	% xyz-motion
	str1 = 'xyz';
	str2 = 'motion';
	gof = squeeze(sum(gof_list,2));
end

imagesc(gof)
ylabel(str1)
xlabel(str2)
title([str1 '-' str2 '(' modelname ')'],'Interpreter','none')
