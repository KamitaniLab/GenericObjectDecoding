function	fname = make_save_fname(jobname, njob, parm)
% File name for Save

% Result file name
fname = sprintf('%s-%s-%d-%d.mat', parm.savefile, jobname, parm.Ntrain, njob);
fname = [parm.rootdir  fname];
