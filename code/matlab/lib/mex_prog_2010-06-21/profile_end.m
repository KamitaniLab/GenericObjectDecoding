function	profile_end(profile_on, profile_dir)

if ~exist('profile_on','var'), profile_on = 0; end;

switch profile_on
case	0,
    ptime(toc)
case	1,
    ptime(toc)
    
    if str2num(strtok(version,'.')) > 6
%        profile off
		if ~exist('profile_dir','var'), profile_dir = 'profile'; end;
        profile viewer
        p = profile('info');
        profsave( p, profile_dir)
    else
		if ~exist('profile_dir','var'), profile_dir = './profile'; end;
	    pwd_dir = pwd;
		if ~exist(profile_dir,'dir'), 
			flag=mkdir(profile_dir);
		end
	    cd(profile_dir)
        profile report profile_job
	    cd(pwd_dir)
    end
    
case	2,
    ptime(toc)
    profile viewer
end
