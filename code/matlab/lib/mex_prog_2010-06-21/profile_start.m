function	profile_start(profile_on)

if ~exist('profile_on','var'), profile_on = 0; end;

if profile_on == 0,
	tic
elseif profile_on > 0,
	profile on -detail builtin
	tic
end;
