function x = make_sin_out(T,Sdim,parm)
% Sinusoidal signal

if isfield(parm,'Tstart'),
	Tstart = parm.Tstart;
else
	Tstart = rand(1);
end

% period
Tmin = parm.Tmin;
Tmax = parm.Tmax;

if Sdim==1,
	Tperiod = Tmax;
elseif isfield(parm,'Tperiod')
	Tperiod = parm.Tperiod;
else
	Tstp = (Tmax - Tmin)/(Sdim-1);
	Tperiod = (0:(Sdim-1))*Tstp + Tmin; 
	Tperiod = circshift(Tperiod, fix(Sdim/2));
%	Tperiod = ((Sdim-1):-1:0)*Tstp + Tmin; 
%	Tperiod = Tmax*rand(1,Sdim) + Tmin; 
end

% frequency
omega  = 2*pi./Tperiod(:);

% Sinusoidal signal
t = (1:T) + Tmax*Tstart;
%t = repadd( omega * t, - (2*pi) * rand(Sdim,1) );
x = sin( omega * t );
