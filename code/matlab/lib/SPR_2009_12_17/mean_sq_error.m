function	[err,rcor] = mean_sq_error(y,z)
% Normalised mean squared error and correlation coefficient
% [err,rcor] = mean_sq_error(y,z)
%  
%  err = sum((y-z).^2)/sum((y - <y>).^2)
% rcor = sum((y - <y>).*(z - <z>))/sqrt(sum((y-<y>).^2)*sum((z-<z>).^2))
% GoF  = 1 - err

[N,T,Ntr] = size(y);

y = reshape(y,[N,T*Ntr]);
z = reshape(z,[N,T*Ntr]);

err  = sum((y(:)-z(:)).^2);

y  = repadd(y , -mean(y,2));
z  = repadd(z , -mean(z,2));

err  = err/sum(y(:).^2);
rcor = sum(y(:).*z(:))/sqrt(sum(y(:).^2)*sum(z(:).^2));
