function  [Fup,ix_up,Fdel,ix_del,ix_bad,dmax] = ...
               fincrease_update_act(A,S,Q,SY,ix_act)
%
% --- dF for  Update component
%
global a_min;

%
% --- Residual correlation for active index 'ix_act'
%
a  = A(ix_act);
s  = S(ix_act);
qq = SY * sum(Q(:,ix_act).^2,1);
b  = 1 - s./a;	% scale factor for Q & S
%
% --- dF for  Update component index
%
ix_up = find( (qq > s.*b) & (b > 0) );

a_up  = a(ix_up);
s_up  = s(ix_up)./b(ix_up);
qq_up = qq(ix_up)./(b(ix_up).^2);
x_up  = (qq_up .* a_up)./(s_up .*(s_up + a_up));

% New alpha for ix_up
anew  = s_up.^2 ./(qq_up - s_up);
delta = abs(anew - a_up);
dmax  = max(delta);

% Exclude component with no alpha change
ix    = find( delta > a_min);
ix_up = ix_up(ix);

if ~isempty(ix_up)
	Fup = x_up(ix) - log(x_up(ix)) - 1;
else
	Fup  = [];
end

%
% --- dF for Deletion component index
%     check bad component : (b > 0) should be satisfied
%
ix_del = find( (qq <= s.*b) & (b > 0) );
ix_bad = find( b <= 0 );

b_del  = max(b(ix_del),eps);
a_del  = a(ix_del);
s_del  = s(ix_del)./b_del;
qq_del = qq(ix_del)./(b_del.^2);
delta  = a_del + s(ix_del);

if ~isempty(ix_del)
	Fdel = - qq_del ./delta + log(delta./a_del);
else
	Fdel  = [];
end

% Back to full space index
ix_up  = ix_act(ix_up);
ix_del = ix_act(ix_del);
ix_bad = ix_act(ix_bad);
return
