function  [Fadd,ix_add] = fincrease_add_act(S,Q,SY,ix_non)
%
% --- dF for inactive index 'ix_non'
%
qq = SY * sum(Q(:,ix_non).^2,1);
s  = S(ix_non);

ix = find( qq > s );
qq = qq(ix);
s  = s(ix);

if ~isempty(ix)
	x_add = qq./s;
	Fadd  = x_add - log(x_add) - 1;
else
	Fadd  = [];
end

% Back to full space index
ix_add = ix_non(ix);
return
