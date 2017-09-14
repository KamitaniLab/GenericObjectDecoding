%mex -v erfc_func.c
%mex -v erfc_func0.c
%mex -v erfc_func2.c
%mex -v trapzf.c
%mex -v trapzf2.c
%mex -v weight_update_iter.c
mex -v weight_out_compo_delay_time.c

%return

%%% Error dY = Y - W * X
mex -v error_delay_time.c
mex -v error_delay_time_sw.c

%%% Error correlation : dYX = dY * X'
mex -v error_corr_delay.c
mex -v error_corr_delay_sw.c

%%% Componentwise weight update
mex -v weight_update_embed.c
mex -v weight_update_embed_aa.c
mex -v weight_update_embed_sw.c
mex -v weight_update_iter.c

%%% Output prediction : Y = W * X
mex -v weight_out_delay_time.c

%%% Multiply/add with repmat
mex -v repmultiply.c
mex -v repadd.c

mex -v trapzf.c
mex -v erfc_func.c

%mex -v error_delay_time_tau.c
%mex -v error_corr_delay_tau.c

%mex -v sequential_weight_update.c
%mex -v sequential_weight_update_nt.c
%mex -v sequential_weight_update_tn.c
