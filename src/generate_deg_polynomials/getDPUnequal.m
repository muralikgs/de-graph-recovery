function [lambda_h, lambda_l] = getDPUnequal(n_h, n_l, d_cl, d_ch, d_vl, d_vh, k_hh, k_ll)

% defining all the constants
n_hh = n_h^2;
n_ll = n_l^2;
n_hl = 2*n_l*n_h;

beta_hh = 0.15*log(n_hh/k_hh);
beta_ll = 0.15*log(n_ll/k_ll);

k_hl = (2*k_hh*k_ll).^0.5;

constraint_3_const = (n_hh/k_hh)^0.25 * 1/(d_ch + d_cl)^0.5;
constraint_2_const = 1/(d_cl^2 + d_ch^2);

% defining vectors and matrices that are necessary

unit_vec_h = [1:d_vh]';
unit_vec_l = [1:d_vl]';

unit_vec_h_2 = (1./unit_vec_h).^0.5;
unit_vec_l_2 = (1./unit_vec_l).^0.5;

quad_mat_h_1 = (1./unit_vec_h) * (1./unit_vec_h)';
quad_mat_l_1 = (1./unit_vec_l) * (1./unit_vec_l)';

cvx_begin
    
    variable lambda_vec_h(d_vh, 1);
    variable lambda_vec_l(d_vl, 1);
    
    minimize (n_h * unit_vec_h' * lambda_vec_h + n_l * unit_vec_l' * lambda_vec_l);
    
    subject to
        
        % constraint 1
        n_h * d_cl * unit_vec_h' * lambda_vec_h - n_l * d_ch * unit_vec_l' * lambda_vec_l == 0;
        
        % constraint 2
        (k_hh/n_hh) * beta_hh * (lambda_vec_h' * quad_mat_h_1 * lambda_vec_h) + (k_ll/n_ll) * beta_ll * (lambda_vec_l' * quad_mat_l_1 * lambda_vec_l) <= constraint_2_const;
        
        % constraint 3
        unit_vec_h_2' * lambda_vec_h <= constraint_3_const;
        
        % constraint 4
        (k_hh/n_hh)^0.5 * (unit_vec_h_2' * lambda_vec_h) - (k_hl/n_hl).^0.5 * (unit_vec_l_2' * lambda_vec_l) <= 0;
        (k_hh/n_hh)^0.25 * (unit_vec_h_2' * lambda_vec_h) - (k_ll/n_ll).^0.25 * (unit_vec_l_2' * lambda_vec_l) <= 0;
%        (k_hl/n_hl)^0.5 * (unit_vec_l_2' * lambda_vec_l) - (k_hh/n_hh)^0.5 * (unit_vec_h_2' * lambda_vec_h) <= 0;
%        (k_hl/n_hl)^0.5 * (unit_vec_h_2' * lambda_vec_h) - (k_ll/n_ll)^0.5 * (unit_vec_l_2' * lambda_vec_l) <= 0;
 
        % constraint 5
        lambda_vec_h(1) == 0;
        lambda_vec_l(1) == 0;
        
        % constraint 6
        lambda_vec_h >= 0;
        lambda_vec_l >= 0;
        sum(lambda_vec_h) == 1;
        sum(lambda_vec_l) == 1;
        
cvx_end
        
lambda_h = lambda_vec_h;
lambda_l = lambda_vec_l;
    
end