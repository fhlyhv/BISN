function [Ksparse, Adj, Kest, Vest, Lambda, omega, run_time, data_est] = BISN_integrated(data, options)

% The function enables BISN to deal with different practical scenarios.
% Inputs:
% data:             n x p matrix of observed data. n is the sample size and 
%                   p is the dimension. Missing data can be denoted by NaN. 
% options:          options.eta: step size (eta >0, 300 by default)
%                   options.maxIter: maximum number of iterations (1e4 by 
%                   default)
%                   options.tol: tolerance to check convergence of the 
%                   algorithm (1e-2 by default)
%                   options.r: the decaying factor (r < 1, 0.5 by default)
%                   options.s: minibatch size (s = p/(1e-3*(p-1)+1)) by 
%                   default)
%                   options.normalize: boolean value to decide whether to
%                   normalize the data before applying BISN ?1 by default). 
%                   Note that when options.prm_learning = 1, the original 
%                   unnormalize the data is still used for reestimating the 
%                   nonzero elements in the precision matrix via maximum 
%                   likelihood.
%                   options.backward_pass: boolean value to decide whether
%                   to enable the backward pass or not. (1 by default)
%                   options.backward_pass = 1 would run the BISN algorithm
%                   again by reversely ordering the data (i.e., the
%                   backward pass). and then average the results from both
%                   forward and backward pass. This will improve the
%                   estimation accuracy, especially when the sample size is
%                   small.
%                   options.prm_learning: boolean value to decide whether
%                   to reestimate the non-zero elements via maximum 
%                   likelihood or not. (0 by default)
% Outputs:
% Ksparse:          p x p matrix with the same zero pattern as the
%                   estimated adjacency matrix Adj. The nonzero elements in
%                   Ksparse is reestimated by maximum likelihood if 
%                   options.prm_learning = 1.
% Adj:              estimated adjacency matrix using the method in Section
%                   V in (Yu et al, Variational wishart approximation for 
%                   graphical model selection: Monoscale and multiscale 
%                   models, 2019).
% Kest:             p x p full matrix. Kest = ML * MD * ML', where MD and
%                   ML denotes the mean of the D and L matrix.
% Vest:             p x p full matrix of estimated variaces of elements in
%                   the precision matrix.
% Lambda:           p x p estimated Lambda matrix.
% omega:            the estiamted global shrinkage parameter
% run_time:         total running time.
% data_est:         n x p matrix of estimated (normalized) data with missing
%                   values imputed by BISN. 
% Note that except Ksparse, all other output parameters are estimated based
% on the normalized data if options.normalize = 1, and the original input
% data if options.normalize = 0.
% AUTHOR: Hang Yu, 2020, NTU.


[n, p] = size(data);


if ~exist('options','var')
    options.eta = 300;
    options.maxIter = 1e4;
    options.tol = 1e-2;
    options.r = 0.5;
    options.s = p / (1e-3 * (p - 1) + 1);
    options.normalize = 1;
    options.backward_pass = 1;
    options.prm_learning = 0;
else
    if ~isfield(options, 'eta')
        options.eta = 300;
    end
    if ~isfield(options, 'maxIter')
        options.maxIter = 1e4;
    end
    if ~isfield(options, 'tol')
        options.tol = 1e-2;
    end
    if ~isfield(options, 'r')
        options.r = 0.5;
    end
    if ~isfield(options, 's')
        options.s = p / (1e-3 * (p - 1) + 1);
    end
    if ~isfield(options, 'normalize')
        options.normalize = 1;
    end
    if ~isfield(options, 'backward_pass')
        options.backward_pass = 1;
    end
    if ~isfield(options, 'prm_learning')
        options.prm_learning = 0;
    end
end

tic;


Lambda = zeros(p);
Lambda1 = zeros(p);
idl = find(tril(ones(p), -1));

data_normalize = data;
[id_row, id_col] = find(isnan(data_normalize));
if ~isempty(id_row)
    id_missing = id_row + (id_col - 1) * n;
    if options.normalize
        obsv_mat = ones(n, p);
        obsv_mat(id_missing) = 0;
        data_normalize(id_missing) = 0;
        data_normalize = data_normalize - repmat(sum(data_normalize) ./ sum(obsv_mat), n, 1);
        data_normalize(id_missing) = 0;
        data_normalize = data_normalize * diag(1 ./ sqrt(sum(data_normalize .^ 2) ./ (sum(obsv_mat) - 1))');
    else
        data_normalize(id_missing) = 0; % set NaN to 0 before input XDat to BISN_missing
    end
    row_missing = unique(id_row);
    id_mat = [id_row, id_col];
    fprintf("forward pass...\n");
    [ML,VL,mD,vD,omega,lambda,data_est] = BISN_missing(data_normalize, row_missing, id_mat, ...
        options.eta, options.maxIter, options.tol, options.r, options.s);
    Kest = ML * spdiags(mD, 0, p, p) * ML';
    ML2 = ML .^ 2;
    mD2 = mD .^ 2;
    ML2pVL = ML .^ 2 + VL;
    Vest = ML2pVL * spdiags(mD2 + vD, 0, p, p) * ML2pVL.' - ...
        ML2 * spdiags(mD2, 0, p, p) * ML2.';
    Lambda(idl) = lambda;
    Lambda = Lambda + Lambda';
    
    if options.backward_pass
        data_normalize(id_missing) = 0;
        data_normalize = data_normalize(:, p:-1:1);
        id_mat(:, 2) = p + 1 - id_mat(:, 2);
        fprintf("backward pass...\n");
        [ML,VL,mD,vD,omega1,lambda,data_est1] = BISN_missing(data_normalize, row_missing, id_mat, ...
            options.eta, options.maxIter, options.tol, options.r, options.s);
        Kest1 = ML * spdiags(mD, 0, p, p) * ML';
        ML2 = ML .^ 2;
        mD2 = mD .^ 2;
        ML2pVL = ML .^ 2 + VL;
        Vest1 = ML2pVL * spdiags(mD2 + vD, 0, p, p) * ML2pVL.' - ...
            ML2 * spdiags(mD2, 0, p, p) * ML2.';
        Lambda1(idl) = lambda;
        Lambda1 = Lambda1 + Lambda1';
        
        omega = (omega + omega1) / 2;
        Lambda = (Lambda + Lambda1(p:-1:1, p:-1:1)) / 2;
        Kest = (Kest + Kest1(p:-1:1, p:-1:1)) / 2;
        Vest = (Vest + Vest1(p:-1:1, p:-1:1)) / 2;
        data_est = (data_est + data_est1(:, p:-1:1)) / 2;
    end
    
else
    if options.normalize
        data_normalize = data_normalize - repmat(mean(data_normalize),n,1);
        data_normalize = data_normalize*diag(1./std(data_normalize)');
    end
    fprintf("forward pass...\n");
    [ML,VL,mD,vD,omega,lambda] = BISN(data_normalize, options.eta, options.maxIter, ...
        options.tol, options.r, options.s);
    Kest = ML * spdiags(mD, 0, p, p) * ML';
    ML2 = ML .^ 2;
    mD2 = mD .^ 2;
    ML2pVL = ML .^ 2 + VL;
    Vest = ML2pVL * spdiags(mD2 + vD, 0, p, p) * ML2pVL.' - ...
        ML2 * spdiags(mD2, 0, p, p) * ML2.';
    Lambda(idl) = lambda;
    Lambda = Lambda + Lambda';
    data_est = data_normalize;
    
    if options.backward_pass
        data_normalize = data_normalize(:, p:-1:1);
        fprintf("backward pass...\n");
        [ML,VL,mD,vD,omega1,lambda] = BISN(data_normalize, options.eta, options.maxIter, ...
            options.tol, options.r, options.s);
        Kest1 = ML * spdiags(mD, 0, p, p) * ML';
        Kest1 = Kest1(p:-1:1, p:-1:1);
        ML2 = ML .^ 2;
        mD2 = mD .^ 2;
        ML2pVL = ML .^ 2 + VL;
        Vest1 = ML2pVL * spdiags(mD2 + vD, 0, p, p) * ML2pVL.' - ...
            ML2 * spdiags(mD2, 0, p, p) * ML2.';
        Vest1 = Vest1(p:-1:1, p:-1:1);
        Lambda1(idl) = lambda;
        Lambda1 = Lambda1 + Lambda1';
        Lambda1 = Lambda1(p:-1:1, p:-1:1);
        
        omega = (omega + omega1) / 2;
        Lambda = (Lambda + Lambda1) / 2;
        Kest = (Kest + Kest1) / 2;
        Vest = (Vest + Vest1) / 2;
    end
end
    
t = toc;
fprintf("forward-backward pass is done, elapsed time is %d seconds\n", t);
run_time = t;

fprintf("estimate adjacency matrix by thresholding lambda / (1 + lambda)...\n");
tic;
lambda = Lambda(idl);
ll = lambda ./ (1+lambda);
[~, fx, x, ~] = kde(ll, 4096, 0, 1);
idx = find(x > 1e-2 & x < 0.6);
fx = fx(idx);
x = x(idx);
fx_min = min(fx);
q = find(fx <= fx_min);
hold on; plot(x(q(1)), 0, 'r+');
legend('kernel density', 'selected threshold');
title('Density function of <\lambda_{jk}> / (<\lambda_{jk}> + 1)');

thr = x(q(1)) / (1 - x(q(1)));
Adj = Lambda < thr;
Ksparse = Kest;
Ksparse(Adj == 0) = 0;
t = toc;
fprintf("adjacency marix has been estimated, elapsed time is %d seconds\n", t);
run_time = run_time + t;

if options.prm_learning == 1
    fprintf("start reestimating the non-zero elements...\n");
    tic;
    if ~isempty(id_row)
        if ~options.normalize
            obsv_mat = ones(n, p);
            obsv_mat(id_missing) = 0;
        end
        data(id_missing) = 0;
        S = data' * data ./ (obsv_mat' * obsv_mat - 1);
    else
        S = cov(data);
    end
    [idr, idc] = find(tril(Adj, -1));
    Ksparse = QUICParameterLearning(Ksparse, S, idr, idc);
    t = toc;
    fprintf("reestimating the non-zero elements is done, elapsed time is %d seconds\n", t);
    run_time = run_time + t;
end
    