function [Ksparse, Adj, Kest, LAMBDA, t] = BISN_integrated(XDat, options)

% The function enables BISN to deal with different practical scenarios.
% Inputs:
% XDat:             n x p matrix of observed data. n is the sample size and 
%                   p is the dimension. Missing data can be denoted by NaN. 
% options:          options.eta: step size (eta >0, 300 by default)
%                   options.maxIter: maximum number of iterations (1e4 by 
%                   default)
%                   options.tol: tolerance to check convergence of the 
%                   algorithm (1e-2 by default)
%                   options.r: the decaying factor (r < 1, 0.5 by default)
%                   options.s: minibatch size (s = p/(1e-3*(p-1)+1)) by 
%                   default)
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
% Lambda:           p x p estimated Lambda matrix.
% t:                total running time.
% AUTHOR: Hang Yu, 2020, NTU.


p = size(XDat, 2);


if ~exist('options','var')
    options.eta = 300;
    options.maxIter = 1e4;
    options.tol = 1e-2;
    options.r = 0.5;
    options.s = p / (1e-3 * (p - 1) + 1);
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
    if ~isfield(options, 'backward_pass')
        options.backward_pass = 1;
    end
    if ~isfield(options, 'prm_learning')
        options.prm_learning = 0;
    end
end

tic;


LAMBDA = zeros(p);
LAMBDA1 = zeros(p);
idl = find(tril(ones(p), -1));

[id_row, id_col] = find(isnan(XDat));
if ~isempty(id_row)
    [id_row1, id_col1] = find(isnan(XDat(:, p:-1:1)));
    XDat(isnan(XDat)) = 0; % set NaN to 0 before input XDat to BISN_missing
    fprintf("forward pass...\n");
    row_missing = unique(id_row);
    id_mat = [id_row, id_col];
    [ML,VL,mD,vD,omega,lambda] = BISN_missing(XDat, row_missing, id_mat, ...
        options.eta, options.maxIter, options.tol, options.r, options.s);
    Kest = ML * diag(mD) * ML';
    LAMBDA(idl) = lambda;
    LAMBDA = LAMBDA + LAMBDA';
    
    if options.backward_pass
        fprintf("backward pass...\n");
        XDat = XDat(:, p:-1:1);
        row_missing = unique(id_row1);
        id_mat = [id_row1, id_col1];
        [ML,VL,mD,vD,omega,lambda] = BISN_missing(XDat, row_missing, id_mat, ...
            options.eta, options.maxIter, options.tol, options.r, options.s);
        Kest1 = ML * diag(mD) * ML';
        Kest1 = Kest1(p:-1:1, p:-1:1);
        LAMBDA1(idl) = lambda;
        LAMBDA1 = LAMBDA1 + LAMBDA1';
        LAMBDA1 = LAMBDA1(p:-1:1, p:-1:1);
        
        LAMBDA = (LAMBDA + LAMBDA1) / 2;
        Kest = (Kest + Kest1) / 2;
    end
    
else
    fprintf("forward pass...\n");
    [ML,VL,mD,vD,omega,lambda] = BISN(XDat, options.eta, options.maxIter, ...
        options.tol, options.r, options.s);
    Kest = ML * diag(mD) * ML';
    LAMBDA(idl) = lambda;
    LAMBDA = LAMBDA + LAMBDA';
    
    if options.backward_pass
        fprintf("backward pass...\n");
        XDat = XDat(:, p:-1:1);
        [ML,VL,mD,vD,omega,lambda] = BISN(XDat, options.eta, options.maxIter, ...
            options.tol, options.r, options.s);
        Kest1 = ML * diag(mD) * ML';
        Kest1 = Kest1(p:-1:1, p:-1:1);
        LAMBDA1(idl) = lambda;
        LAMBDA1 = LAMBDA1 + LAMBDA1';
        LAMBDA1 = LAMBDA1(p:-1:1, p:-1:1);
        
        LAMBDA = (LAMBDA + LAMBDA1) / 2;
        Kest = (Kest + Kest1) / 2;
    end
end
    
t = toc;
fprintf("forward-backward pass is done, elapsed time is %d s\n", t);

fprintf("estimate adjacency matrix by thresholding lambda / (1 + lambda)...\n");
lambda = LAMBDA(idl);
ll = lambda ./ (1+lambda);
[~, fx, x, ~] = kde(ll);
idx = find(x > 1e-2 & x < 0.6);
fx = fx(idx);
x = x(idx);
fx_min = min(fx);
q = find(fx <= fx_min);
hold on; plot(x(q(1)), 0, 'r+');
legend('kernel density', 'selected threshold');
title('Density function of <\lambda_{jk}> / (<\lambda_{jk}> + 1)');

thr = x(q(1)) / (1 - x(q(1)));
Adj = LAMBDA < thr;
Ksparse = Kest;
Ksparse(Adj == 0) = 0;
t = toc;
fprintf("adjacency marix has been estimated, elapsed time is %d seconds\n", t);

if options.prm_learning == 1
    fprintf("start reestimating the non-zero elements...\n");
    S = cov(XDat);
    [idr, idc] = find(tril(Adj, -1));
    Ksparse = QUICParameterLearning(Ksparse, S, idr, idc);
    t = toc;
    fprintf("reestimating the non-zero elements is done, elapsed time is %d seconds\n", t);
end
    