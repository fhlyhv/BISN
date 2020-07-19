%% generate artificial data
clear all;

p = 500; % dimension
n = 4*p;  % sample size
ne = 2*p; % no. of non-zero elements in the cholesky decomposition of K
[data,Ktrue] = ArtiDatGen(p,n,ne);

% set missing data points
missing_data = 0;  % set missing_data = 1 to include missing data in XDat
if missing_data
    id_missing = randperm(n * p, round(0.05 * n * p));
    data(id_missing) = NaN;
end

%% call BISN (The following options result in the original BISN algorithm)
options.normalize = 0;        % set to 1 if the data is not normalized
% options.eta = 100;            % shrink eta if the algorithm diverges
options.backward_pass = 0;    % set to 1 if the sample size n is small
options.prm_learning = 0;     % set to 1 if the sample size n is small and the nonzero entries in K cannot be well estimated
Ksparse = BISN_integrated(data, options);

%% check performance
fprintf('Comparison between BISN estimates and the true precision matrix\n');
idl = find(tril(ones(p),-1));
pr = full(sum(Ksparse(idl)~=0&Ktrue(idl)~=0)/sum(Ksparse(idl)~=0));
fprintf('precision = %.2f.\n',pr);
rc = full(sum(Ksparse(idl)~=0&Ktrue(idl)~=0)/sum(Ktrue(idl)~=0));
fprintf('recall = %.2f.\n',rc);
f1s = 2*pr*rc/(pr+rc);
fprintf('F1-score = %.2f.\n',f1s);






