%% generate artificial data


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
%% Normalize the data to have zero mean and unit variance
[n,p] = size(data);
if any(isnan(data(:)))
    id = ~isnan(data);
    normalized_data = data;
    normalized_data(~id) = 0;
    normalized_data = normalized_data - repmat(sum(normalized_data) ./ sum(id), n, 1);
    normalized_data = normalized_data * diag(1 ./ sqrt(sum(normalized_data .^ 2) ./ (sum(id) - 1))');
else
    normalized_data = data - repmat(mean(data),n,1);
    normalized_data = normalized_data*diag(1./std(normalized_data)');
end

%% call BISN
% options.eta = 100;            % shrink eta if the algorithm diverges
% options.backward_pass = 0;    % set to be 1 if the sample size n is small
options.prm_learning = 1;     % set to be 1 if the sample size n is small and the nonzero entries in K cannot be well estimated
Ksparse = BISN_integrated(normalized_data, options);

%% check performance
fprintf('Comparison between BISN estimates and the true precision matrix\n');
idl = find(tril(ones(p),-1));
pr = full(sum(Ksparse(idl)~=0&Ktrue(idl)~=0)/sum(Ksparse(idl)~=0));
fprintf('precision = %.2f.\n',pr);
rc = full(sum(Ksparse(idl)~=0&Ktrue(idl)~=0)/sum(Ktrue(idl)~=0));
fprintf('recall = %.2f.\n',rc);
f1s = 2*pr*rc/(pr+rc);
fprintf('F1-score = %.2f.\n',f1s);






