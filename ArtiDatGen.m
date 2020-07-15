function [XDat,Ktrue] = ArtiDatGen(P,N,Pt)

% N -- sample size
% P -- dimension
% Pt -- no. of non-zero elements in the Cholesky decomposition of the
% precision matrix



[rid,cid] = find(triu(ones(P),1));
Pe = length(rid);

% Pt = ceil(r*Pe); 
ide = randperm(Pe,Pt).';

Ku = sparse(rid(ide),cid(ide),sign(rand(Pt,1)-0.5).*(0.5+0.5*rand(Pt,1)),P,P);
Ku = spdiags(1+0.5*rand(P,1),0,Ku);
Su = speye(P)/Ku;
Sd = spdiags(sqrt(sum(Su.^2,2)),0,P,P);
Su = Sd\Su;
Ku = Ku*Sd;
Ktrue = Ku.'*Ku;
XDat = randn(N,P)/Ku.'; %*Su.';%
id = randperm(P)';
XDat = XDat(:,id);
Ktrue = Ktrue(id,id);


% if P <= 10000
%     XDat = randn(N,P)/Ku.'; %*Su.';%
% else
% %     S = 0;
% %     for i = 1:N
% %         XDat = Su*randn(P,1);
% %         S = (i-1)/i*S+XDat*XDat.'/i;
% %     end
%     
%     A = sparse([(1:P).';rid],[(1:P).';cid],[sqrt(chi2rnd(N+1-(1:P).'));randn(Pe,1)],P,P).';
%     Su = Su/sqrt(N);
%     SA = Su*A;
%     S = zeros(P);
%     for i = 1:P
%         S(i,i) = sum(SA(i,:).^2);
%         for j = i+1:P
%             S(i,j) = SA(i,:)*SA(:,j);
%             S(j,i) = S(i,j);
%         end
%     end 
% end