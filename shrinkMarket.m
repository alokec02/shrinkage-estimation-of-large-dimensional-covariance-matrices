function [sigma,shrinkage]=shrinkMarket(x,shrink)

% function sigma=covmarket(x)
% x (t*p): t iid observations on n random variables
% sigma (p*p): invertible covariance matrix estimator
%
% This estimator is a weighted average of the sample
% covariance matrix and a "prior" or "shrinkage target".
% Here, the prior is given by a one-factor model.
% The factor is equal to the cross-sectional average
% of all the random variables.
%
% if shrink is specified, then this constant is used for shrinkage

% de-mean returns
t=size(x,1);
n=size(x,2);
meanx=mean(x);
x=x-meanx(ones(t,1),:);
xmkt=mean(x')';

% compute sample covariance matrix and prior
sample=cov([x xmkt]);
covmkt=sample(1:n,n+1);
varmkt=sample(n+1,n+1);
sample(:,n+1)=[];
sample(n+1,:)=[];
prior=covmkt*covmkt'./varmkt;
prior(logical(eye(n)))=diag(sample);

if (nargin < 2 | shrink == -1) % compute shrinkage parameters and constant

  % what we call p 
  y=x.^2;
  phiMat=y'*y/t - 2*(x'*x).*sample/t + sample.^2;
  phi=sum(sum(phiMat));
  
  % what we call r
  y = x.^2;
  help1 = xmkt(:,ones(1,n));
  z=x.*help1;
  help2 = covmkt(:,ones(1,n));
  term1 = y'*z/t - (x'*help1).*sample/t - (x'*x).*help2/t + help2.*sample;
  term1 = help2'.*term1/varmkt;
  term3 = z'*z/t - help1'*help1.*sample/t - varmkt*x'*x/t + varmkt*sample;
  term3 = (covmkt*covmkt').*term3/varmkt^2;
  rhoMat = 2*term1-term3;
  rhoMat(logical(eye(n)))=zeros(n,1);
  rho=sum(diag(phiMat))+sum(sum(rhoMat));
  
  % what we call c
  gamma=norm(sample-prior,'fro')^2;
  
  % compute shrinkage constant
  kappa = (phi - rho)/gamma;
  shrinkage = max(0,min(1,kappa/t));
  
else % use specified constant
  shrinkage = shrink;
end

% compute the estimator
sigma=shrinkage*prior+(1-shrinkage)*sample;



