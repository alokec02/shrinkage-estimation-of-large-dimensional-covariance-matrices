function [sigma,shrinkage]=shrinkCorr(x,shrink)

% function sigma=covcorr(x)
% x (t*n): t iid observations on n random variables
% sigma (n*n): invertible covariance matrix estimator
%
% Shrinks towards constant correlation matrix
% if shrink is specified, then this constant is used for shrinkage

% de-mean returns
[t,n]=size(x);
meanx=mean(x);
x=x-meanx(ones(t,1),:);

% compute sample covariance matrix
sample=(1/t).*(x'*x);

% compute prior
var=diag(sample);
sqrtvar=sqrt(var);
rBar=(sum(sum(sample./(sqrtvar(:,ones(n,1)).*sqrtvar(:,ones(n,1))')))-n)/(n*(n-1));
prior=rBar*sqrtvar(:,ones(n,1)).*sqrtvar(:,ones(n,1))';
prior(logical(eye(n)))=var;


if (nargin < 2 | shrink == -1) % compute shrinkage parameters and constant
			       
  % what we call pi-hat
  y=x.^2;
  phiMat=y'*y/t - 2*(x'*x).*sample/t + sample.^2;
  phi=sum(sum(phiMat));
  
  % what we call rho-hat
  term1=((x.^3)'*x)/t;
  help = x'*x/t;
  helpDiag=diag(help);
  term2=helpDiag(:,ones(n,1)).*sample;
  term3=help.*var(:,ones(n,1));
  term4=var(:,ones(n,1)).*sample;
  thetaMat=term1-term2-term3+term4;
  thetaMat(logical(eye(n)))=zeros(n,1);
  rho=sum(diag(phiMat))+rBar*sum(sum(((1./sqrtvar)*sqrtvar').*thetaMat));
  
  % what we call gamma-hat
  gamma=norm(sample-prior,'fro')^2;
  
  % compute shrinkage constant
  kappa = (phi - rho)/gamma;
  shrinkage = max(0,min(1,kappa/t));
  
else % use specified constant
  shrinkage = shrink;
end

% compute the estimator
sigma=shrinkage*prior+(1-shrinkage)*sample;

 
	

