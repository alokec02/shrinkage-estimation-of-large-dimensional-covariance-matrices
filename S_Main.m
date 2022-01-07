clear all
close all
% M file for downloading data from Yahoo! Finance, from the Mathworks site.
% Weekly price data for the DJIA (Dow 30) constituents as of August, 2009 for the 10
% year period from August 31, 1999 to August 31, 2009
Data=hist_stock_data('31081999', '31082009', 'djia.txt', 'frequency', 'w');

% Kraft had an IPO on June 12, 2001. It is excluded from the sample.  
Data([20]) = [];

for i=1:29
DataAdj(:,i)=Data(i).AdjClose(end:-1:1);
end

% This code follows the approach taken by Ledoit and Wolf to estimate the covariance
% matrix of returns data using shrinkage estimators. The shrinkCorr.m
% and shrinkMarket.m files are available on Dr. Ledoit's website, www.ledoit.net. 
% shrinkIdentity.m is adapted from the other two files. 

%In Meucci 4.4, we see that while the bias is quite limited for
% benchmark estimators, the inefficiency is high, contributing the bulk of
% the error. The Markowitz mean variance optimization (MVO) is implemented 
% with quadprog in the Matlab Optimization toolbox.

% In Ledoit and Wolf (2003), the authors examine several approaches, primarily the ones below:
% a) Shrinkage to Identity
% b) Shrinkage to Constant Correlation 
% c) Shrinkage to Market Model (Sharpe)
% d) Sample covariance matrix (i.e. no shrinkage)
% 
%partitions ten years of data into an in sample and out of sample period 
Data=xlsread('Data2.xls');
InSample=Data(1:261,:);
OutSample=Data(262:end,:);
% Linear returns (could also use compound returns, preferred with more frequent
% sampling 
InSampleRet=(InSample(2:end,:)-InSample(1:end-1,:))./InSample(1:end-1,:);
OutSampleRet=(OutSample(2:end,:)-OutSample(1:end-1,:))./OutSample(1:end-1,:);
% sample mean and sample covariance:
r=mean(InSampleRet)';
s_bar_in=cov(InSampleRet);
s_bar_out_ev=cov(OutSampleRet);

%PCA
[U,S,V]=svd(Data);
Y=U*S;
a1=Y(:,1)*V(:,1)';
a2=Y(:,2)*V(:,2)';
a3=Y(:,3)*V(:,3)';
a4=Y(:,4)*V(:,4)';
a5=Y(:,5)*V(:,5)';
[Ae, Be]=eig(s_bar_in);



% MVO framework,inequality condition, no short sales and gross exposure
% constraints, <10% in each stock, Meucci 6.2
dimvector=size(InSample);
TotObs=dimvector(1);
NumAssets=dimvector(2);
AEq=[ones(1,NumAssets)];
BEq=1;
A=[-eye(NumAssets)
eye(NumAssets)];
b=[zeros(NumAssets,1)
0.10*ones(NumAssets,1)];
f=zeros(NumAssets,1);

% Shrinkage toward the identity matrix
[s_identity shr_identity]=shrinkIdentity(InSampleRet);
Identity_MV=quadprog(s_identity,-f,A,b,AEq,BEq,[],[]);
Identity_var=Identity_MV'*s_identity*Identity_MV;

% Constant Correlation
[s_ConstCorr shr_ConstCorr]=shrinkCorr(InSampleRet);
ConstCorr_MV=quadprog(s_ConstCorr,-f,A,b,AEq,BEq,[],[]);
ConstCorr_var=ConstCorr_MV'*s_ConstCorr*ConstCorr_MV;

% Market Model (Sharpe)
[s_Sharpe shr_Sharpe]=shrinkMarket(InSampleRet);
Sharpe_MV=quadprog(s_Sharpe,-f,A,b,AEq,BEq,[],[]);
Sharpe_var=Sharpe_MV'*s_Sharpe*Sharpe_MV;

% Sample covariance 
Sample_MV=quadprog(s_bar_in,-f,A,b,AEq,BEq,[],[]);
Sample_var=Sample_MV'*s_bar_in*Sample_MV;

% condition number gives an indication of the accuracy of the results from
% a matrix inversion and system of linear equations- lower values are considered
% more stable.

% cond(s_identity)=111.4
% cond(s_ConstCorr)=65.90
% cond(s_Sharpe)=111.90
% cond(s_bar_in)~10^8

fprintf(1, [''])
fprintf(1, ['The minimum variance portfolios had the following in-sample standard deviations:\n'])
fprintf(1, ['\n'])
disp( ['Shrinkage to Identity: ' num2str(sqrt(Identity_var)*sqrt(52))])
disp( ['Shrinkage to Constant Correlation: ' num2str(sqrt(ConstCorr_var)*sqrt(52))])
disp( ['Shrinkage to Market Model: ' num2str(sqrt(Sharpe_var)*sqrt(52))])
disp( ['Sample Covariance: ' num2str(sqrt(Sample_var)*sqrt(52))])
fprintf(1, [' \n'])

% performance measures in the out of sample period (results displayed below) 
posIdentity=find(Identity_MV>0.00001);
Identity_MV_2=Identity_MV(posIdentity);
ptsIdentity=OutSample(:,posIdentity);
retIdentity=(ptsIdentity(2:end,:)-ptsIdentity(1:end-1,:))./ptsIdentity(1:end-1,:);
Identity_std=std(retIdentity*Identity_MV_2);

posCorr=find(ConstCorr_MV>0.00001);
ConstCorr_MV_2=ConstCorr_MV(posCorr);
ptsCorr=OutSample(:,posCorr);
retCorr=(ptsCorr(2:end,:)-ptsCorr(1:end-1,:))./ptsCorr(1:end-1,:);
Corr_std=std(retCorr*ConstCorr_MV_2);

posSharpe=find(Sharpe_MV>0.00001);
Sharpe_MV_2=Sharpe_MV(posSharpe);
ptsSharpe=OutSample(:,posSharpe);
retSharpe=(ptsSharpe(2:end,:)-ptsSharpe(1:end-1,:))./ptsSharpe(1:end-1,:);
Market_std=std(retSharpe*Sharpe_MV_2);

posSample=find(Sample_MV>0.00001);
Sample_MV_2=Sample_MV(posSample);
ptsSample=OutSample(:,posSample);
retSample=(ptsSample(2:end,:)-ptsSample(1:end-1,:))./ptsSample(1:end-1,:);
Sample_std=std(retSample*Sample_MV_2);

fprintf(1, ['\n'])
fprintf(1, ['MV portfolios during the out-of-sample period: \n'])
fprintf(1, [' Standard Deviations\n'])
fprintf(1, [' \n'])
disp( ['Shrinkage to Identity: ' num2str(Identity_std*sqrt(52))])
disp( ['Shrinkage to Constant Correlation: ' num2str(Corr_std*sqrt(52))])
disp( ['Shrinkage to Market Model: ' num2str(Market_std*sqrt(52))])
disp( ['Sample Covariance: ' num2str(Sample_std*sqrt(52))])
disp( [' '])




% MVO with quarterly rebalancing
Data=xlsread('Data2.xls');
%assignes time period between rebalancings
fwdinterval=52;
interval=12;
Inc=10;
ret_Sharpe=[];
ret_ConstCorr=[];
ret_Identity=[];
ret_Sample=[];
%Starts the reblancing portfolio
for i=1:Inc
%Picks out the relevant data for the period
if i==Inc
InSample=Data(1+interval*(i-1):1+fwdinterval+interval*(i-1),:);
OutSample=Data(1+fwdinterval+interval*(i-1):fwdinterval+interval*(i),:);
else
InSample=Data(1+interval*(i-1):1+fwdinterval+interval*(i-1),:);
OutSample=Data(1+fwdinterval+interval*(i-1):1+fwdinterval+interval*(i),:);
end
% returns and covariance
InSampleRet=(InSample(2:end,:)-InSample(1:end-1,:))./InSample(1:end-1,:);
OutSampleRet=(OutSample(2:end,:)-OutSample(1:end-1,:))./OutSample(1:end-1,:);
s_bar_in=cov(InSampleRet);
dimvector=size(InSample);
TotObs=dimvector(1);
NumAssets=dimvector(2);
AEq=[ones(1,NumAssets)];
BEq=1;
% MVO constraints- <10%, no shorting
A=[-eye(NumAssets)
eye(NumAssets)
];
b=[zeros(NumAssets,1)
0.10*ones(NumAssets,1)];
f=(zeros(NumAssets,1));
[s_Sharpe shr_Sharpe]=shrinkMarket(InSampleRet);
Sharpe_MV=quadprog(s_Sharpe,-f,A,b,AEq,BEq,[],[]);

[s_ConstCorr shr_ConstCorr]=shrinkCorr(InSampleRet);
ConstCorr_MV=quadprog(s_ConstCorr,-f,A,b,AEq,BEq,[],[]);

[s_identity shr_identity]=shrinkIdentity(InSampleRet);
Identity_MV=quadprog(s_identity,-f,A,b,AEq,BEq,[],[]);

Sample_MV=quadprog(s_bar_in,-f,A,b,AEq,BEq,[],[]);

OutSample_ret=(OutSample(2:end,:)-OutSample(1:end-1,:))./OutSample(1:end-1,:);
ret_Sharpe=[ret_Sharpe; OutSample_ret*Sharpe_MV];
ret_ConstCorr=[ret_ConstCorr; OutSample_ret*ConstCorr_MV];
ret_Identity=[ret_Identity; OutSample_ret*Identity_MV];
ret_Sample=[ret_Sample; OutSample_ret*Sample_MV];
end
% Evaluation of returns of the min-var portfolios:
stdMarket=std(ret_Sharpe)*sqrt(52);
stdCorr=std(ret_ConstCorr)*sqrt(52);
stdIdentity=std(ret_Identity)*sqrt(52);
stdSample=std(ret_Sample)*sqrt(52);

fprintf(1, ['\n'])
fprintf(1, ['MV portfolios with periodic rebalancing on out-of-sample data:\n'])
fprintf(1, ['\n'])
fprintf(1, [' Standard Deviations\n'])
fprintf(1, [' \n'])
disp( ['Shrinkage to Identity: ' num2str(stdIdentity)])
disp( ['Shrinkage to Market Model: ' num2str(stdMarket)])
disp( ['Shrinkage to Constant Correlation: ' num2str(stdCorr)])
disp( ['Sample Covariance: ' num2str(stdSample)])
fprintf(1, [' \n'])

    % this section examines tracking error constrained active portfolios-
    % Jorion (2003), and a comparison of the standard deviation of the MV
    % portfolio to rhe portfolios described above. The covariance matrix
    % used is rhe sample covariance, but the methodology could accomodate
    % shrinkage ewstimators easily as well.
    % Jorion finds TEV-constrained portfolios are described by an ellipse 
    % on the traditional mean–variance plane. 

    %p is the matrix of assets returns which are in the benchmark
    %q is the composition of the benchmark (weights)
    %T1 is the period on which we compute the data (observations)
    %T2 is the frequency of rebalancing of the portfolio (convention is 12 weeks)
    %var_x is parameter T, the variance we want for tracking error
    %var_P is the variance we want for the global portfolio (benchmark +
    %deviation portfolio)
    %percentage is the percentage for the VAR computation
    %nb_days_VAR is the number of days we want to compute VAR
   
    % as before, loading the Excel data
    Data=xlsread('Data2.xls');
InSample=Data(1:261,:);
OutSample=Data(262:end,:);
% Linear returns (could also use compound returns, preferred with more frequent
% sampling 
InSampleRet=(InSample(2:end,:)-InSample(1:end-1,:))./InSample(1:end-1,:);
OutSampleRet=(OutSample(2:end,:)-OutSample(1:end-1,:))./OutSample(1:end-1,:);
% sample mean and sample covariance:
q=mean(InSampleRet)';
s_bar_in=cov(InSampleRet);
s_bar_out_ev=cov(OutSampleRet);

    % Dow Jones is price weighted, component weights as of 8/09
    p=InSample;
q=[0.059492116
0.010161324
0.023890085
0.02202733
0.012361941
0.037095253
0.037636081
0.059889017
0.018590797
0.042861612
0.026039007
0.058
0.011
0.0357
0.0214
0.0159
0.0972
0.0502
0.0318
0.0454
0.0247
0.0194
0.0131
0.0457
0.0355
0.0449
0.0264
0.0411
0.204];
 
% T1 and T2 are user-defined, perhaps there are other results that could be
% better. To parallel the earlier MVO rebalancing, T2 is 12 weeks
% (approximately quarterly).
    T1=60;
    T2=12;
    q=q/sum(q); 
    r = diff(p)./p(1:end-1,:);
    rb=r(:,1:end)*q;
    %rb are the benchmark returns
    [nb_days,nb_assets] = size(r);
    nb_steps = floor((nb_days-T1)/T2);
    o=ones(nb_assets,1);
    return2 = zeros(T1,1);
    returnq = zeros(T1,1);
    return3 = zeros(T1,1);
    X2 = zeros(T1+1,nb_assets);
    X3 = zeros(T1+1,nb_assets);
    A = zeros(T1+1,1);
    for i=1:(nb_steps+1)
        
        covariance_matrix = cov(r((1+T2*(i-1)):(T2*(i-1)+T1),1:end));
        Expected_returns = (mean(r((1+T2*(i-1)):(T2*(i-1)+T1),1:end)))';%transpose
        % returnq = [returnq; p((T2*(i-1)+T1+1:(T2*i+T1)),2:end)*q];
        %computation of MV portfolio, following Merton (1972) as described
        %in Jorion (2003)
        a=Expected_returns'*inv(covariance_matrix)*Expected_returns;
        b=Expected_returns'*inv(covariance_matrix)*o;
        c=o'*inv(covariance_matrix)*o;%Expected_returns;
        if (a-b*b/c>0)
            e=a-b*b/c;
        else
            e=-1;
        end
        d=abs(a-b*b/c);
        mu_MV = b/c;
        var_MV = 1/c;
        
        %computation of the tracking error constrained portfolio (Jorion, 2003): 
        % the optimization problem in excess return space. 
        % One can trace out the tracking-error frontier by maximizing 
        % the expected excess return, ? = x?E, subject to a 
        % fixed amount of tracking error, T = x?Vx, and x?1 = 0
        
        var_x=.10;
        x2 = sqrt(var_x/d)*inv(covariance_matrix)*(Expected_returns - b/c*o);
        cont(i)=sum(x2);
        if (1+T2*(i-1))>522
        0
        end
        %scaling coefficient
        sc_coeff=.01;
        if(i < (nb_steps+1))
            A = [A;ones(T2,1)*e];
            X2 = [X2;ones(T2,1)*x2'];
            return2 = [return2 ; r((T2*(i-1)+T1+1:(T2*i+T1)),1:end)*x2*sc_coeff];
            returnq = [returnq ; r((T2*(i-1)+T1+1:(T2*i+T1)),1:end)*q];    
        else
            A = [A;ones(length(r)-(T2*(i-1)+T1),1)*e];
            return2 = [return2 ; r((T2*(i-1)+T1+1:end),1:end)*x2*sc_coeff];
            returnq = [returnq ; r((T2*(i-1)+T1+1:end),1:end)*q];
            X2 = [X2;ones(length(r)-(T2*(i-1)+T1),1)*x2'];
        end 
%         returnq = [returnq ; r((T2*(i-1)+T1+1:(T2*i+T1)),2:end)*q];
        %computation of tracking error portfolio controlling for total
        %volatility  
         mu_B = q'*Expected_returns;
        var_B = q'*covariance_matrix*q;
        var_P=.20;
        y = var_P - var_B - var_x;
        delta1 = mu_B - b/c;
        delta2 = var_B - 1/c;
        lambda3 = -delta1/delta2 + y/delta2*sqrt(abs((d*delta2-delta1^2)/(4*var_x*delta2-y^2)));
        lambda1 = - (lambda3 + b)/c;
        lambda2=-lambda3-2*sqrt(abs((d*delta2-delta1^2)/(4*var_x*delta2-y^2)));
        x3 = -1/(lambda2+lambda3)*inv(covariance_matrix)*(Expected_returns + lambda1*o + lambda3*covariance_matrix*q);
        test2(i) = sum(x3);
        X3 = [X3;ones(T2,1)*x3'];
        if (i<nb_steps+1)
            return3 = [return3 ; r((T2*(i-1)+T1+1:(T2*i+T1)),1:end)*x3*sc_coeff];
            X3 = [X3;ones(T2,1)*x3'];
        else
            return3 = [return3 ; r((T2*(i-1)+T1+1:end),1:end)*x3*sc_coeff];
            X3 = [X3;ones(length(r)-(T2*(i-1)+T1),1)*x3'];
        end
        testvalue(i)=(d*delta2-delta1^2)/(4*var_x*delta2-y^2);
    end
        
% Sharpe Ratio calculations, taux is the risk free rate (or alternatively, benchmark return adjusting the SD in the denominator) 
%(assumed to be 2.9% here, the 5 Yr CD rate on bankrate.com) 
 
        taux=.029;        
        SR2 = mean(return2+returnq-taux/(261))/sqrt(var(return2+returnq))*sqrt(52); 
        SR3 = mean(return3+returnq-taux/(261))/sqrt(var(return3+returnq))*sqrt(52);
        SRtot2 = mean(return2+returnq-taux/(261))/sqrt(var(return2+returnq))*sqrt(52);
        SRb=mean(returnq-taux/(26100))/sqrt(var(+returnq))*sqrt(52);
        SR=[SRb,SR2,SR3];
        vol=[sqrt(var(returnq))*sqrt(52),sqrt(var(return2+returnq))*sqrt(52),sqrt(var(return3+returnq))*sqrt(52)];
        return_med = [mean(returnq)*261,mean(return2)*261,mean(return3)*261];
        return_med_tot = [mean(returnq)*(261),mean(return2+returnq)*(261),mean(return3+returnq)*(261)];
        covariance_matrix = cov(r((1+T2*(i-1)):(T2*(i-1)+T1),1:end));
        v=q'*covariance_matrix*q;