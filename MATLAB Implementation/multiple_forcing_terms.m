%% HAVOK model with multiple forcing terms - This implementation is for two consecutive forcing terms, and can be adapted for more forcing terms.

clear
close
clc

%% CODE 1: Lorenz system generation

% Lorenz parameters
sigma = 10;
beta = 8/3;
rho = 28;
n = 3;
% Initial condition
x0 = [ -8; 8; 27];

dt = 0.001;
tspan = (dt:dt:300);
%N=length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,xdat]=ode45(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,options);  


lambda = 0; % Threshold for sparse regression (use 0.02 to kill terms)
rmax=15; % Maximum singular vectors to include


%% CODE 2: Delay Hankel matrix and its decomposition
clear V, clear dV, clear H
clear Vfull, clear Hfull % dV is only needed to run the linear system. In our case, we calculate Vfull, to obtain a benchmark


% Parameters that can be varied

stackmax=100; % The number of shift stacked rows
% Training dataset
sdata = 200000; % amount of data used for embedding, vary this to simulate varied input.
% Test dataset
testset=40000;
testdata = testset+sdata; % Includes additional timesteps used to create Vtest, as this will be used for simulation.
% Simulation intervals
L = 1:sdata+testset-stackmax-5; % simulation interval ; 5 removed due to 4th order central difference
Ltest = sdata-stackmax-5:sdata+testset-stackmax-5; % error testing interval
svht_test = 30;

% partial delay matrix
H=zeros(stackmax,sdata-stackmax);
k=1;
while k<stackmax+1
    H(k,:)= xdat(k:sdata-stackmax+k-1,1);
    k=k+1;
end

disp('train Delay Hankel created')

% SVD of Hankel matrix
[U,S,V] = svd(H,'econ');

% partial delay matrix
Htest=zeros(stackmax,testdata-stackmax);
k=1;
while k<stackmax+1
    Htest(k,:)= xdat(k:testdata-stackmax+k-1,1);
    k=k+1;
end

disp('test Delay Hankel created')

% SVD of Hankel matrix
[Utest,Stest,Vtest] = svd(Htest,'econ');


dV = zeros(length(V)-5,svht_test);  % changed from r columns to 24 columns
for i=3:length(V)-3
    for k=1:24
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end

% concatenate
x = V(3:end-3,1:svht_test); % changed from 1:r to 1:24
dx = dV;
xtest=Vtest(3:end-3,1:svht_test); % changed from 1:r to 1:24

xtestfinal = [x ; xtest(length(x)+1:end,:)];

% Optimum svht threshold calculations

sigs = diag(S); % storing all the sigmas
beta = size(H,1)/size(H,2); % # of rows/ # of columns
thresh = optimal_SVHT_coef(beta,0)*median(sigs); % Output is r=24
r = length(sigs(sigs>thresh)); % Upper bound of r that can be considered
sprintf('The value of r is %d',r)

%% CODE 3: Varying forcing term, for each r. Table of reconstruction errors.

% for each set of right singular values considered, we vary the forcing
% term f. 
tic
r=5; % to be set by the user. Run the loop according to desired r,f
c=1;
error=zeros((svht_test-r)*(svht_test-r-1),6); % size of error matrix depends on the number of r,f tested. Needs to be updated as required.

while r<30

    % Phase 2
    polyorder = 1;

    for f=4:29 % Multiple f corresponds to consecutive forcing terms. f=4 would mean the 4th and 5th right singular vector are used for forcing.

        Theta = poolData(x,max(f+1,r),1,0);
        normTheta=zeros(1,size(Theta,2));

        for k=1:size(Theta,2)
            normTheta(k) = norm(Theta(:,k));
            Theta(:,k) = Theta(:,k)/normTheta(k);
        end
        
        if f+1<r
            clear Xi
            Xi = zeros(r+1,r-2); % pre allocating Xi
            for k=1:r-2
                if k>=f
                    j=k+2;
                else
                    j=k;
                end
                Xi(:,k) = sparsifyDynamics(Theta,dx(:,j),lambda*j,1);
            end

            for k=1:length(Xi)
                Xi(k,:) = Xi(k,:)/normTheta(k);
            end

            A = Xi(2:r+1,1:r-2)';
            B = A(:,f:f+1);
            A = [A(:,1:f-1) A(:,f+2:r)];


            sys = ss(A,B,eye(r-2),0*B);
            [y,t] = lsim(sys,xtestfinal(L,f:f+1),dt*(L-1),[x(1,1:f-1) x(1,f+2:r)]);

            % error on test data
            error(c,1)=r;
            error(c,2)=f;
            error(c,3)=f+1;
            error(c,4)=norm(xtestfinal(Ltest,1)-y(Ltest,1)); % L2 norm of V1 error
            diff=[xtestfinal(Ltest,1)-y(Ltest,1),xtestfinal(Ltest,2)-y(Ltest,2),xtestfinal(Ltest,3)-y(Ltest,3)];
            error(c,5)=norm(diff,2); % L2 norm of V1-3 error
            error(c,6)=norm(diff,"fro"); % Frobenius norm of V1-3 error
            c=c+1;
            sprintf('r = %d',r)
            sprintf('f1 = %d and f2 = %d',f,f+1)

        else
            clear Xi
            Xi = zeros(r+1,r-2); % pre allocating Xi
            for k=1:r-2
                Xi(:,k) = sparsifyDynamics([Theta(:,1:r-1) Theta(:,f+1:f+2)],dx(:,k),lambda*k,1);   % lambda = 0 gives better results
            end

            for k=1:length(Xi)
                Xi(k,:) = Xi(k,:)/normTheta(k);
            end

            A = Xi(2:r+1,1:r-2)';
            B = A(:,r-1:r);
            A = A(:,1:r-2);
            sys = ss(A,B,eye(r-2),0*B);
            [y,t] = lsim(sys,xtestfinal(L,f:f+1),dt*(L-1),x(1,1:r-2));

            % error on test data
            error(c,1)=r;
            error(c,2)=f;
            error(c,3)=f+1;
            error(c,4)=norm(xtestfinal(Ltest,1)-y(Ltest,1)); % L2 norm of V1 error
            diff=[xtestfinal(Ltest,1)-y(Ltest,1),xtestfinal(Ltest,2)-y(Ltest,2),xtestfinal(Ltest,3)-y(Ltest,3)];
            error(c,5)=norm(diff,2); % L2 norm of V1-3 error
            error(c,6)=norm(diff,"fro"); % Frobenius norm of V1-3 error
            c=c+1;
            sprintf('r = %d',r)
            sprintf('f1 = %d and f2 = %d',f,f+1)
        end
    end
    r=r+1;
end
toc

%% CODE 4: r=5, f1=4 and f2=5 reconstruction
tic
r=5; 

while r<6

    % Phase 2
    polyorder = 1;

    for f=4:4 % Multiple f corresponds to consecutive forcing terms. f=4 would mean the 4th and 5th right singular vector are used for forcing.

        Theta = poolData(x,max(f+1,r),1,0);
        normTheta=zeros(1,size(Theta,2));

        for k=1:size(Theta,2)
            normTheta(k) = norm(Theta(:,k));
            Theta(:,k) = Theta(:,k)/normTheta(k);
        end
        
        if f+1<r
            clear Xi
            Xi = zeros(r+1,r-2); % pre allocating Xi
            for k=1:r-2
                if k>=f
                    j=k+2;
                else
                    j=k;
                end
                Xi(:,k) = sparsifyDynamics(Theta,dx(:,j),lambda*j,1);
            end

            for k=1:length(Xi)
                Xi(k,:) = Xi(k,:)/normTheta(k);
            end

            A = Xi(2:r+1,1:r-2)';
            B = A(:,f:f+1);
            A = [A(:,1:f-1) A(:,f+2:r)];


            sys = ss(A,B,eye(r-2),0*B);
            [y,t] = lsim(sys,xtestfinal(L,f:f+1),dt*(L-1),[x(1,1:f-1) x(1,f+2:r)]);

            sprintf('r = %d',r)
            sprintf('f1 = %d and f2 = %d',f,f+1)

        else
            clear Xi
            Xi = zeros(r+1,r-2); % pre allocating Xi
            for k=1:r-2
                Xi(:,k) = sparsifyDynamics([Theta(:,1:r-1) Theta(:,f+1:f+2)],dx(:,k),lambda*k,1);   % lambda = 0 gives better results
            end

            for k=1:length(Xi)
                Xi(k,:) = Xi(k,:)/normTheta(k);
            end

            A = Xi(2:r+1,1:r-2)';
            B = A(:,r-1:r);
            A = A(:,1:r-2);
            sys = ss(A,B,eye(r-2),0*B);
            [y,t] = lsim(sys,xtestfinal(L,f:f+1),dt*(L-1),x(1,1:r-2));

            sprintf('r = %d',r)
            sprintf('f1 = %d and f2 = %d',f,f+1)
        end
    end
    r=r+1;
end
toc

%% FIGURE 1: Reconstructed attractor
figure
L = 300:50000;
plot3(y(L,1),y(L,2),y(L,3),'Color',[0 0 .5],'LineWidth',1.5)
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(-13.508433026692551,64.193062918106762)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')

%% FUNCTION 1: Lorenz function
function dy = lorenz(t,y,sigma,beta,rho) %#ok<INUSL> 
        % Copyright 2015, All Rights Reserved
        % Code by Steven L. Brunton
        % For Paper, "Discovering Governing Equations from Data: 
        %        Sparse Identification of Nonlinear Dynamical Systems"
        % by S. L. Brunton, J. L. Proctor, and J. N. Kutz

dy = [
sigma*(y(2)-y(1));
y(1)*(rho-y(3))-y(2);
y(1)*y(2)-beta*y(3);
];
end

%% FUNCTION 2: Optimal Singular value hard threshold coefficient

function coef = optimal_SVHT_coef(beta, sigma_known) 
% function omega = optimal_SVHT_coef(beta, sigma_known)
%
% Coefficient determining optimal location of Hard Threshold for Matrix
% Denoising by Singular Values Hard Thresholding when noise level is known or
% unknown.  
%
% See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
% Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870
%
% IN: 
%    beta: aspect ratio m/n of the matrix to be denoised, 0<beta<=1. 
%          beta may be a vector 
%    sigma_known: 1 if noise level known, 0 if unknown
% 
% OUT: 
%    coef:   optimal location of hard threshold, up the median data singular
%            value (sigma unknown) or up to sigma*sqrt(n) (sigma known); 
%            a vector of the same dimension as beta, where coef(i) is the 
%            coefficient correcponding to beta(i)
%
% Usage in known noise level:
%
%   Given an m-by-n matrix Y known to be low rank and observed in white noise 
%   with mean zero and known variance sigma^2, form a denoised matrix Xhat by:
%
%   [U D V] = svd(Y);
%   y = diag(Y);
%   y( y < (optimal_SVHT_coef(m/n,1) * sqrt(n) * sigma) ) = 0;
%   Xhat = U * diag(y) * V';
% 
%
% Usage in unknown noise level:
%
%   Given an m-by-n matrix Y known to be low rank and observed in white
%   noise with mean zero and unknown variance, form a denoised matrix 
%   Xhat by:
%  
%   [U D V] = svd(Y); 
%   y = diag(Y); 
%   y( y < (optimal_SVHT_coef_sigma_unknown(m/n,0) * median(y)) ) = 0; 
%   Xhat = U * diag(y) * V';
% 
% -----------------------------------------------------------------------------
% Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013
% 
% This program is free software: you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
% details.
%
% You should have received a copy of the GNU General Public License along with
% this program.  If not, see <http://www.gnu.org/licenses/>.
% -----------------------------------------------------------------------------


if sigma_known == 1 % sigma_known = 0 if noise level is unknown, sigma_known = 1 if noise level known
    coef = optimal_SVHT_coef_sigma_known(beta);
else
    coef = optimal_SVHT_coef_sigma_unknown(beta);
end
end

function lambda_star = optimal_SVHT_coef_sigma_known(beta) % separate function for known noise levels
assert(all(beta>0)); % making sure beta is positive (The ratio rows/columns of the Hankel matrix)
assert(all(beta<=1)); % making sure beta is less than or equal to 1 (Skinny matrix)
assert(numel(beta) == length(beta)); % to make sure beta is a vector

w = (8 * beta) ./ (beta + 1 + sqrt(beta.^2 + 14 * beta + 1));
lambda_star = sqrt(2 * (beta + 1) + w);
end

function omega = optimal_SVHT_coef_sigma_unknown(beta) 
    assert(all(beta>0));
    assert(all(beta<=1));
    assert(numel(beta) == length(beta));
    
    coef = optimal_SVHT_coef_sigma_known(beta);
    
    MPmedian = zeros(size(beta));
    i=1;
    while i<length(beta)+1
        MPmedian(i) = MedianMarcenkoPastur(beta(i));
        i=i+1;
    end
    omega = coef ./ sqrt(MPmedian);
end


function I = MarcenkoPasturIntegral(x,beta) %#ok<DEFNU> 
    if beta <= 0 || beta > 1
        error('beta beyond')
    end
    lobnd = (1 - sqrt(beta))^2;
    hibnd = (1 + sqrt(beta))^2;
    if (x < lobnd) || (x > hibnd)
        error('x beyond')
    end
    dens = @(t) sqrt((hibnd-t).*(t-lobnd))./(2*pi*beta.*t);
    I = integral(dens,lobnd,x);
    fprintf('x=%.3f,beta=%.3f,I=%.3f\n',x,beta,I);
end


function med = MedianMarcenkoPastur(beta)
    MarPas = @(x) 1-incMarPas(x,beta,0);
    lobnd = (1 - sqrt(beta))^2;
    hibnd = (1 + sqrt(beta))^2;
    change = 1;
    while change && (hibnd - lobnd > .001)
      change = 0;
      x = linspace(lobnd,hibnd,5);
      for i=1:length(x)
          y(i) = MarPas(x(i)); %#ok<AGROW> 
      end
      if any(y < 0.5)
         lobnd = max(x(y < 0.5));
         change = 1;
      end
      if any(y > 0.5)
         hibnd = min(x(y > 0.5));
         change = 1;
      end
    end
    med = (hibnd+lobnd)./2;
end

function I = incMarPas(x0,beta,gamma)
    if beta > 1
        error('betaBeyond');
    end
    topSpec = (1 + sqrt(beta))^2;
    botSpec = (1 - sqrt(beta))^2;
    MarPas = @(x) IfElse((topSpec-x).*(x-botSpec) >0, ...
                         sqrt((topSpec-x).*(x-botSpec))./(beta.* x)./(2 .* pi), ...
                         0);
    if gamma ~= 0
       fun = @(x) (x.^gamma .* MarPas(x));
    else
       fun = @(x) MarPas(x);
    end
    I = integral(fun,x0,topSpec);
    
    function y=IfElse(Q,point,counterPoint)
        y = point;
        if any(~Q)
            if length(counterPoint) == 1
                counterPoint = ones(size(Q)).*counterPoint;
            end
            y(~Q) = counterPoint(~Q);
        end
        
    end
end

%% FUNCTION 3: pool data function
function yout = poolData(yin,nVars,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

n = size(yin,1); % candidate fundtions have to be this long
% yout = zeros(n,1+nVars+(nVars*(nVars+1)/2)+(nVars*(nVars+1)*(nVars+2)/(2*3))+11);

ind = 1;
% poly order 0
yout(:,ind) = ones(n,1);
ind = ind+1;

if(polyorder>=1)
    % poly order 1
    for i=1:nVars
        yout(:,ind) = yin(:,i);
        ind = ind+1;
    end
end

if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout(:,ind) = yin(:,i).*yin(:,j);
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                ind = ind+1;
            end
        end
    end
end

if(polyorder>=4)
    % poly order 4
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l);
                    ind = ind+1;
                end
            end
        end
    end
end

if(polyorder>=5)
    % poly order 5
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    for m=l:nVars
                        yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l).*yin(:,m);
                        ind = ind+1;
                    end
                end
            end
        end
    end
end

if(usesine)
    for k=1:10
        yout = [yout sin(k*yin) cos(k*yin)]; %#ok<AGROW> 
    end
end
end

%% FUNCTION 4: Sparsify dynamics

function Xi = sparsifyDynamics(Theta,dXdt,lambda,n)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

% compute Sparse regression: sequential least squares
Xi = Theta\dXdt;  % initial guess: Least-squares

% lambda is our sparsification knob.
for k=1:10
    smallinds = (abs(Xi)<lambda);   % find small coefficients
    Xi(smallinds)=0;                % and threshold
    for ind = 1:n                   % n is state dimension
        biginds = ~smallinds(:,ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind); 
    end
end
end