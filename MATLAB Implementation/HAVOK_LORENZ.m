%% Final HAVOK code 

% This code has been adapted from the original code provided along with the
% publication titled "Chaos as an intermittently forced linear system"
% authored by Steven Brunton et. al.
% Link : https://www.nature.com/articles/s41467-017-00030-8


clc, clear, close

%% CODE 1: Lorenz system generation

% Lorenz parameters
sigma = 10;
beta = 8/3;
rho = 28;
n = 3;
% Initial condition
x0 = [ -8; 8; 27];

dt = 0.001;
tspan = (dt:dt:200);
N=length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[~,xdat]=ode45(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,options);

%% CODE 2: Delay Hankel matrix and its decomposition

clear V, clear dV, clear H, clear U, clear S % not needed as we clear everything and run anyway.
stackmax=100; % The number of shift stacked rows
lambda = 0; % Threshold for sparse regression (use 0.02 to kill terms)
rmax=15; % Maximum singular vectors to include

% Delay embedding in the form of Hankel matrix
H=zeros(stackmax,size(xdat,1)-stackmax);
k=1;
while k<stackmax+1
    H(k,:)= xdat(k:length(xdat)-stackmax+k-1,1);
    k=k+1;
end

disp('Delay embedding created')

% SVD of Hankel matrix
[U,S,V] = svd(H,'econ');

% Optimum threshold calculations
sigs = diag(S); % storing all the sigmas
beta = size(H,1)/size(H,2); % # of rows/ # of columns
thresh = optimal_SVHT_coef(beta,0)*median(sigs); 
r = length(sigs(sigs>thresh)); % Upper bound of r that can be considered, Output is r=24
r = min(rmax,r);

display(strcat('optimal r value for the SVD is : ', num2str(r)))

%% CODE 3: Computing derivatives
% fourth order central difference

dV = zeros(length(V)-4,r); 
% need to remove only 4, why are 5 removed from length(V)?
% 
for i=3:length(V)-2 % should be 2, instead of 3
    for k=1:r
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end  
% concatenate
x = V(3:end-2,1:r);
dx = dV;

%%  CODE 4: BUILD HAVOK REGRESSION MODEL ON TIME DELAY COORDINATES
% This implementation uses the SINDY code, but least-squares works too
% Build library of nonlinear time series
polyorder = 1;
Theta = poolData(x,r,1,0);
% normalize columns of Theta (required in new time-delay coords)
for k=1:size(Theta,2)
    normTheta(k) = norm(Theta(:,k)); %#ok<SAGROW> 
    Theta(:,k) = Theta(:,k)/normTheta(k);
end 
m = size(Theta,2);
% compute Sparse regression: sequential least squares
% requires different lambda parameters for each column
clear Xi
for k=1:r-1
    Xi(:,k) = sparsifyDynamics(Theta,dx(:,k),lambda*k,1);  %#ok<SAGROW> % lambda = 0 gives better results 
end

for k=1:length(Xi)
    Xi(k,:) = Xi(k,:)/normTheta(k);
end
A = Xi(2:r+1,1:r-1)';
B = A(:,r);
A = A(:,1:r-1);
%
L = 1:50000;
sys = ss(A,B,eye(r-1),0*B);
[y,t] = lsim(sys,x(L,r),dt*(L-1),x(1,1:r-1));  

%% FIGURE 1: Attractor

% please enter the path to a folder, to store the figures
cd 'C:\MathMods\Semester 4\Thesis\Code\HAVOK\HAVOK_Modified\Figures\'

figure;
L = 1:200000;
plot3(xdat(L,1),xdat(L,2),xdat(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
axis on
view(-5,12)
axis tight
xlabel('x'), ylabel('y'), zlabel('z')
set(gca,'FontSize',14)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 2: Time series

figure
plot(tspan,xdat(:,1),'k','LineWidth',2)
xlabel('t'), ylabel('x')
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],'YTick',[-20 -10 0 10 20])
set(gcf,'Position',[100 100 550 300])
xlim([0 100])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 3: Embedded attractor

figure
L = 1:170000;
plot3(V(L,1),V(L,2),V(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(-13.508433026692551,64.193062918106762)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 4: Model time series

L = 300:25000;
L2 = 300:50:25000;
figure
subplot(2,1,1)
plot(tspan(L),x(L,1),'Color',[.4 .4 .4],'LineWidth',2.5)
hold on
plot(tspan(L2),y(L2,1),'.','Color',[0 0 .5],'LineWidth',5,'MarkerSize',15)
xlim([0 max(tspan(L))])
ylim([-.0051 .005])
ylabel('v_1')
box on
subplot(2,1,2)
plot(tspan(L),x(L,r),'Color',[.5 0 0],'LineWidth',1.5)
xlim([0 max(tspan(L))])
ylim([-.025 .024])
xlabel('t'), ylabel('v_{15}')
box on
set(gcf,'Position',[100 100 550 350])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 5: Reconstructed attractor

figure
L = 300:50000;
plot3(y(L,1),y(L,2),y(L,3),'Color',[0 0 .5],'LineWidth',1.5)
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(34,22)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 6: Forcing statistics

% old hist function not replaced properly, see the commented code for the
% update attempt

figure
Vtest = std(V(:,r))*randn(200000,1);
[h,hc] = hist(V(:,r)-mean(V(:,r)),[-.03:.0025:.03]);%[-.03  -.02 -.015  -.0125 -.01:.0025:.01 .0125  .015 .02  .03]);
[hnormal,hnormalc] = hist(Vtest-mean(Vtest),[-.02:.0025:.02]);
semilogy(hnormalc,hnormal/sum(hnormal),'--','Color',[.2 .2 .2],'LineWidth',1.5)
hold on
semilogy(hc,h/sum(h),'Color',[0.5 0 0],'LineWidth',1.5)
xlabel('v_{15}')
ylabel('P(v_{15})')
ylim([.0001 1])
xlim([-.025 .025])
legend('Normal Distribution','Lorenz Forcing')
set(gcf,'Position',[100 100 550 300])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 7: U modes

figure
CC = [2 15 32;
    2 35 92;
    22 62 149;
    41 85 180;
    83 124 213;
    112 148 223;
    114 155 215];
for k=1:5
    plot(U(:,k),'linewidth',1.5+2*k/30,'Color',CC(k,:)/255), hold on
end
plot(U(:,6),'Color',[.5 .5 .5],'LineWidth',1.5)
plot(U(:,15),'linewidth',1.5,'Color',[0.75 0 0])
plot(U(:,1:r),'Color',[.5 .5 .5],'LineWidth',1.5)
plot(U(:,15),'linewidth',1.5,'Color',[0.75 0 0])
hold on
for k=5:-1:1
    plot(U(:,k),'linewidth',1.5+2*k/30,'Color',CC(k,:)/255)
end
xlabel('time index, k'), ylabel('u_r')
l1=legend('r=1','r=2','r=3','r=4','r=5','...','r=15');
set(l1,'location','NorthWest')
set(gcf,'Position',[100 100 550 300])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 8: Prediction of lobe switching

% compute indices where forcing is "active"
L = 1:length(V);
inds = V(L,r).^2>4.e-6;
L = L(inds);
startvals = [];
endvals = [];
start = 1683;
clear interval hits endval newhit
numhits = 92; % Updated this value from 100 to 92, for the reason stated below.
 % This loop does not appear to be running till numhits=100, it stops at
 % 93, Updating loop to terminate before 100 if error appears
for k=1:numhits
    startvals = [startvals; start]; %#ok<AGROW> 
    endmax = start+500;
    interval = start:endmax;
    hits = find(inds(interval));
    endval = start+hits(end);
    endvals = [endvals; endval]; %#ok<AGROW> 
    newhit = find(inds(endval+1:end));
    start = endval+newhit(1);
end
% Color code attractor by whether or not forcing is active
figure
for k=1:numhits
    plot3(V(startvals(k):endvals(k),1),V(startvals(k):endvals(k),2),V(startvals(k):endvals(k),3),'r','LineWidth',1.5), hold on
end
for k=1:numhits-1
    plot3(V(endvals(k):startvals(k+1),1),V(endvals(k):startvals(k+1),2),V(endvals(k):startvals(k+1),3),'Color',[.25 .25 .25],'LineWidth',1.5), hold on
end
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(34,22)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 9: Plot prediction as time series 
figure
ax1=subplot(3,1,1);
plot(tspan(1:length(V)),V(:,1),'k'), hold on
for k=1:numhits
    plot(tspan(startvals(k):endvals(k)),V(startvals(k):endvals(k),1),'r','LineWidth',1.5), hold on
end
for k=1:numhits-1
    plot(tspan(endvals(k):startvals(k+1)),V(endvals(k):startvals(k+1),1),'Color',[.25 .25 .25],'LineWidth',1.5), hold on
end
ylabel('v_1')
ax2=subplot(3,1,2);
plot(tspan(1:length(V)),V(:,r),'k'), hold on
for k=1:numhits
    plot(tspan(startvals(k):endvals(k)),V(startvals(k):endvals(k),r),'r','LineWidth',1.5), hold on
end
for k=1:numhits-1
    plot(tspan(endvals(k):startvals(k+1)),V(endvals(k):startvals(k+1),r),'Color',[.25 .25 .25],'LineWidth',1.5), hold on
end
ylabel('v_{15}')
ax3=subplot(3,1,3);
plot(tspan(startvals(1)),V(startvals(1),r),'r'), hold on
plot(tspan(endvals(1)),V(startvals(2),r),'Color',[.25 .25 .25])
for k=1:numhits
    plot(tspan(startvals(k):endvals(k)),V(startvals(k):endvals(k),r).^2,'r','LineWidth',1.5), hold on
end
for k=1:numhits-1
    plot(tspan(endvals(k):startvals(k+1)),V(endvals(k):startvals(k+1),r).^2,'Color',[.25 .25 .25],'LineWidth',1.5), hold on
end
xlabel('t'), ylabel('v_{15}^2')
legend('Forcing Active','Forcing Inactive')
linkaxes([ax1,ax2,ax3],'x')
xlim([25 65])
set(gcf,'Position',[100 100 550 450])
set(gcf,'PaperPositionMode','auto')

%% FIGURE 10: Test integer model 

figure
r=15;
A = zeros(14);
A = A+diag([-5 -10 -15 -20 25 -30 -35 -40 45 -50 -55 60 -65],1);
A = A+diag([5 10 15 20 -25 30 35 40 -45 50 55 -60 65],-1);
B = zeros(14,1);
B(end) = -70;
sysNew = ss(A,B,sys.c,sys.d);
[y,t] = lsim(sysNew,V(:,r),dt*(1:length(V)),V(1,1:r-1));
L = 1:199900;
plot(tspan(L),V(L,1),'k','LineWidth',2), hold on
plot(tspan(L),y(L,1),'r','LineWidth',2)
set(gcf,'Position',[100 100 550 300])
set(gcf,'PaperPositionMode','auto')

%%  FIGURE 11: Skeleton of Lorenz system

figure
k=27;
L0 = endvals(k-1):startvals(k);
plot(V(L0,1),V(L0,2),'Color',[.25 .25 .25],'LineWidth',1.5), hold on
L1 = startvals(k):endvals(k);
plot(V(L1,1),V(L1,2),'r','LineWidth',1.5)
L2 = endvals(k):startvals(k+1);
plot(V(L2,1),V(L2,2),'Color',[.25 .25 .25],'LineWidth',1.5)
L3 = startvals(k+1):startvals(k+1)+100;
plot(V(L3,1),V(L3,2),'r--','LineWidth',1.5)
L0 = endvals(k-1):10:startvals(k+1);
xlabel('v_1'), ylabel('v_2')
axis tight
set(gcf,'Position',[100 100 550 350])
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

function coef = optimal_SVHT_coef(beta, sigma_known) 



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

n = size(yin,1);
% yout = zeros(n,1+nVars+(nVars*(nVars+1)/2)+(nVars*(nVars+1)*(nVars+2)/(2*3))+11);

ind = 1;
% poly order 0
yout(:,ind) = ones(n,1);
ind = ind+1;

% poly order 1
if(polyorder>=1)
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
% Theta ~ x (V)
% dXdt ~ dx (dV)
% Theta * Xi = dxdt

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