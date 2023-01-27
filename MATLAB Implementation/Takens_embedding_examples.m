%% Takens State Space Reconstructions on the Lorenz Attractor

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
Var=1;% change the value. 1=x-reconstructions, 2=y-reconstructions, 3=z-reconstructions 

k=1;
while k<stackmax+1
    H(k,:)= xdat(k:length(xdat)-stackmax+k-1,Var); 
    k=k+1;
end

disp('Delay embedding created')

% SVD of Hankel matrix
[U,S,V] = svd(H,'econ');

%% FIGURE 1: Attractor

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

%% FIGURE 3: Embedded attractor

figure
L = 1:170000;
plot3(V(L,1),V(L,2),V(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(-5,80)
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
