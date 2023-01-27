% HAVOK on Lorenz system - Lobe switching predictions when multiple forcing
% terms are used (2 consecutive forcing terms)

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
svht_test = 30; % upper bound of r till which rf analysis will run

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


dV = zeros(length(V)-5,svht_test);  
for i=3:length(V)-3
    for k=1:svht_test
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end

% concatenate
x = V(3:end-3,1:svht_test); 
dx = dV;
xtest=Vtest(3:end-3,1:svht_test); 

xtestfinal = [x ; xtest(length(x)+1:end,:)];

%% CODE 3: Actual lobe switching numbers

clear xpos, clear xswitch, clear minusplusswitch, clear plusminusswitch, clear lobeswitchact

for l=1:testset
    xpos(l,1)=sdata+l;
    if xdat(sdata+l,1)>0 % change column for different coordinates. 1=x-coordinate, 2=y-coordinate, 3=z-coordinate.
        xpos(l,2)=1;
    else
        xpos(l,2)=0;
    end
end
sprintf('xpos created')

c=2;
xswitch(1,2) = xpos(1,2);
xswitch(1,1) = xpos(1,1);
for j=2:length(xpos)
    if xpos(j,2) ~= xpos(j-1,2)
        xswitch(c,2) = xpos(j,2);
        xswitch(c,1) = xpos(j,1);
        c=c+1;
    end
end

sprintf('xswitch created')
plusminusswitch = 0;
minusplusswitch = 0;

for i=1:length(xswitch)-1
    if xswitch(i,2) == 1 && xswitch(i+1,2) ==0
        plusminusswitch = plusminusswitch + 1;
    elseif xswitch(i,2) == 0 && xswitch(i+1,2) ==1
        minusplusswitch = minusplusswitch + 1;
    end
end

lobeswitchact = plusminusswitch + minusplusswitch;

sprintf('Actual number of lobe switching = %d',lobeswitchact) % unique for a time period selected. r, f does not influence it. Obtained from only studying x plot.

figure
plot(xdat(sdata+1:testdata,1)) % plotting x from lorenz system. change column for different coordinates. 1=x-coordinate, 2=y-coordinate, 3=z-coordinate.

hold on
plot(zeros(length(sdata+1:testdata),1))

%% CODE 4: Lobeswithcing predicted by forcing term

f1=10; % required f1 can be entered here.
f2=11; % required f2 can be entered here.
fthresh = 0.00395; % threshold above which we label switching. This changes with change in the forcing term. Must be manually varied to obtain actual lobe switching numbers.
sweep_window = 500; % Sweep window has to be varied depending on the plot.

xtest2f = xtestfinal(:,f1)-xtestfinal(:,f2);

clear vlobeswitch
e = 1;
d = sdata+1-stackmax-5;
while d < testdata-stackmax-5 - 1
    if xtest2f(d) < fthresh && xtest2f(d+2) > fthresh
        vlobeswitch(e) = d;
        d=d+sweep_window;
        e=e+1;
        sprintf('value of d is %d',d)
    elseif xtest2f(d) > -fthresh && xtest2f(d+2) < -fthresh
        vlobeswitch(e) = d;
        d=d+sweep_window;
        e=e+1;
    else
        d=d+1;
    end
end
sprintf('Lobe switching predicted is %d',e-1)

% Coloring when forcing term exceeds threshold

figure
plot(xtest2f(sdata:testdata-stackmax-5,:),'k')
hold on
a = [(1:testset)' ones(testset,1)*fthresh];
b = [(1:testset)' ones(testset,1)*(-fthresh)];
plot(a(:,1),a(:,2),'k')
hold on
plot(b(:,1),b(:,2),'k')

g=1;
while g < length(vlobeswitch) + 1
    if vlobeswitch(g)+sweep_window < length(xtestfinal)
        plot(vlobeswitch(g)-sdata:vlobeswitch(g)+sweep_window-sdata,xtest2f(vlobeswitch(g):vlobeswitch(g)+sweep_window),'r')
    else
        plot(vlobeswitch(g)-sdata:length(xtestfinal)-sdata,xtest2f(vlobeswitch(g):end),'r')
    end
    g=g+1;
end
str = {strcat('f1 =',num2str(f1),' ; f2 = ',num2str(f2),' ; f1-f2',' ; Sweep window =',num2str(sweep_window),' ; Threshold = ',num2str(fthresh),' ; Lobe switching predicted =',num2str(e-1))};
title(str)
%axis([0 40000 0.025 -0.025]);

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