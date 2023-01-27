% HAVOK on Lorenz system - Lobe switching predictions when single forcing
% term is used.

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
clear Vfull, clear Hfull % dV is only needed to run the linear system. In our case, we calculate Vfull, to obtain a benchmark.

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
svht_test = 30; % upper bound of r till which r-f analysis will run

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
    for k=1:svht_test
        dV(i-2,k) = (1/(12*dt))*(-V(i+2,k)+8*V(i+1,k)-8*V(i-1,k)+V(i-2,k));
    end
end

% concatenate
x = V(3:end-3,1:svht_test); % changed from 1:r to 1:24 to 1:svht_test
dx = dV;
xtest=Vtest(3:end-3,1:svht_test); % changed from 1:r to 1:24 to svht_test

xtestfinal = [x ; xtest(length(x)+1:end,:)];

% Optimum svht threshold calculations

sigs = diag(S); % storing all the sigmas
beta = size(H,1)/size(H,2); % # of rows/ # of columns
thresh = optimal_SVHT_coef(beta,0)*median(sigs); % Output is r=24
r = length(sigs(sigs>thresh)); % Upper bound of r that can be considered
sprintf('Optimum SVHT cutoff is r = %d',r)

%% CODE 3: Actual lobe switching numbers

% got 593 lobe switching instead of 605 shown in the original paper for t = 200:1200 ( 200000:1200000)

clear xpos, clear xswitch, clear minusplusswitch, clear plusminusswitch, clear lobeswitchact

for l=1:testset
    xpos(l,1)=sdata+l;
    if xdat(sdata+l,1)>0
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

sprintf('Actual number of lobe switching = %d',lobeswitchact) % unique for a time period selected, r, f does not influence it. Obtained from only studying x plot.

figure
plot(xdat(sdata+1:testdata,1)) % plotting x from lorenz system
hold on
plot(zeros(length(sdata+1:testdata),1))

%% CODE 4: varying forcing term, for each r.

tic
r=15; % change to 4 to test for range of r values
c=1;
error=zeros((svht_test-r)^2,5); % size of error matrix depends on the number of r,f tested.

while r<16 % change to svht_test +1 to test for a range of r values

    % Phase 2
    polyorder = 1;

    for f=15:15 % change to 4:svht_test to test for range of f values

        Theta = poolData(x,max(f,r),1,0);
        normTheta=zeros(1,size(Theta,2));

        for k=1:size(Theta,2)
            normTheta(k) = norm(Theta(:,k));
            Theta(:,k) = Theta(:,k)/normTheta(k);
        end
       
        if f<r
            clear Xi
            Xi = zeros(r+1,r-1); % pre allocating Xi
            for k=1:r-1
                if k>=f
                    j=k+1;
                else
                    j=k;
                end
                Xi(:,k) = sparsifyDynamics(Theta,dx(:,j),lambda*j,1);
            end

            for k=1:length(Xi)
                Xi(k,:) = Xi(k,:)/normTheta(k);
            end

            A = Xi(2:r+1,1:r-1)';
            B = A(:,f);
            A = [A(:,1:f-1) A(:,f+1:r)];

            sys = ss(A,B,eye(r-1),0*B);
            [y,t] = lsim(sys,xtestfinal(L,f),dt*(L-1),[x(1,1:f-1) x(1,f+1:r)]);

            % error on test data
            error(c,1)=r;
            error(c,2)=f;
            error(c,3)=norm(xtestfinal(Ltest,1)-y(Ltest,1)); % L2 norm of V1 error
            diff=[xtestfinal(Ltest,1)-y(Ltest,1),xtestfinal(Ltest,2)-y(Ltest,2),xtestfinal(Ltest,3)-y(Ltest,3)];
            error(c,4)=norm(diff,2); % L2 norm of V1-3 error
            error(c,5)=norm(diff,"fro"); % Frobenius norm of V1-3 error
            c=c+1;
            sprintf('r = %d',r)
            sprintf('f = %d',f)

        else
            clear Xi
            Xi = zeros(r+1,r-1); % pre allocating Xi
            for k=1:r-1
                Xi(:,k) = sparsifyDynamics([Theta(:,1:r) Theta(:,f+1)],dx(:,k),lambda*k,1);   % lambda = 0 gives better results
            end

            for k=1:length(Xi)
                Xi(k,:) = Xi(k,:)/normTheta(k);
            end

            A = Xi(2:r+1,1:r-1)';
            B = A(:,r);
            A = A(:,1:r-1);

            sys = ss(A,B,eye(r-1),0*B);
            [y,t] = lsim(sys,xtestfinal(L,f),dt*(L-1),x(1,1:r-1));


            % error on test data
            error(c,1)=r;
            error(c,2)=f;
            error(c,3)=norm(xtestfinal(Ltest,1)-y(Ltest,1)); % L2 norm of V1 error
            diff=[xtestfinal(Ltest,1)-y(Ltest,1),xtestfinal(Ltest,2)-y(Ltest,2),xtestfinal(Ltest,3)-y(Ltest,3)];
            error(c,4)=norm(diff,2); % L2 norm of V1-3 error
            error(c,5)=norm(diff,"fro"); % Frobenius norm of V1-3 error

            c=c+1;
            sprintf('r = %d',r)
            sprintf('f = %d',f)
        end
    end
    r=r+1;
end
toc


%% CODE 5: Reconstructed attractor 

figure
L = 300:50000;
plot3(y(L,1),y(L,2),y(L,3),'Color',[0 0 .5],'LineWidth',1.5)
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(-13.508433026692551,64.193062918106762)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')


%% CODE 6: Lobeswithcing predicted by forcing term

f=15;
fthresh = 0.00182;
sweep_window = 500;

clear vlobeswitch
e = 1;
d = sdata+1-stackmax-5;
while d < testdata-stackmax-5 - 1
    if xtestfinal(d,f) < fthresh && xtestfinal(d+2,f) > fthresh
        vlobeswitch(e) = d;
        d=d+sweep_window;
        e=e+1;
        sprintf('value of d is %d',d)
    elseif xtestfinal(d,f) > -fthresh && xtestfinal(d+2,f) < -fthresh
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
plot(xtestfinal(sdata:testdata-stackmax-5,f),'k')
hold on
a = [(1:testset)' ones(testset,1)*fthresh];
b = [(1:testset)' ones(testset,1)*(-fthresh)];
plot(a(:,1),a(:,2),'k')
hold on
plot(b(:,1),b(:,2),'k')

g=1;
while g < length(vlobeswitch) + 1
    if vlobeswitch(g)+sweep_window < length(xtestfinal)
        plot(vlobeswitch(g)-sdata:vlobeswitch(g)+sweep_window-sdata,xtestfinal(vlobeswitch(g):vlobeswitch(g)+sweep_window,f),'r')
    else
        plot(vlobeswitch(g)-sdata:length(xtestfinal)-sdata,xtestfinal(vlobeswitch(g):end,f),'r')
    end
    g=g+1;
end
dim = [0.2 0.5 0.3 0.3];
str = {strcat('f =',num2str(f),' ; Sweep window =',num2str(sweep_window),' ; Threshold = ',num2str(fthresh),' ; Lobe switching predicted =',num2str(e-1))};
title(str)


%% CODE 7: Plotting the forcing term alone.
f=19;
plot(xtestfinal(sdata:testdata-stackmax-5,f),'k')

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