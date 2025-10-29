% New Generation Reservoir Computing using pseudorandom nonlinear projection
% of time-delay embedded input according to paper
% "Next-Generation Reservoir Computing for Dynamical Inference"
%  https://doi.org/10.48550/arXiv.2509.11338
%
% Prediction of Lorenz system
% Map: X_{n+1} = W_{out} * P(X_n,X_{n-1},...)  
%
close all
clear all

% discretization step
dt= 0.04;

%Model parameters
sig=10.; 
r=28;
b=8/3;

%Lyapunov time for given parameters
Tlyap = 1.1;

P.sig=sig;
P.r=r;
P.b=b;

% feature vector length
M = 1000;
% number of samples for feature vector construction
H = 25; 

opts = odeset('RelTol',1e-7,'AbsTol',1e-9);

% Integration time for transitional process
Ttr = 1000;

% Integration time for training data
Ttr1 = 1000;
Ttr1 = round(Ttr1/dt)*dt;

%Integration time for validation
Tvalid = 10%(validation_points-1)*dt;
Tvalid = round(Tvalid/dt)*dt;

%initial conditions
yinit = (randn(3,1)+1);  


    

[T,Y] = ode45(@(t,X)eqsns(t,X,P),[0 Ttr],yinit,opts);
yinit = Y(end,:);
% training signal generation
[T,Y] = ode45(@(t,X)eqsns(t,X,P),0:dt:Ttr1,yinit,opts);
yinit = Y(end,:);
train_points = length(T);
% validation signal generation
[Tvalid,Yvalid] = ode45(@(t,X)eqsns(t,X,P),0:dt:Tvalid,yinit,opts);

% figure
% for i1 = 1:3
%    subplot(3,1,i1)
%    plot(T,Y(:,i1));
% end

X = Y(:,1);
xmx = max(X);
xmn = min(X);

eps = 0.1;
a = (2*eps-1)/(xmn-xmx);
b= eps-a*xmn;

% normalize training signal to the inteval [-eps; 1-eps];
Xnorm = a*X+b;


% array to keep pairs of indices (size: M x 2)
pairs = pairs_generation(H,M);


% % %  Producing feature matrix P % % %
P = feature_matrix(Xnorm, H, M, pairs,0.01);
U = Xnorm(H+1:end);

% figure
% histogram(P(:),1000)


% % %   Estimating weight matrix Wout: U = P*Wout   % % % 
PT = P.';
Wout = pinv(PT*P)*PT*U;
   
Upred = P*Wout;



figure
subplot(211)
plot(T(H+1:end),U), hold on
plot(T(H+1:end),Upred,'--')
title('Training signal- one step pred')
legend({'measur','prediction'})
xlabel('t')
subplot(212)
plot(T(H+1:end),Upred-U), hold on
ylabel('\Delta x')
xlabel('t')


Loss = mean( (Upred-U(1,:).').^2 );
fprintf('Loss = %e\n',Loss)



% % % % % %   V A L I D A T I O N   % % % % % % % %

% normalize validation signal
Xvalidnorm = a*Yvalid(:,1)+b;

Pvalid = feature_matrix(Xvalidnorm, H, M, pairs,0.01);
Uvalid = Xvalidnorm(H+1:end);

Uvalid_pred = Pvalid*Wout;


figure
subplot(211)
plot(Tvalid(H+1:end)-Tvalid(H),Uvalid), hold on
plot(Tvalid(H+1:end)-Tvalid(H),Uvalid_pred,'--')
legend({'True','Predic'})
xlabel('t')
title('Validation signal - one step pred.')
subplot(212)
plot(Tvalid(H+1:end)-Tvalid(H),Uvalid-Uvalid_pred), hold on
legend({'True - Predic'})
xlabel('t')


% % % % % % %    LONG TIME PREDICTION     % % % % % % %

% As starting initial conditions we use the begining of validation signal
validation_points = length(Xvalidnorm);
X = Xvalidnorm(1:H).';

% long time prediction algorithm
P_long_pred = NaN(1,M);
U_long_pred = NaN(validation_points-H,1);
for t1 = 1:validation_points-H-1
    Utmp = NaN(H+M,1);    
    %current state
    Utmp(1:H,1) = X;
    len = H +1;
    for i1 = 1:M
        pi1 = pairs(i1,1);
        pj1 = pairs(i1,2);
    
        P_long_pred(1,i1) = (1-Utmp(pi1))^Utmp(pj1);
    
        Utmp(len) = P_long_pred(1,i1);   
        len = len +1;
    end
 
    U_long_pred(t1) = P_long_pred*Wout;
    X = [X(2:end) U_long_pred(t1)];   
end

figure
subplot(211)
plot( (Tvalid-Tvalid(H))/Tlyap,Xvalidnorm), hold on
plot( (Tvalid(H+1:end)-Tvalid(H))/Tlyap,U_long_pred,'--')
legend({'True','Pred'})
xlim([-Tvalid(H)/Tlyap inf])
title('Long time prediction')
xlabel('$\Lambda t$','Interpreter','latex')
subplot(212)
plot( (Tvalid(H+1:end)-Tvalid(H))/Tlyap ,Xvalidnorm(H+1:end)-U_long_pred), hold on
legend({'True-Pred'})
xlabel('$\Lambda t$','Interpreter','latex')
xlim([0,3])
ylim([-0.025 0.025])



function P = feature_matrix(Xnorm, H,M,pairs,eps)

    % % %  Producing feature matrix P % % %
    l1 = 1;
    l2 = H;
    train_points = length(Xnorm);
    P = NaN(train_points-H,M);
    Xtrain = Xnorm;
    for t1 = 1:train_points-H
        %temporal vector to store H steps of signal together with projections
        Utmp = NaN(H+M,1);    
        %current H states
        Utmp(1:H,1) = Xtrain(l1:l2)+ eps*randn(H,1);      
        len = H +1;
        for i1 = 1:M
            pi1 = pairs(i1,1);
            pj1 = pairs(i1,2);
        
            P(t1,i1) = (1-Utmp(pi1))^Utmp(pj1);
        
            Utmp(len) = P(t1,i1);   
            len = len +1;
        end
        l1 = l1 + 1;
        l2 = l2 + 1;
    end

end


function pairs= pairs_generation(H,M)
    pairs = NaN(M,2);
    % maximal number from which to take two random numbers
    len = H;
    % initial random pair of integers
    pr1 = randi(len,1,2);
    for i1 = 1:M
        %check if random pair pr1 is not alredy in pairs array,
        % also that pr1(1) and pr1(2) have different values        
        while (any(ismember(pairs,pr1,"rows")) || (pr1(1)==pr1(2)))
            pr1 = randi(len,1,2);
        end
        pairs(i1,:)= pr1;
        len = len + 1;
        pr1 = randi(len,1,2);
    end

end


function dy = eqsns(~,X,P)
    sig=P.sig; 
    r=P.r;
    b=P.b;

    x=X(1); y=X(2); z=X(3);

    dy = zeros(3,1);
    dy(1)=-sig*(x-y);
    dy(2)= r*x-y-x*z;
    dy(3)= x*y - b *z;
    
end


