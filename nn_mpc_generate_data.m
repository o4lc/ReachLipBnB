clc;

clear all;
close all;
set(0,'DefaultFigureWindowStyle','docked')
dt = .1;
% A = [1 1; 0 1];
% B = [.5; 1];
% c = [0; 0];

g = 9.8;
A = [zeros(3, 3), eye(3);
    zeros(3, 6)];
B = [zeros(3, 3)
    g 0 0
    0 -g 0
    0 0 1];
c = [zeros(5, 1);-g];

A = eye(6) + dt * A;
B = dt * B;
c = dt * c;

nx = size(A, 1); nu = size(B, 2);

horizon = 45;

model = LTISystem('A', A, 'B', B, "f", c, "Ts", dt);

% model.x.min = [-5; -5;];
% model.x.max = [5; 5;];
model.x.min = [-5; -5; -5; -1; -1; -1];
model.x.max = [5; 5; 5; 1; 1; 1];

%model.x.with('terminalSet');
%model.x.terminalSet = Polyhedron( 'Ae', eye(nx), 'be', zeros(nx,1));

X = [eye(nx) model.x.max;-eye(nx) -model.x.min];

% model.u.min = [-1];
% model.u.max = [1];
model.u.min = [-pi/9; -pi/9; 0];
model.u.max = [pi/9; pi/9; 2 * g];
U = [eye(nu) model.u.max;-eye(nu) -model.u.min];


Q = eye(nx);
model.x.penalty = QuadFunction(Q);

R = eye(nu);
model.u.penalty = QuadFunction(R);

mpc = MPCController(model, horizon);
fprintf("yello")
tic
expmpc = mpc.toExplicit();
toc
fprintf("hello")
fileName = sprintf("mpcQuadHorizon%d-%s.mat", horizon, string(datetime));
save(fileName)
%%

% Generate data for neural xwnetwork training
generate_trainData = 1;

if(generate_trainData)
    
    sampleResol = 20;
    
    % Vertices = [-12 -7; -12 7; 12 -7; 12 7];
    % dom = Polyhedron(Vertices);
    
    dom = expmpc.feedback.Domain;
    samples = dom.grid(sampleResol);
    
    %samples = rect2d(model.x.min,model.x.max);
    %samples = samples';
    
    dataSize = size(samples, 1);
    tic
    labels = zeros(nu, dataSize);
    for ii = 1:dataSize
        createdLabels = expmpc.evaluate(samples(ii, :)');
        labels(:, ii) = createdLabels;
        if isnan(isnan(createdLabels))
            fprintf("NaN in labels :| discarding")
            break
        end
        
    end
    toc
    Xtrain = samples';
    Ytrain = labels;
    if any(any(isnan(labels)))
        fprintf("NaN in labels :| discarding")
    end
    

    fileName = 'quadRotorTrainData.mat';
    save(fileName,'Xtrain','Ytrain', 'A', 'B');
%     save('double_integrator/system.mat','model','mpc')
end
