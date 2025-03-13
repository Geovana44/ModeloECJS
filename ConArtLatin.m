%% 1. Leer datos y configurar parámetros
% file_path = 'Demandaóptima_Comunidad8p.xlsx';
file_path = 'Demandaóptima_Comunidad6pp';
df = readtable(file_path);

% Extraer columnas
G1 = df.Glim1;
G2 = df.Glim2;
G3 = df.Glim3;
G4 = df.Glim4;
G5 = df.Glim5;
G6 = df.Glim6;
D1 = df.D11;
D2 = df.D22;
D3 = df.D33;
D4 = df.D44;
D5 = df.D55;
D6 = df.D66;

% Crear matrices
generation = [G1, G2, G3, G4, G5, G6];
demand = [D1, D2, D3, D4, D5, D6];

% Parámetros fijos
theta = 10 * ones(1,6);
lamda = 100 * ones(1,6);
etha0 = 1 * ones(1,6);
%etha0 = [0.1, 1, 10, 0.1, 0.1, 0.1];
Pgs = 1650;
Pgb = 50;
prosumers = 6;
N = 1:prosumers;
PmasCP = N;

% Coeficientes a y b
a= [0.089, 0.110, 0.069, 0, 0, 0] 
b= [52, 58, 40, 37, 32, 0, 0]
prosumers = 6;

WelfareAcu2 = zeros(24, 5); % 24 hours, 5 metrics per hour
SolP2 = cell(24, 1); % Celda para almacenar pii_opt en cada iteración
Solpot2 = cell(24, 1); % Celda para almacenar Pij_opt en cada iteración
for t=1:24
% Coeficientes a y b

a= [0.089, 0.110, 0.069, 0, 0, 0] 

b= [52, 58, 40, 37, 32, 0, 0]
% Seleccionar tiempo t=1
%t = 14;
Glim = generation(t, :);
Dopt = demand(t, :);

% Filtrar consumidores y generadores
is_consumer = (Glim ./ Dopt) <= 1;
is_generator = (Glim ./ Dopt) >= 1;

Di = Dopt(is_consumer) - Glim(is_consumer);
Gi = Glim(is_consumer);
thetai = theta(is_consumer);
lamdai = lamda(is_consumer);
etha = etha0(is_consumer);

Dj = Dopt(is_generator);
Gj = Glim(is_generator) - Dopt(is_generator);
thetaj = theta(is_generator);
lamdaj = lamda(is_generator);

% Actualizar coeficientes para generadores
a = a(is_generator);
b = b(is_generator);

% Configurar estructura de parámetros
args = struct();
args.consumidores = length(Di);
args.generadores = length(Gj);
args.consumer = 1:args.consumidores;
args.generator = 1:args.generadores;
args.etha = etha;
args.lamdai = lamdai;
args.Gi = Gi;
args.thetai = thetai;
args.Di = Di;
args.Dj = Dj;
args.a = a;
args.b = b;
args.thetaj = thetaj;
args.lamdaj = lamdaj;
args.Gj = Gj;

%% 2. Configurar problema de optimización
% Valores iniciales
pii0 = Pgb * ones(args.consumidores, 1); % Vector columna
sli0 = rand(args.generadores * args.consumidores, 1);

lb_pii = Pgb * ones(args.consumidores, 1);
ub_pii = Pgs * ones(args.consumidores, 1);
lb_Pij = zeros(numel(sli0), 1);
ub_Pij = inf(numel(sli0), 1);

lb = [lb_pii; lb_Pij];
ub = [ub_pii; ub_Pij];

%% 3. Ejecutar optimización
[pii_opt, Pij_opt, total_welfare, tiempo] = CombinedOpt(...
    pii0, sli0, lb, ub, [args.generadores, args.consumidores], args);

%% 4. Mostrar resultados
disp('=== Resultados ===');
fprintf('Tiempo de ejecución: %.2f segundos\n', tiempo);
fprintf('Bienestar total: %.2f\n', total_welfare);

disp('Precios óptimos:');
disp(pii_opt');

disp('Asignaciones óptimas Pij:');
disp(Pij_opt);

% Verificación de restricciones
[Costos, re,pagos] = Costosk(Pij_opt, args.a, args.b, pii_opt,args.generator);
[df, di, gf, gj] = PruebaRestriG(args.Di, Pij_opt, args.Gj);

disp('=== Verificación ===');
fprintf('Demanda total optimizada: %.2f \n', df);
fprintf('Demanda total Requerida: %.2f\n', di);
fprintf('Generación utilizada: %.2f \n', gf);
fprintf('Generación Disponible: %.2f\n', gj);
disp(Costos-re)
fprintf('Recompen: %.2f\n', re);
fprintf('Pagos: %.2f\n', pagos);
fprintf('RecompenT: %.2f\n', sum(re));
fprintf('PagosT: %.2f\n', sum(pagos));
fprintf('delta: %.2f\n', sum(re)-sum(pagos));
WelfareiSol = Welfarei(pii_opt', Pij_opt, args.consumidores, args.generadores, args.etha, args.lamdai, args.Di, args.thetai);
Wj_totalSol = Welfarejgen(Pij_opt, args.Gj, args.a, args.b, args.thetaj, args.lamdaj, pii_opt, args.generadores, args.consumidores, args.generator);

SolP2{t} = pii_opt

Solpot2{t}=  Pij_opt(:)'
WelfareAcu2(t, :) = [total_welfare,-WelfareiSol,-Wj_totalSol, sum(re), sum(pagos)];
end
%% Funciones de optimización
function total_welfare = CombinedObjective(x, consumidores, generator, consumer, etha, lamdai, Di, thetai, Gj, a, b, thetaj, lamdaj, generadores)
    pii = x(1:length(consumer))'; % Vector fila (1×consumidores)
    Pij = reshape(x(length(consumer)+1:end), [generadores, consumidores]);
    matriz = ones(consumidores) - eye(consumidores);
    
    compe = arrayfun(@(i) sum(matriz(i, :) .* pii .* sum(Pij(:, i), 1)), 1:consumidores);
    Wi = arrayfun(@(i) lamdai(i) * Di(i) - thetai(i) * Di(i)^2 + ...
        (sum(Pij(:, i)) / log(abs(pii(i)) + 1)) - compe(i) * etha(i), 1:consumidores);
    welfare_i = -sum(Wi);
    disp(-welfare_i)
    
   
    Wj = arrayfun(@(j) lamdaj(j) * Gj(j) - thetaj(j) * Gj(j)^2 - ...
         sum(Pij(j, :) /log(1+pii)) - ...
         a(j) * (sum(Pij(j, :))^2) - ...
         b(j) * sum(Pij(j, :)), generator);

    welfare_j = -sum(Wj);
    disp('gene')
    disp(-welfare_j)
    
    total_welfare = welfare_i + welfare_j;
end

function [c, ceq] = CombinedConstraints(x, args)
    c = [];
    ceq = [];
    pii = x(1:args.consumidores)';
    Pij = reshape(x(args.consumidores+1:end), [args.generadores, args.consumidores]);
    
    % Restricciones de generadores
    for j = 1:args.generadores
        costos_j = args.a(j)*sum(Pij(j,:))^2 + args.b(j)*sum(Pij(j,:));
        ingresos_j = sum(pii .* Pij(j,:));
        c = [c; costos_j - ingresos_j]; % c <= 0
    end
    
    % Restricciones de capacidad
    c = [c; sum(Pij,2) - args.Gj']; % sum(Pij) <= Gj
    
    % Restricciones de demanda
    c = [c; sum(Pij,1)'-args.Di']; % sum(Pij)<= Di
    
    % Restricciones de igualdad
    if sum(args.Di) <= sum(args.Gj)
        ceq = sum(Pij,1)' - args.Di';
    else
        ceq = sum(Pij,2) - args.Gj';
    end
end

function [pii_opt, Pij_opt, total_welfare, tiempo] = CombinedOpt(pii0, sli0, lb, ub, Pij_shape, args)
    tic;
    x0 = [pii0; sli0];
    
    options = optimoptions('fmincon',...
        'Algorithm','interior-point',... %interior-point
        'Display','iter',...
        'StepTolerance',1e-6,...
        'OptimalityTolerance',1e-6,...
        'ConstraintTolerance',1e-10);
    
    [x_opt, fval] = fmincon(@(x)CombinedObjective(x, args.consumidores, args.generator,...
        args.consumer, args.etha, args.lamdai, args.Gi, args.thetai, args.Dj,...
        args.a, args.b, args.thetaj, args.lamdaj, args.generadores),...
        x0, [], [], [], [], lb, ub, @(x)CombinedConstraints(x, args), options);
    
    pii_opt = x_opt(1:length(pii0))';
    Pij_opt = reshape(x_opt(length(pii0)+1:end), Pij_shape);
    total_welfare = -fval;
    tiempo = toc;
end

%% Funciones auxiliares
function [Costos, re, pagos] = Costosk(Pij, a, b, pii, generator)
    Costos = arrayfun(@(j) a(j)*sum(Pij(j,:))^2 + b(j)*sum(Pij(j,:)), generator);
    re = sum(Pij .* pii, 2)';
    pagos = pii.*sum(Pij,1);
end

function [demandafinal, Di, generacionfinal, Gj] = PruebaRestriG(Di, Pij, Gj)
    demandafinal = sum(Pij,1);
    generacionfinal = sum(Pij,2)';
end

function Welfarei = Welfarei(x, Pij_flat, consumidores, generadores, etha, lamdai, Di, thetai)
    pii = x';
    Pij = reshape(Pij_flat, [generadores, consumidores]);
    matriz = ones(consumidores) - eye(consumidores);
    compe = arrayfun(@(i) sum(matriz(i, :) .* pii .* sum(Pij(:, i), 1)), 1:consumidores);

    Wi = arrayfun(@(i) lamdai(i) * Di(i) - thetai(i) * Di(i)^2 + ...
        (sum(Pij(:, i))/log(abs(pii(i)) + 1)) - compe(i) * etha(i), 1:consumidores);

    Welfarei = -sum(Wi);
end


function Wj_total = Welfarejgen(y, Gj, a, b, thetaj, lamdaj, pii, generadores, consumidores, generator)
    Pij = reshape(y, [generadores, consumidores]);
    %disp(Pij)
    Wj = arrayfun(@(j) lamdaj(j) * Gj(j) - thetaj(j) * Gj(j)^2 - ...
         sum(Pij(j, :) /log(1+pii)) - ...
         a(j) * (sum(Pij(j, :))^2) - ...
         b(j) * sum(Pij(j, :)), generator);
    %disp(Wj)
    Wj_total = -sum(Wj);
end