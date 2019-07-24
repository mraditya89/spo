%% Program Description
% SPO - Clustering Solve Diophantine Equation
% @author : Yudi Aditya
% Email   : yudi.aditya.89@gmail.com

%% MAIN PROGRAM
clc
clear

%% Fungsi Objektif
% 1. Persamaan linear Diophantine
% 2. Luca dan Soydan
% 3. Cangul dkk
% 4. Perez dkk
% 5. Perez dkk
% 6. Amaya
% 7. Pell, Ai dkk
% 8. Markoff
% 9. Page 6 No 8
% 10. Page 6 No 9
% 11. Page 7 No 2
% 12. Page 7 No 8

kasus       = 10 ;
%% Data Algoritma
m_cluster   = 2000; %Number of search point
gamma       = 0.1;  %Cut off Parameter for function Fx
k_cluster   = 20;   %Maximum number of iteration at the clustering phase
theta       = 180/4;   %Spiral rotation
r           = 0.95;  %Spiral ratio
m           = 50;  %Number of search point
k_max       = 50;  %Number of iteration
epsilon     = 10^-5;%Parameter of root acceptance
delta       = 0.01;  %Parameter to distinguish between root

f           = operasi;
tic
%% Tools
[n, min_x, max_x] = f.data(kasus);
S = eye (n);
for i=1:n-1
    for j=1:i
        R = f.matriksR(n-i,n+1-j,theta, n);
        S = S*R;
    end
end
S =S*r;

root =[];
fitness_fx = [];

%% Subroutine 1 : Clustering
[c_cluster, domain_max, domain_min] = f.clustering(m_cluster, k_cluster, S, gamma,n, max_x, min_x, kasus);
[row,col] = size (c_cluster);

% Subroutine 2 : Spiral Process
for i=1:row
    [optimum_x, optimum_fx]   = f.spo(S, n, k_max, max_x, min_x, domain_max(i,:), domain_min(i,:), m, epsilon, root, delta, kasus);
    
    % Subroutine 3 : Selection
    [root, fitness_fx] = f.selection (optimum_x, optimum_fx, fitness_fx, root, epsilon, delta, kasus);
end
toc


[row, col] = size(root);

root;
sorting_root=[];
for i=1:row
    temp_root = sort(root(i,:));
    if i==1
        sorting_root = [sorting_root; temp_root];
        m_sorting = 1;
    else
        for j=1:m_sorting
            if sorting_root(j,:) == temp_root;
                break;
            else
                sorting_root = [sorting_root; temp_root];
                m_sorting =m_sorting+1;
                break;
            end
        end
    end    
end
sorting_root;
