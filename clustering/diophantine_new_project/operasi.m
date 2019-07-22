function f = operasi
    f.data = @Data;
    f.domain = @Check_dom;
    f.matriksR = @MatriksR;
    f.objektif = @Objektif;
    f.spo = @Spo;  
    f.clustering = @Clustering;
    f.selection = @Selection;
end

function domain = Check_dom (x, max_x, min_x)
    % Fungsi yang menentukan 
    % x     = Koordinat titik yang dicek
    % x_max = Batas max x
    % x_min = Batas min x

    dom1 = false;
    dom2 = false;
    domain = false;
    if (x<=max_x) dom1 =true; end
    if (x>=min_x) dom2 =true; end
    if (dom1==true && dom2==true)
        domain = true;
    end
end

function R = MatriksR (i,j,theta, n)
    % Fungsi untuk membentuk matriks rotasi n dimensi
    % i,j   = indeks baris - kolom
    % theta = sudut rotasi
    % n     = dimensi 
    for r=1:n
        for c =1:n
            if (r == i && c ==i)
                R(r,c) = cos(theta*pi/180);
            elseif (r == j && c ==j)
                R(r,c) = cos(theta*pi/180);
            elseif (r == j && c ==i)    
                R(r,c) = sin(theta*pi/180);     
            elseif (r == i && c ==j)
                R(r,c) = -sin(theta*pi/180);           
            else 
                if (r==c)
                    R(r,c) = 1;
                else
                    R(r,c) = 0;
                end
            end
        end
    end
end

function [n, min_x, max_x] = Data (kasus)
    if (kasus == 1)
        % Pers Linear Diophantine
        n     = 2;
        min_x = 0*ones(1,n);
        max_x = 50*ones(1,n);
    elseif (kasus == 2)
        % Luca dan Soydan
        n     = 3;
        min_x = 0*ones(1,n);
        max_x = 50*ones(1,n);
    elseif (kasus ==3)
        % Cangul dkk
        n     = 4;
        min_x = 0*ones(1,n);
        max_x = 20*ones(1,n);
    elseif (kasus ==4)
        %Perez dkk
        n     = 6;
        min_x = 0*ones(1,n);
        max_x = 20*ones(1,n);
    elseif (kasus ==5)
        %Perez dkk
        n     = 10;
        min_x = 0*ones(1,n);
        max_x = 10*ones(1,n);
    elseif (kasus ==6)
        %Amaya
        n     = 7;
        min_x = -10*ones(1,n);
        max_x = 10*ones(1,n);
    elseif (kasus==7)
        %Pell
        n     = 3;
        min_x = ones(1,n);
        max_x = 75*ones(1,n);
    elseif (kasus==8)
        %Markoff
        n     = 3;
        min_x = ones(1,n);
        max_x = 10*ones(1,n);
    elseif (kasus==9)
        %halaman 6 no 8
        n     = 9;
        min_x = ones(1,n);
        max_x = 26*ones(1,n);
   elseif (kasus==10)
        %halaman 6 no 9
        n     = 10;
        min_x = ones(1,n);
        max_x = 26*ones(1,n);
   elseif (kasus==11)
        %halaman 7 no 2
        n     = 2;
        min_x = ones(1,n);
        max_x = 10*ones(1,n);
   elseif (kasus==12)
        %halaman 7 no 8
        n     = 2;
        min_x = ones(1,n);
        max_x = 10*ones(1,n);
    end
end

function fitness = Objektif (x, kasus)
    % Fungsi untuk memgevalusi nilai fungsi objektif (fitness)
    % x     = Koordinat titik
    if (kasus ==1)
        %Persamaan Linear Diophantine
        f = 8*x(1)-6*x(2)-14;
        fitness = 1/(1+abs(f));
    elseif (kasus ==2)
        %Luca dan Soydan
        n = 3;
        f = 2^x(3)+n*x(1)^2-x(2)^n;
        fitness = 1/(1+abs(f));
    elseif (kasus ==3)
        %Cangul dkk
        n = 4;
        f = x(1)^2+2^x(3)*11^x(4)-x(2)^n;
        fitness = 1/(1+abs(f));
	elseif (kasus ==4)
        %Perez dkk
        f1 = 5*x(1)+10*x(2)-5*x(3)+x(5)^3+8*x(6)-1772;
        f2 = 3*x(1)+18*x(3)-5*x(5)+17*x(6)-153;
        f3 = 6*x(1)+x(3)-99*x(2)+(15*x(6))^2-1772;
        f4 = -x(1)+5*x(2)+8*x(3)-6*x(4)+15*x(5)+10*x(6)-277;
        f5 = (x(1)+x(2))^2-7*x(3)+5*x(4)+12*x(5)-8*x(6)-150;
        f6 = x(2)+5*x(3)-3*x(5)-x(6)-4;
        fitness = 1/(1+abs(f1)+abs(f2)+abs(f3)+abs(f4)+abs(f5)+abs(f6));
    elseif (kasus ==5)
        %Perez dkk
        f1  = x(1)^2-2*(x(2)+x(4))^3+x(5)-3*x(6)-x(7)+4*x(9)+15*x(10)+24; 
        f2  = 2*x(1)+(x(2)+3*x(4))^3+(5*x(7))^2-6*x(8)+x(9)-9*x(10)-31;
        f3  = 3*x(1)-(2*x(2))^2+10*x(3)-9*x(4)+3*x(5)+x(6)-2*x(7)-8*x(8)+12*x(9)-5*x(10)+25;
        f4  = 5*x(1)+2*x(2)-8*x(4)-3*x(5)+4*x(6)+x(7)-x(9)-23;
        f5  = x(1)-x(3)+2*x(5)-x(7)-x(9)+3;
        f6  = x(2)+(2*x(4))^2-6*x(6)-x(8)+2*x(10)-8;
        f7  = 3*x(1)+2*x(2)-5*x(3)-(x(4))^4-2*x(5)+x(6)+4*x(7)-10*x(8)+8*x(9)+9;
        f8  = x(1)-3*x(2)+4*x(4)+x(6)-6*x(7)+x(8)-2*x(9)+16;
        f9  = (2*x(1)+x(2))^2+3*x(3)-10*x(5)-(x(6)+3*x(7))^3-x(8)-6*x(9)-27;     
        fitness = 1/(1+abs(f1)+abs(f2)+abs(f3)+abs(f4)+abs(f5)+abs(f6)+abs(f7)+abs(f8)+abs(f9));
    elseif (kasus==6)
        %amaya
        f1 = 3*x(1)+x(2)-6;
        f2 = 4*x(1)+3*x(2)+x(3)+x(5)-15;
        f3 = 3*x(1)+4*x(2)+3*x(3)+x(4)+x(5)+x(6)-20;
        f4 = 3*x(2)+4*x(3)+3*x(4)+x(5)+x(6)+x(7)-15;
        f5 = 3*x(3)+4*x(4)+x(6)+x(7)-6;
        f6 = 3*x(4)+x(7)-1;
        fitness = 1/(1+abs(f1)+abs(f2)+abs(f3)+abs(f4)+abs(f5)+abs(f6));
    elseif (kasus==7)
        %Pell, Ai
        p  = 11;
        f1 = x(1)^2-24*x(2)^2-1;
        f2 = x(2)^2-p*x(3)^2-1;
        fitness = 1/(1+abs(f1)+abs(f2));
     elseif (kasus==8)
        %Markoff
        f1 = x(1)^2+x(2)^2+x(3)^2-3*x(1)*x(2)*x(3);
        fitness = 1/(1+abs(f1));
    elseif (kasus==9)
        %halaman 6 no 8
        f1 = 0;
        for i=1:9
            f1 = f1 + x(i)^2;
        end
        f1 = f1 - 720;
        fitness = 1/(1+abs(f1));
    elseif (kasus==10)
        %halaman 6 no 9
        f1 = 0;
        for i=1:10
            f1 = f1 + x(i)^2;
        end
        f1 = f1 - 956;
        fitness = 1/(1+abs(f1));
    elseif (kasus==11)
        %halaman 7 no 2
        f1 = x(1)^3+x(2)^3-1008;
        fitness = 1/(1+abs(f1));
    elseif (kasus==12)
        %halaman 7 no 8
        f1 = x(1)^9+x(2)^9-1000019683;
        fitness = 1/(1+abs(f1));
    end   
end

function [optimum_x, optimum_fx] = Spo (S, n, k_max, max_x,min_x,domain_max, domain_min, m_cluster,epsilon, root, delta, kasus)
    % Fungsi yang menjalakan metoda algoritma optimasi spiral
    % S         = Matriks Spiral
    % n         = Dimensi 
    % k_max     = Maksimum Iterasi
    % m_cluster = Banyak Titik
    
    [row,col] = size (root);
    for k=1:k_max
        % Step 1 : Generate Sobol Sequence for n dimension     
        if (k==1)
           P = sobolset(n);
           x = net(P,m_cluster);
           for i=1:n
                x(:,i) = domain_min(i)+x(:,i)*(domain_max(i)-domain_min(i));
           end
        end
        y = round(x);
        % Step 2 : Find the Spiral Center            
        for i=1:m_cluster
            % Checking domain 
            domain = Check_dom (y(i,:),max_x, min_x);
            % Find fitness function
            if (domain == true)            
                fit_x(i) = Objektif (y(i,:), kasus);
            else 
                fit_x(i) = -realmax;
            end 
        end
             
        idx = find(fit_x==max(fit_x));
        idx = idx(1);
        x_star = x(idx,:);
        
        % Spiral Process
        for i=1:m_cluster 
            if (i~=idx)
                X = x(i,:);
                X_new = S*X' -(S-eye(n))*x_star';
                x(i,:) = X_new;
            end
        end
              
    end
    optimum_x   = y(idx,:);
    optimum_fx  = fit_x(idx);
end

function [c_cluster,domain_max,domain_min] = Clustering(m_cluster, k_cluster, S, gamma, n, max_x, min_x, kasus)
    % m_cluster : Number of search point
    % k_cluster : Number of Iteration
    % S         : Spiral Matrix
    % gamma     : Toleransi Fx
    
    c_cluster = [];
    domain_max = [];
    domain_min = [];
    %% Step 1 : Generate Sobol sequence for n dimension
    P = sobolset(n);
    x = net(P,m_cluster);
    for i=1:n
        x(:,i) = min_x(i)+x(:,i)*(max_x(i)-min_x(i));
    end
    
    %% Step 2 : Find the Spiral Center               
    % Determine Fitness Function
    y = round(x);
    for i=1:m_cluster
        fit_x(i) = Objektif (y(i,:), kasus);
    end
    
    % Find arg max
    idx    = find(fit_x == max(fit_x));
    idx    = idx(1);
    x_star = x(idx,:);
    r      = 1/2*min(abs(max_x-min_x));
    
    %Store center and radii cluster
    c_cluster   = x(idx,:);
    r_cluster   = [r];
    
    %% Step 3 : Clustering Phase 1
    for k=1:k_cluster  
        [row,col] = size (x);
        loop = true;
        i = 1;
        while loop == true
            % Check x whether a member of exsisting cluster            
            member = false;
            for j=1:length(r_cluster)
                if (x(i,:) == c_cluster(j,:))
                    member = true;
                    break;
                end
            end
                                      
            if (Objektif(y(i,:),kasus)>gamma && member == false)
                % Clustering function
                temp1 = realmax;
                for j=1:length(r_cluster)
                    temp2 = norm(x(i,:)-c_cluster(j,:));
                    if (temp1>temp2)
                        temp1 = temp2;
                        idx_cluster = j;
                    end
                end
                
                %Data y, xc, xt
                xc = c_cluster(idx_cluster,:);
                xt = (x(i,:)+xc)/2;
                    
                Fy  = Objektif (y(i,:),kasus);
                Fxc = Objektif (round(xc),kasus);
                Fxt = Objektif (round(xt),kasus);
                
                domain = Check_dom (y(i,:),max_x, min_x);
                if (domain==true)
                    if (Fxt<Fy & Fxt<Fxc)
                        %Set new cluster center di y dan radius y-xt
                        c_cluster = [c_cluster; x(i,:)];
                        r = norm(x(i,:)-xt);
                        r_cluster = [r_cluster; r];

                    elseif (Fxt>Fy & Fxt>Fxc)
                        %Set new cluster center di y dan radius y-xt
                        c_cluster = [c_cluster; x(i,:)];
                        r = norm(x(i,:)-xt);
                        r_cluster = [r_cluster; r];
                        x = [x; xt];
                        y = [y; round(xt)];
                        [row,col] =size (x);
                    
                    elseif (Fy>Fxc)
                        %Set cluster center di y
                        c_cluster (idx_cluster,:) = x(i,:);
                    end
                end
                r = norm (x(i,:)-xt);
                r_cluster(idx_cluster) = r;
            end
            
            if (i== row)
                loop = false;
            end
            i=i+1; 
        end
                       
        %% Step 4 : Spiral Process
        % Determine Fitness Function
        if (k_cluster >1)
            for i=1:m_cluster
                domain = Check_dom (y(i,:),max_x, min_x);
                if (domain== true)            
                    fit_x(i) = Objektif (y(i,:), kasus);
                else 
                    fit_x(i) = 0;
                end
            end
            % Find Arg Max
            idx    = find(fit_x==max(fit_x));
            idx    = idx(1);
            x_star = x(idx,:);
        end
        
        % Update x 
        for i=1:m_cluster 
            if (i~=idx)
                X = x(i,:);
                X_new = S*X' -(S-eye(n))*x_star';
                x(i,:) = X_new;
            end
        end
        y =round(x);
    end
        
    for i=1:length(r_cluster)
    	domain_max (i,:) = c_cluster(i,:)+r_cluster(i);
        domain_min (i,:) = c_cluster(i,:)-r_cluster(i);
    end
end

function [root, fitness_fx] = Selection (optimum_x, optimum_fx, fitness_fx, root, epsilon, delta, kasus)
    
    % Distinguish two solution    
    [r,c] = size(root);
    new_root = true;
    for j=1:r
        if (optimum_x == root(j,:))
            new_root = false;
            break;
        end
    end

    % Decision root or not
    fitness = 1-optimum_fx;
    if (new_root == true & fitness<=epsilon)
        root = [root ; optimum_x];
        fitness_fx = [fitness_fx; fitness];
    end 
end
