function [A,B] = sgd(x,y)
%UNTITLED2 
%   Input:
%   x:   num-by-dim matrix .mun is the number of data points,
%   dim is the the dimension of a point
%   y:   num-by-1 vector, specifying the class that each point belongs
% to +1 or -1
%   output:
%   w: dim-by -1 vector ,the mormal dimension of hyperpalne
%   b: a scalar, the bias
   [n,d]=size(x);
   if(n ~= size(y,1))
       disp('size is not correspondent');
   end
   
   epoch=250;
   
   A=randn(64,1);
   B=randn(64,1);
   
   for i=1:epoch
       eta=1/2000;
       [n1 m1] = size(x);
       [n2 m2] = size(y);
       
       %data shuffle
       tem=[x,y];
       tem=tem(randperm(size(tem,1)),:);   
       x=tem(:,1:d);
       y=tem(:,d+1);
       
       loss=;
       
       for k=1:n
           yk=y(k,1);
           xk=x(k,:);
           c=yk*(xk*w+b);
           if(c<1)
               w=w+eta*(-w+yk*xk');
               b=b+eta*yk;
               loss=[loss;1-c];
           else
               w=w-eta*w;
               loss=[loss;0];
           end 
       end
  
       

   end
end