x = -10:0.01:10;
lamda = 0;
y0 = 0.5.*(-1-x).^2 + 0.5.*(1-x).^2 + lamda*abs(x-1);

lamda = 1;
y1 = 0.5.*(-1-x).^2 + 0.5.*(1-x).^2 + lamda*abs(x-1);

lamda = 2;
y2 = 0.5.*(-1-x).^2 + 0.5.*(1-x).^2 + lamda*abs(x-1);

lamda = 3;
y3 = 0.5.*(-1-x).^2 + 0.5.*(1-x).^2 + lamda*abs(x-1);

plot(x,y0,'r');
hold on;
plot(x,y1,'b');
hold on;
plot(x,y2,'y');
hold on;
plot(x,y3,'g');
hold on;
legend('lamda = 0','lamda = 1','lamda = 2','lamda = 3');
%%
lamda = 10000;
x=-50:0.2:50;
y=-50:0.2:50;
[X,Y] = meshgrid(x,y);
Z = (-1-X).^2+(3-Y).^2 + (1-X).^2 + (-1-Y).^2 + lamda*abs(X-Y);
figure
mesh(X,Y,Z);hold on;
%%
x=-50:0.2:50;
y=-50:0.2:50;
[X,Y] = meshgrid(x,y);
Z = X.^2+Y.^2 + (X-2).^2 + (X-3).^2 +(Y+1).^2 + (Y+2).^2 + 20*(X-Y);
figure
contour(X,Y,Z);hold on;
Z1 = X.^2+Y.^2 + (X-2).^2 + (X-3).^2 +(Y+1).^2 + (Y+2).^2 - 20*(X-Y);
contour(X,Y,Z1);
hold on;
plot(x,y);
Z2 = X.^2+Y.^2 + (X-2).^2 + (X-3).^2 +(Y+1).^2 + (Y+2).^2 + 20*abs(X-Y);
figure
contour(X,Y,Z2);
%%
load('optdigitsubset.txt');
[n,m]=size(optdigitsubset);
data = optdigitsubset;
lam = [0,0.1,1,10,100,1000];
Aerr = zeros(6,1);
Terr = zeros(6,1);
for l = 1:6
    apparenterr = zeros(100,1);
    trueerr = zeros(100,1);
for j = 1:100
a = fix(randn(1,1)*1000);
while  1>a || a>554
    %update
    a = fix(randn(1,1)*1000);
end

b = fix(abs(randn(1,1)*1000));
while  555>b || b>1125
    %update
    b = fix(abs(randn(1,1)*1000));
end


a = ceil(rand * 554);
b = 554 + ceil(rand * 571);

subset1 = (optdigitsubset(a,:))';
subset2 = (optdigitsubset(b,:))';
lamda = lam(l);
cvx_begin
    variable A(m)
    variable B(m)
    minimize( (sum(sum_square(subset1 - A))) + (sum(sum_square(subset2 - B))) + lamda * norm(A - B, 1))
cvx_end


%figure(1);
%title('lamda = 0');
%subplot(1,2,1);
%imshow(reshape(A/255,[8 8])),title('Representor 1');
%subplot(1,2,2);
%imshow(reshape(B/255,[8 8])),title('Representor 2');

%CLASSIFIER
truth = [zeros(554,1);ones(571,1)];
result = zeros(1125,1);
for i = 1:1125
      if norm(data(i,:)' - A,2) >= norm(data(i,:)' - B,2)
          result(i,1) = 1;
      %else
          %result(i,1) = 0;
      end
end

err = 0;
Aerr = 0;
for i = 1:1125
    if i ~= a && i ~=b
        if truth(i) ~= result(i)
        err = err + 1;
        end
    else
        if truth(i) ~= result(i)
        Aerr = Aerr + 1;
        end
    end
end

true_avg_err = err/1123;
A_avg_err = Aerr/2;
apparenterr(j) = A_avg_err;
trueerr(j) = true_avg_err;
end
disp(lamda);
disp(mean(apparenterr));
disp(mean(trueerr));
Aerr(l) = mean(apparenterr);
Terr(l) = mean(trueerr);
end
plot(lam,Terr,'g');
hold on;
plot(lam,Aerr,'b');
xlabel('lambda','FontSize',10);
legend('Test Error','Apparent Error');
%%
%sub-gradient
load('optdigitsubset.txt');
[n,m]=size(optdigitsubset);
subset1 = (optdigitsubset(1:554,:))';
subset2 = (optdigitsubset(555:1125,:))';

lamda = 300;
cvx_begin
    variable A(m)
    variable B(m)
    minimize( 1/554*(sum(sum_square(subset1 - repmat(A,1,554)))) + 1/571*(sum(sum_square(subset2 - repmat(B,1,571)))) + lamda * norm(A - B, 1))
cvx_end


figure(1);
subplot(1,2,1);
imshow(reshape(A/255,[8 8])),title('Representor 1');
subplot(1,2,2);
imshow(reshape(B/255,[8 8])),title('Representor 2');
