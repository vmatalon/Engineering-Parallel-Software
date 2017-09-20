function [] = plotBSC(naive,simd,cache)

figure(1);
x = naive(:,1);
y = naive(:,2);
plot(x,y);
title('Naive Implementation');
xlabel('size of array');
ylabel('average bandwith');

figure(2);
y = simd(:,2);
x = simd(:,1);
plot(x,y);
title('SIMD MemCpy');
xlabel('size of array');
ylabel('average bandwith');

figure(3);
y = cache(:,2);
x = cache(:,1);
plot(x,y);
title('SIMD MemCpy with Cache');
xlabel('size of array');
ylabel('average bandwith');

end