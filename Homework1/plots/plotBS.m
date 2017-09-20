function [] = plotBS(naive,simd,cache)

figure(1);
x = naive(:,1);
y = naive(:,2);
plot(x,y);
title('Memory Bandwith: Naive');
xlabel('size of array: bytes');
ylabel('average bandwith: Mbps');

figure(2);
y = simd(:,2);
x = simd(:,1);
plot(x,y);
title('Memory Bandwith: SIMD MemCpy');
xlabel('size of array: bytes');
ylabel('average bandwith: Mbps');

figure(3);
y = cache(:,2);
x = cache(:,1);
plot(x,y);
title('Memory Bandwith: SIMD MemCpyCache');
xlabel('size of array: bytes');
ylabel('average bandwith: Mbps');

end

