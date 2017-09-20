function [] = plotValues(r1,r10)
close all;

c1 = 371*389*16*3/(2^20);
for i = 1:16
    r1(i,2) = c1/r1(i,2);
end

figure(1);
hold on;
plot(r1(:,1),r1(:,2));
title('Scaling plot with fixed blurring radius');
xlabel('Threads');
ylabel('Mflops/sec');
hold off;


c2 = 371*389*16*625*3/(2^20);
for i = 1:16
    r10(i,2) = c2/r10(i,2);
end

figure(2);
hold on;
plot(r10(:,1),r10(:,2));
title('Scaling plot with varied blurring radius');
xlabel('Threads');
ylabel('Mflops/sec');
hold off;

end

