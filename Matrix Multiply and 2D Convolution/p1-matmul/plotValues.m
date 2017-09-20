function [] = plotValues(forr,taskk,pthss)

figure(1);
hold on;
plot(forr(:,1),forr(:,2),'r');
plot(taskk(:,1),taskk(:,2),'b');
plot(pthss(:,1),pthss(:,2),'g');
title('Matrix Multiply: scaling plot');
xlabel('Threads');
ylabel('Mflops/sec');
legend('OpenMP For','OpenMP Task','Pthreads','Location',...
    'SouthEast');
hold off; 

end

