% This script generates a graph that shows the path of consumption for a
% small open economy after a drop in theta

SOENoStakescPath = figure;
axes('XTick',[1],'YTick',[],'XTickLabel','0');
hold on;
plot(timePath,CensusMeans(:,scriptcPos),'.b');
hold off;
axis([-3 length(CensusMeans)+2 0 0.8]);
xlabel('Time');
ylabel('c');
title('Path of Consumption for a Small Open Economy Without Stakes');
if UsingMatlab==1
    saveas(SOENoStakescPath,'SOENoStakescPathPlot','pdf');
end
