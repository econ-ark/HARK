% This script generates a graph that shows the path of consumption for a
% small open economy after a drop in theta

SOEStakescPathAfterThetaDrop = figure;
axes('XTick',[4],'YTick',[],'XTickLabel','0');
hold on;
plot(timePath,CensusMeans(:,scriptcPos),'.b');
hold off;
axis([-3 length(CensusMeans)+2 0 1.3]);
xlabel('Time');
ylabel('c');
title('Path of Consumption After Drop in Theta for a Small Open Economy With Stakes');
if UsingMatlab==1
    saveas(SOEStakescPathAfterThetaDrop,'SOEStakescPathAfterThetaDropPlot','pdf');
end
