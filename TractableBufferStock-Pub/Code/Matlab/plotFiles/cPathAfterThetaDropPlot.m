% This script generates a graph shat shows that path of consumption over
% time after a drop in theta

cPathAfterThetaDrop = figure;
axes('XTick',[4],'YTick',[],'XTickLabel','0');
hold on;
plot(timePath,scriptcPath,'.b');
hold off;
axis([-3 80 0 1.5]);
xlabel('Time');
ylabel('c^e_t');
title('Path of Consumption After Drop in Theta');
if UsingMatlab==1;
    saveas(cPathAfterThetaDrop,'cPathAfterThetaDropPlot','pdf');
end
