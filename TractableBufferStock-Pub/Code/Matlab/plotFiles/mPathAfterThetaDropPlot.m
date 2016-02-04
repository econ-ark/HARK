% This script generates a graph shat shows that path of assets over
% time after a drop in theta

mPathAfterThetaDrop = figure;
axes('XTick',[4],'YTick',[],'XTickLabel','0');
hold on;
plot(timePath,scriptmPath,'.b');
hold off;
axis([-3 80 0 13]);
xlabel('Time');
ylabel('m^e_t');
title('Path of Assets After Drop in Theta');
if UsingMatlab==1
    saveas(mPathAfterThetaDrop,'mPathAfterThetaDropPlot','pdf');
end
