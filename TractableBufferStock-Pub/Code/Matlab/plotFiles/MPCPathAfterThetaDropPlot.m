% This script generates a graph shat shows that path of MPC over
% time after a drop in theta

MPCPathAfterThetaDrop = figure;
axes('XTick',[4],'YTick',[],'XTickLabel','0');
hold on;
plot(timePath,MPCPath,'.b');
plot([1,length(timePath)],[kappa,kappa],':k');
text(length(timePath)/2,kappa,'\uparrow','HorizontalAlignment','center','VerticalAlignment','top');
text(length(timePath)/2,kappa*6/7,'Perfect Foresight MPC','HorizontalAlignment','center','VerticalAlignment','top');
hold off;
axis([-3 80 0 .13]);
xlabel('Time');
ylabel('\kappa_t');
title('Path of MPC After Drop in Theta');
if UsingMatlab==1
    saveas(MPCPathAfterThetaDrop,'MPCPathAfterThetaDropPlot','pdf');
end
