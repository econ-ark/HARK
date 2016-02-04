% This script generates a graph that shows the path of assets and
% consumption after a decrease in theta

mcPathFig = figure;
[cExNew cEyNew] = plotMyFunc(@cE,0,scriptmE+2);
axes('XTick',[],'YTick',[]);
hold on;
mcPath = SimGeneratePath(scriptmEBase,100);
plot(mcPath(:,1),mcPath(:,2),'.b');
plot(cEx,cEy,'-k',cExNew,cEyNew,'-k',mEdelZeroX,mEdelZeroY,':b');
text(scriptmEBase,1.02*scriptcEBase,'Orig Target \rightarrow ','HorizontalAlignment','right');
text(scriptmE,scriptcE*1.04,'New Target \downarrow','HorizontalAlignment','right');
text(1.25*scriptmEBase,1.2*scriptcEBase,'Orig c(m) \rightarrow','HorizontalAlignment','right');
text(scriptmEBase,cE(scriptmEBase),' \leftarrow New c(m)');
hold off;
axis([mMinNew scriptmE+2 cMinNew, 1.3*scriptcEBase]);
xlabel('m');
ylabel('c');
title('Path of Assets and Consumption After Drop in Theta');
if UsingMatlab==1
    saveas(mcPathFig,'mcPathPlot','pdf');
end
