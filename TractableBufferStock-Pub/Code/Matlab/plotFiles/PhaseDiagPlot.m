% This script produces a graph that shows the phase diagram for the
% individual's consumption problem.

PhaseDiag = figure;
graph2top = scriptmE*1.5;
[cEdelZeroX cEdelZeroY] = plotMyFunc(@scriptcEDelEqZero,0,20);
[mEdelZeroX mEdelZeroY] = plotMyFunc(@scriptmEDelEqZero,0,20);
[cEx2 cEy2] = plotMyFunc(@cE,0,graph2top);
axes('XTick',[],'YTick',[]);
hold on;
plot(cEx2,cEy2,':b',cEdelZeroX,cEdelZeroY,'-k',mEdelZeroX,mEdelZeroY,'-k');
text(scriptmE/2,cE(scriptmE/2),'Stable Arm \rightarrow  ','HorizontalAlignment','right');
text(scriptmE,scriptcE,['\uparrow';'SS      '],'VerticalAlignment','top');
text(1.25*scriptmE,scriptcEDelEqZero(1.25*scriptmE),'\Deltac_{t}^{e}=0 \rightarrow  ','HorizontalAlignment','right');
text(scriptmE/3,scriptmEDelEqZero(scriptmE/3),'\Deltam_{t}^{e}=0 \downarrow','VerticalAlignment','bottom');
plot_arrow(scriptmE,scriptcE*0.5,1.1*scriptmE,scriptcE/2);
plot_arrow(scriptmE,scriptcE*0.5,scriptmE,0.3*scriptcE,'headwidth',0.40);
plot_arrow(scriptmE,1.5*scriptcE,0.9*scriptmE,1.5*scriptcE);
plot_arrow(scriptmE,1.5*scriptcE,scriptmE,1.7*scriptcE,'headwidth',0.40);
hold off;
axis([0 graph2top 0 1.2*scriptcEDelEqZero(graph2top)]);
xlabel('m^{e}_{t}');
ylabel('c^{e}_{t}');
title('Phase Diagram with Stable Arm');
if UsingMatlab==1
    saveas(PhaseDiag,'PhaseDiagPlot','pdf');
end
