% This script produces a graph comparing consumption under uncertainty to
% consumptions under perfect foresight

ConsFunc = figure;
graph1top = scriptmE*5;
[cEx cEy] = plotMyFunc(@cE,0,graph1top);
[cEPFx cEPFy] = plotMyFunc(@cEPF,0,graph1top);
[deg45x deg45y] = plotMyFunc(@Identity, 0,graph1top);
axes('XTick',[],'YTick',[]);
hold on;
plot(cEx,cEy,'-k',cEPFx,cEPFy,'--k',deg45x,deg45y,':k');
text(0.8*cEPF(graph1top), 0.8*cEPF(graph1top),' \leftarrow 45 Degree Line');
text(scriptmE/3,cE(scriptmE/3),' \leftarrow Consumption Function');
text(3/4*graph1top,cEPF(3/4*graph1top),'Perfect Foresight \rightarrow  ','HorizontalAlignment','right');
hold off;
axis([0 graph1top 0 cEPF(graph1top)]);
xlabel('m^{e}_{t}');
ylabel('c');
title('Consumption Under Uncertainty vs. Perfect Foresight');
if UsingMatlab==1;
    saveas(ConsFunc,'ConsFuncPlot','pdf');
end
