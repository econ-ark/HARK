% This script produces a graph that shows the buffer savings (?)

BufferFigA = figure;
string1(1) = {'\uparrow                    '};
string1(2) = {'\Omega\nablac_{t+1}(m^{e}_t)'};
string1(3) = {'\downarrow                  '};
HorizAxis = -0.08;
graph3top = scriptmE*2.5;
[Ctp1OCtX Ctp1OCtY] = plotMyFunc(@LogscriptCtp1OscriptCt,0.25*scriptmE,2.5*scriptmE);
axes('XTick',[scriptmE],'YTick',[(littleR-theta)/rho littleG+mho],'XTickLabel',['mE'],'YTickLabel',['(r-theta)/rho';'        gamma']);
hold on;
plot(Ctp1OCtX,Ctp1OCtY,'-k');
plot([scriptmE,scriptmE],[HorizAxis,1.5*LogscriptCtp1OscriptCt(1.5)],':k');
plot([0,graph3top],[littleG+mho,littleG+mho],'-k');
plot([0,graph3top],[(littleR-theta)/rho,(littleR-theta)/rho],'-k');
text(scriptmE*1.6,0.5*(LogscriptCtp1OscriptCt(2*scriptmE)+(littleR-theta)/rho),string1,'HorizontalAlignment','left');
text(scriptmE/2,LogscriptCtp1OscriptCt(scriptmE/2),' \leftarrow \Delta Log C_{t+1}\approx \rho^{-1}(r-\vartheta)+\Omega\nablac_{t+1}(m^{e}_t)');
hold off;
axis([0 2.5*scriptmE HorizAxis 0.25]);
xlabel('m^{e}_{t}');
ylabel('Growth');
title('Buffer Saving by Level of Assets');
if UsingMatlab==1
    saveas(BufferFigA,'BufferFigPlotA','pdf');
end
