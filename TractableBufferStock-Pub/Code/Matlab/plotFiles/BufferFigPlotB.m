% This script produces a graph that shows how buffer saving changes with an
% increase in the interest rate

BufferFigB = figure;
[Ctp1OCtXa Ctp1OCtYa] = plotMyFunc(@LogscriptCtp1OscriptCt,0.25*scriptmEBase,2.5*scriptmEBase);
axes('XTick',[scriptmEBase scriptmE],'YTick',[(littleRBase-theta)/rho (littleR-theta)/rho littleG+mho],'XTickLabel',['mE ';'mE`'],'YTickLabel',[' (r-theta)/rho';'(r`-theta)/rho';'         gamma']);
hold on;
plot(Ctp1OCtX,Ctp1OCtY,'-k',Ctp1OCtXa,Ctp1OCtYa,'--k');
plot([0,graph3top],[(littleRBase-theta)/rho,(littleRBase-theta)/rho],'-k');
plot([scriptmEBase,scriptmEBase],[HorizAxis,1.5*LogscriptCtp1OscriptCt(1.5)],':k');
plot([scriptmE,scriptmE],[HorizAxis,1.5*LogscriptCtp1OscriptCt(1.5)],':k');
plot([0,graph3top],[littleG+mho,littleG+mho],'-k');
plot([0,graph3top],[(littleR-theta)/rho,(littleR-theta)/rho],'--k');
text(scriptmE/2,LogscriptCtp1OscriptCt(scriptmE/2),' \leftarrow \Delta Log C`_{t+1}');
text(5/6*scriptmEBase,ypos,'\Delta Log C_{t+1} \rightarrow ','HorizontalAlignment','right');
plot_arrow(1.4*scriptmE,(littleRBase-theta)/rho,1.4*scriptmE,(littleR-theta)/rho,'headwidth',3.5);
plot_arrow(1.4*scriptmE,LogscriptCtp1OscriptCt(scriptmE*1.4)-(littleR-littleRBase)*2/(rho*3),1.4*scriptmE,LogscriptCtp1OscriptCt(scriptmE*1.4),'headwidth',4.5);
hold off;
axis([0 2.5*scriptmEBase HorizAxis 0.25]);
xlabel('m^{e}_{t}');
ylabel('Growth');
title('Buffer Saving Changes With Interest Rate');
if UsingMatlab==1
    saveas(BufferFigB,'BufferFigPlotB','pdf');
end
