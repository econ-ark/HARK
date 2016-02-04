% This script produces a graph that shows how expected consumption growth
% changes after a change in the transition rate Mho

cGroIncreaseMho = figure;
resetParams;
xmin = 0.1*scriptmEBase; 
xmax = 2.1*scriptmEBase; 
ymin = (min(littleRBase,littleR)-max(thetaBase,theta))/max(rhoBase,rho)...
       -abs(log(biggammaBase)-(littleRBase-theta)/rhoBase);
ymax = 2.5*(log(max(biggamma,biggammaBase))-ymin);
axis([xmin xmax ymin ymax]);
text_x1 = 0.22*xmax; 
text_y1 = LogscriptCtp1OscriptCt(text_x1);
arrow_x1 = 0.75*xmax; 
arrow_x2 = 0.60*xmax; 
arrow_y1 = log(biggammaBase);
arrow_y3 = LogscriptCtp1OscriptCt(arrow_x2);
mho = mhoNew; setValues; FindStableArm;
text_x2 = 0.22*xmax; 
text_y2 = LogscriptCtp1OscriptCt(text_x2);
arrow_y2 = log(biggamma);
arrow_y4 = LogscriptCtp1OscriptCt(arrow_x2);
[Ctp1OCtXa Ctp1OCtYa] = plotMyFunc(@LogscriptCtp1OscriptCt,xmin,xmax);
hold on;
plot(Ctp1OCtX,Ctp1OCtY,'-k',Ctp1OCtXa,Ctp1OCtYa,'--k');
plot([scriptmEBase,scriptmEBase],[ymin,log(biggammaBase)],':k');
plot([scriptmE,scriptmE],[ymin,log(biggamma)],':k');
plot([0,xmax],[log(biggammaBase),log(biggammaBase)],'-k');
plot([0,xmax],[log(biggamma),log(biggamma)],'--k');
%plot([0,xmax],[littleG+mhoBase,littleG+mhoBase],'-k'); % quite far off
%plot([0,xmax],[littleG+mho,littleG+mho],'--k'); % quite far off
plot([0,xmax],[(littleRBase-theta)/rho,(littleRBase-theta)/rho],'-k');
text('Interpreter','latex','FontName',fontname,'FontSize',fontsize,...
     'String','$$ \Delta \log c_{t+1}^{e} \ \rightarrow $$',...
     'Position',[text_x1,text_y1],...
     'HorizontalAlignment','right');
text('Interpreter','latex','FontName',fontname,'FontSize',fontsize,...
     'String','$$ \leftarrow \ \Delta \log \check{c}_{t+1}^{e} $$',...
     'Position',[text_x2,text_y2],...
     'HorizontalAlignment','left');
arrow([arrow_x1 arrow_y1],[arrow_x1 arrow_y2],'TipAngle',12,'Length',6);
arrow([arrow_x2 arrow_y3],[arrow_x2 arrow_y4],'TipAngle',12,'Length',6);


if scriptmE < scriptmEBase
[hx,hy] = format_ticks(gca,...
          {'$\check{m}^{e}$','$\grave{\check{m}}^{e}$','$m^{e}_{t}$'},...
          {'$(r-\theta)/\rho$','$\gamma$','$\grave{\gamma}$','Growth'},...
          [scriptmE scriptmEBase,xmax],...
          [(littleRBase-theta)/rho,log(biggammaBase),log(biggamma),ymax],...
          0,0,[0.04],0,...
          'FontName',fontname,'FontSize',fontsize);
else
[hx,hy] = format_ticks(gca,...
          {'$\grave{\check{m}}^{e}$','$\check{m}^{e}$','$m^{e}_{t}$'},...
          {'$(r-\theta)/\rho$','$\gamma$','$\grave{\gamma}$','Growth'},...
          [scriptmEBase scriptmE,xmax],...
          [(littleRBase-theta)/rho,log(biggammaBase),log(biggamma),ymax],...
          0,0,[0.04],0,...
          'FontName',fontname,'FontSize',fontsize);
end
title('Buffer-Stock Saving: Expected Consumption Growth and Change in Risk',...
      'FontName',fontname,'FontSize',fontsize);

if UsingMatlab==1
    saveas(cGroIncreaseMho,'cGroIncreaseMhoPlot_MO','pdf');
    saveas(cGroIncreaseMho,'cGroIncreaseMhoPlot_MO','eps');
end