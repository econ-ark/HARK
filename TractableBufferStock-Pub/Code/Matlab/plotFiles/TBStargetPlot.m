% This script produces a graph showing the target level of assets and
% consumption as related to sustainable consumption

TBStarget = figure;
mMinNew = 0;
mMaxNew = 20;
cMinNew = 0;
cMaxNew = 1.5;
axes('XTick',[],'YTick',[]);
hold on;
plot(cEx,cEy,'-k',mEdelZeroX,mEdelZeroY,':b');
text(scriptmEBase,1.02*scriptcEBase,'Target \rightarrow ','HorizontalAlignment','right');
text(scriptmE,0.98*scriptcE,' \leftarrow Sustainable c');
text(0.7*scriptmEBase,0.85*scriptcEBase,'c(m) \rightarrow','HorizontalAlignment','right');
hold off;
axis([mMinNew mMaxNew cMinNew cMaxNew]);
xlabel('m');
ylabel('c');
title('Target Level of Assets and Sustainable Consumption');
if UsingMatlab==1
    saveas(TBStarget,'TBStargetPlot','pdf');
end
