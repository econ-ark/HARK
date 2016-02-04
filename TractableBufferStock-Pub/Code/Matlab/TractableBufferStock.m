% The main tractable buffer stock file that runs the model

clear all;
close all;

% The preferred program is Matlab. Octave is still in development.
% Octave does not currently support the saveas() command. 
% Main version number should be zero/below Matlab's current 2008b version.

ver = version();
octavevercheck = 0;
UsingMatlab = 0;
if str2num(ver(1))>octavevercheck;
    UsingMatlab = 1 ;
end

thispath = mfilename('fullpath');
thisfile = mfilename;
pathlength = size(thispath,2);
namelength = size(thisfile,2);
CurDir = thispath(1:(pathlength-namelength));
cd(CurDir);
path(CurDir,path);
path([CurDir 'SOEfiles'],path);
path([CurDir 'indivScripts'],path);
path([CurDir 'indivFuncs'],path);
path([CurDir 'plotFiles'],path);
cd('figures');

warning off MATLAB:nearlySingularMatrix;
format long;
globalizeTBSvars;
initializeParams;
scriptmEBase = scriptmE;
scriptcEBase = scriptcE;
kappaEBase = kappaE;
biggammaBase = biggamma;
FindStableArm;

warning(['The order in which the plots are created can matter.']);
warning(['Some plots require other plots to be created first.']);
warning(['The "latex" interpreter supports only the Helvetica font.']);
warning(['The "tex" interpreter supports a limited number of fonts, Greek letters, and symbols.']);
warning(['To use a system-independent fixed-width font, set FixedWidth']);

% Select Font Properties for all Plots
% set(text_handle,'FontName','FixedWidth'); %system-independent fonts
fontname = 'Helvetica';
fontsize = 10;

%error('The test ends here. Comment out this line of code after debugging')


% Plot the Phase Diagram
PhaseDiagPlot;


% Plot the Consumption Function
ConsFuncPlot;


% Plot Expected Consumption Growth
BufferFigPlotA;


% Plot Change in the Interest Rate
littleRNew = littleRBase + 0.03;
BufferFigPlotB; % First Run BufferFigPlotA


% Plot Change in the Transition Rate
mhoNew = mhoBase + 0.1;
cGroIncreaseMhoPlot; % First Run BufferFigPlotA


% Plot Target Level of Assets
TBStargetPlot;


%Plot Change in Time Preference 
thetaNew = thetaBase - 0.04;
mcPathPlot; % First Run TBStargetPlot


error('The test ends here. Comment out this line of code after debugging')


%Plot Dynamics after Change in Time Preference
HowMany = 75;
scriptmPath = [scriptmEBase;scriptmEBase;scriptmEBase;mcPath(1:HowMany,1)];
scriptcPath = [scriptcEBase;scriptcEBase;scriptcEBase;mcPath(1:HowMany,2)];
j = 5;
MPCPath = [kappaEBase;kappaEBase;kappaEBase;kappaEBase];
while j<=length(scriptmPath)
    MPCPath = [MPCPath;D(@cE,scriptmPath(j))];
    j = j+1;
end
timePath = transpose(1:length(scriptcPath));
cPathAfterThetaDropPlot;
mPathAfterThetaDropPlot;
MPCPathAfterThetaDropPlot;


%Plot Dynamics of Small Open Economy after Change in Time Preference
resetParams; FindStableArm; VerboseOutput=0;
setupSOE;
CensusMakeStakes;
for j=1:4
    AddNewGen([scriptbE scriptN bigG]);
end
theta = thetaNew; setValues; FindStableArm;
for j=1:75
    AddNewGen([scriptbE scriptN bigG]);
end
timePath = transpose(1:length(CensusMeans));
SOEStakescPathAfterThetaDropPlot;

CensusMakeNoStakes;
for j=1:100
    AddNewGen([0 scriptN bigG]);
end
timePath = transpose(1:length(CensusMeans));
SOENoStakescPathPlot;


disp('This is all I can do for now.');