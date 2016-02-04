%% BEGIN HEADER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name: arrowflow.m
%
% Description:
%                  Draws arrows of motion for phase diagrams
%
% Usage:
%                  arrowflow(x,y,sx,sy,r);
%
% Input:    
%                  x,y    : initial position coordinates
%                           default is 0,0
%                  sx,sy  : scaling of arrows along x and y coordinates
%                           as a percentage of axis length, e.g. 10%
%                           default is 10,10
%                  r      : angle of rotation
%                           r=1  --> East-North
%                           r=2  --> North-West
%                           r=3  --> West-South
%                           r=4  --> South-East
%                           default is 1
%
% Output:    
%                  h      : handles of plot of the arrows
%
% Options:
%                  None
%                  arrow properties may be modified in script below
%                  refer to arrow.m script for available options
%
%
% Function Calls:
%                  arrow.m, created by Erik A. Johnson, version of 11/15/02
%
% Required Files:
%                  None
%
% Matlab Version:
%                  Tested on Matlab 2008a
%
% Example:  
%                  figure;
%                  plot(0:0,1:1);
%                  plot(0:1,1:0);
%                  arrowflow(1/3,2/3,0,0,1);
%                  arrowflow(2/3,2/3,0,0,3);
%
% Issues:
%                  To keep arrows of same length requires: axis square;
%
% To Do :
%                  set arrows to same length without call to: axis square;
%                  pass options,
%                  e.g. 'LineWidth', 'BaseAngle', 'TipAngle', 'Length'
%                  I don't plan to do this, so please feel free to improve.
%
% Change Log:
%                  04/25/2009: 
%                  created by Patrick Toche (ptoche@cityu.edu.hk)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BEGIN FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function h = arrowflow(x,y,sx,sy,r)

if ~exist('x','var');
    x = 0;
end;

if ~exist('y','var');
    y = 0;
end;

if ~exist('sx','var');
    sx = 10;
end;

if ~exist('sy','var');
    sy = 10;
end;

if ~exist('r','var');
    r = 1;
end;

x1 = x; 
y1 = y; 

lim=get(gca,'XLim');
lx=sx/100*abs((lim(2)-lim(1)));

lim=get(gca,'YLim');
ly=sy/100*abs((lim(2)-lim(1)));

if r == 1;
    x2 = x1+lx;
    y2 = y1;
    x3 = x1;
    y3 = y1+ly;
end;

if r == 2;
    x2 = x1;
    y2 = y1+ly;
    x3 = x1-lx;
    y3 = y1;
end;

if r == 3;
    x2 = x1-lx;
    y2 = y1;
    x3 = x1;
    y3 = y1-ly;
end;

if r == 4;
    x2 = x1+lx;
    y2 = y1;
    x3 = x1;
    y3 = y1-ly;
end;

h1 = arrow([x1 y1],[x2 y2],'TipAngle',12,'Length',6);
h2 = arrow([x1 y1],[x3 y3],'TipAngle',12,'Length',6);
h  = [h1 h2];
warning(['arrowflow currently requires the axis square property']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%