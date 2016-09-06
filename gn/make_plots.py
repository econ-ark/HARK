# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:44:30 2015

@author: ganong
"""


from rpy2 import robjects
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr

# The R 'print' function
rprint = robjects.globalenv.get("print")
stats = importr('stats')
grdevices = importr('grDevices')
base = importr('base')
#datasets = importr('datasets')

grid.activate()

import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib.ggplot2 import element_blank, \
                               theme_bw, \
                               theme, \
                               element_rect, \
                               element_text 
                               #element_line
from rpy2.robjects import pandas2ri
pandas2ri.activate()                                                        


robjects.r('''
    library(RColorBrewer)
    library(grid)
    #print(brewer.pal(8,"Set3")) Greys
    palette <- brewer.pal("Greys", n=9)
    #print(palette)
  color_background = palette[2]
  color_grid_major = palette[3]
  color_axis_text = palette[6]
  color_axis_title = palette[7]
  color_title = palette[9]
  #palette_lines <- brewer.pal("Dark2", n=3)
  palette_lines <- brewer.pal("Set2", n=8)
''')

size = 9
fte_theme = theme(**{'axis.ticks':element_blank(),
      'panel.background':element_rect(fill=robjects.r.color_background, color=robjects.r.color_background),
      'plot.background':element_rect(fill=robjects.r.color_background, color=robjects.r.color_background),
      'panel.border':element_rect(color=robjects.r.color_background),
      'panel.grid.minor':element_blank(),
      'axis.ticks':element_blank(),
      'legend.position':"right",
      'legend.background': element_rect(fill="transparent"),
      'legend.text': element_text(size=size,color=robjects.r.color_axis_title),
      'legend.title': element_text(size=size,color=robjects.r.color_axis_title),
      'plot.title':element_text(color=robjects.r.color_title, size=10, vjust=1.25),
      'axis.text.x':element_text(size=size,color=robjects.r.color_axis_text),
      'axis.text.y':element_text(size=size,color=robjects.r.color_axis_text),
      'axis.title.x':element_text(size=size,color=robjects.r.color_axis_title, vjust=0),
      #'panel.grid.major':element_line(color=robjects.r.color_grid_major,size=.25),
      'axis.title.y':element_text(size=size,color=robjects.r.color_axis_title,angle=90)})

#??? efficiently change legend titles
#right now it takes two legend calls to make this work
#alternatives that tried and failed
#base_plot = lambda gr_name = 'variable': ggplot2.aes_string(x='x', y='value',group=gr_name,colour=gr_name, shape = gr_name)
#colors = ggplot2.scale_colour_manual(values=robjects.r.palette_lines, name = ltitle)

pandas2ri.activate() 
#set up basic, repetitive plot features
base_plot =  ggplot2.aes_string(x='x', y='value',group='variable',colour='variable', shape = 'variable')
line = ggplot2.geom_line()
point = ggplot2.geom_point() 
bar = ggplot2.geom_bar(stat="identity")
vert_line_onset = ggplot2.geom_vline(xintercept=-1, linetype=2, colour="red", alpha=0.25)           
vert_line_exhaust = ggplot2.geom_vline(xintercept=5, linetype=2, colour="red", alpha=0.25)  
ltitle = "crazy"         
ltitle_default = 'Variable'
#colors = lambda ltitle = ltitle_default: ggplot2.scale_colour_manual(values=robjects.r.palette_lines, name = ltitle)
colors = ggplot2.scale_colour_manual(values=robjects.r.palette_lines)
legend_t_c = lambda ltitle = ltitle_default: ggplot2.scale_color_discrete(name = ltitle) 
legend_t_s = lambda ltitle = ltitle_default: ggplot2.scale_shape_discrete(name = ltitle)
loc_default = robjects.r('c(1,0)')
legend_f  = lambda loc = loc_default: ggplot2.theme(**{'legend.position':loc, 'legend.justification':loc})
ggsave = lambda filename, plot: robjects.r.ggsave(filename="~/dropbox/hampra/out2/" + filename + ".pdf", plot=plot, width = 6, height = 4)

colors_alt = ggplot2.scale_colour_manual(values=robjects.r.palette_lines[1])
shape_alt = ggplot2.scale_shape_manual(values=17)

#create a new function straight from the R enviornment
#xx in principle it should be possible to load  class using what's already been imported 
#above, but I'm not going to worry about that for now
ggplot2_env = robjects.baseenv['as.environment']('package:ggplot2')

class GBaseObject(robjects.RObject):
    @classmethod
    def new(*args, **kwargs):
        args_list = list(args)
        cls = args_list.pop(0)
        res = cls(cls._constructor(*args_list, **kwargs))
        return res
        
class Annotate(GBaseObject):
    _constructor = ggplot2_env['annotate']
annotate = Annotate.new