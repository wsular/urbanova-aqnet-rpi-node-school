#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:28:45 2021

@author: matthew
"""

def figure_format(fig):
    
    fig.legend.click_policy="mute"
    fig.legend.location='top_left'

    fig.legend.location='top_right'
    fig.legend.label_text_font_size = "14pt"
    fig.legend.label_text_font = "times"
    fig.legend.label_text_color = "black"

    fig.xaxis.axis_label_text_font_size = "14pt"
    fig.xaxis.major_label_text_font_size = "14pt"
    fig.xaxis.axis_label_text_font = "times"
    fig.xaxis.axis_label_text_color = "black"
    
    fig.yaxis.axis_label_text_font_size = "14pt"
    fig.yaxis.major_label_text_font_size = "14pt"
    fig.yaxis.axis_label_text_font = "times"
    fig.yaxis.axis_label_text_color = "black"
    
    fig.toolbar.logo = None
    fig.toolbar_location = None
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    
    return fig