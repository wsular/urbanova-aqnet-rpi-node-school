#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:58:05 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:24:27 2020

@author: matthew
"""

from bokeh.models import ColumnDataSource, Whisker
from bokeh.models import Panel, Tabs
from bokeh.plotting import figure
from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.plotting import reset_output

#plots = []
#for i in range(3):
#    p = figure()
#    glyphs = [p.line(np.arange(10), np.random.random(10)) for j in range(2)]
#    plots.append(p)
#show(column(*plots))


#whiskers_dict = {'Audubon': Audubon_filtered, 'Adams': Adams_filtered, 'Balboa': Balboa_filtered, 'Browne':Browne_filtered,
#                 'Grant':Grant_filtered, 'Jefferson': Jefferson_filtered,'Lidgerwood':Lidgerwood_filtered,
#                 'Regal':Regal_filtered, 'Sheridan':Sheridan_filtered, 'Stevens':Stevens_filtered}


def plot_stat_diff(location_filtered, df_dictionary):
    
    tabs_list = []
    
    for key,combo in location_filtered.items():
        print(1)
        print(combo['Location'].values[0])
        print(combo['location2_name'].values[0])
        plot_name = combo['Location'].values[0] + '_' + combo['location2_name'].values[0]
        print(plot_name)
        print(2)
        p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Calibrated PM 2.5 (ug/m3)')

        p1.title.text = combo['Location'].values[0] + ' ' + combo['location2_name'].values[0] + ' ' + 'Comparison'    

        
        # Time series of statistically different measurements
       # p1.scatter(combo.index,   combo.PM2_5_corrected, legend = combo['Location'].values[0], color = 'blue',  line_width = 2)
       # p1.scatter(combo.index,   combo.location_PM2_5_corrected, legend = combo['location2_name'].values[0], color = 'red',  line_width = 2)

        # Time series of all measurements
        main_location_name = combo['Location'].values[0]
        comparison_location_name = combo['location2_name'].values[0]
        
        print(main_location_name)
        print(comparison_location_name)
        
        p1.line(df_dictionary[main_location_name].index,   df_dictionary[main_location_name].PM2_5_corrected, legend = main_location_name + '_all', color = 'blue', line_alpha = 0.4, line_width = 2)
        p1.line(df_dictionary[comparison_location_name].index,   df_dictionary[comparison_location_name].PM2_5_corrected, legend = comparison_location_name + '_all', color = 'red', line_alpha = 0.4, line_width = 2)
        
        print((df_dictionary[comparison_location_name]).head())
        
        # Add error bars only to the measurements whose error bars dont overlap
        
        source_error = ColumnDataSource(data=dict(base=combo.index, lower=combo.lower_uncertainty, upper=combo.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
        
        source_error = ColumnDataSource(data=dict(base=combo.index, lower=combo.location_lower, upper=combo.location_upper))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )


        p1.legend.click_policy="hide"

    #source_error = ColumnDataSource(data=dict(base=Audubon.index, lower=Audubon.lower_uncertainty, upper=Audubon.upper_uncertainty))

    #p1.add_layout(
    #    Whisker(source=source_error, base="base", upper="upper", lower="lower")
    #)

        tab = Panel(child=p1, title=combo['Location'].values[0] + '_' + combo['location2_name'].values[0])
        
        tabs_list.append(tab)
    
        tabs = Tabs(tabs=tabs_list)
        
       # output_file('/Users/matthew/Desktop/data/stat_diff_plots/' + plot_name + '.html')
    output_file('/Users/matthew/Desktop/data/stat_diff_plots/' + combo['Location'].values[0] + '.html')
    show(tabs)
    
    print(combo['Location'].values[0])

    
    reset_output()