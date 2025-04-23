import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def combine_plots(plots, title = None, height = 400, width = None, cols = 2):
    n_figs = len(plots)
    if cols is None:
        cols = n_figs
    rows = int(np.ceil(n_figs / cols))
    
    # Use subtitle titles from plots
    subplot_titles = [fig.layout.title.text if hasattr(fig.layout, 'title') and fig.layout.title.text else f"Subplot {i+1}" for i, fig in enumerate(plots)]

    combined_fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    # --- Define common axis properties to copy ---
    # Add more properties here if needed (e.g., tickformat, tickvals, ticktext)
    AXIS_PROPERTIES_TO_COPY = [
        'title', 'titlefont', 'range', 'type', 'autorange', 'showgrid', 'gridcolor', 'gridwidth',
        'zeroline', 'zerolinecolor', 'zerolinewidth', 'showline', 'linecolor', 'linewidth', 'mirror',
        'ticks', 'ticklen', 'tickwidth', 'tickcolor', 'showticklabels', 'tickfont', 'tickangle', 'fixedrange'
    ]

    any_original_legend = False
    legend_group_titles = {} # To store titles for legend groups
    
    for i, fig in enumerate(plots):
        row = i // cols + 1
        col = i % cols + 1
        # Use subplot title or a generic name for the legend group title
        group_title = subplot_titles[i] if subplot_titles and subplot_titles[i] else f"Plot {i+1}"
        legend_group_id = f"group{i+1}" # Unique group ID remains important

        # Store the title for this group
        legend_group_titles[legend_group_id] = group_title
        
        # Check if the original figure had a legend visible
        original_showlegend = fig.layout.showlegend if hasattr(fig.layout, 'showlegend') else False
        if original_showlegend:
             any_original_legend = True

        # Add Traces with Legend Grouping
        for trace in fig.data:
            trace.legendgroup = legend_group_id
            trace.legendgrouptitle = {'text': group_title} # Assign group title directly to trace

            original_trace_showlegend = trace.showlegend if hasattr(trace, 'showlegend') else True
            trace.showlegend = original_trace_showlegend and original_showlegend

            combined_fig.add_trace(trace, row=row, col=col)
            
        # ---  Axis Property Copying ---
        axis_idx = (row - 1) * cols + col
        xaxis_name = f"xaxis{axis_idx}"
        yaxis_name = f"yaxis{axis_idx}"
        src_layout = fig.layout
        src_xaxis = src_layout.xaxis if hasattr(src_layout, 'xaxis') else None
        src_yaxis = src_layout.yaxis if hasattr(src_layout, 'yaxis') else None
        tgt_xaxis = combined_fig.layout[xaxis_name]
        tgt_yaxis = combined_fig.layout[yaxis_name]
        xaxis_update = {}
        yaxis_update = {}
        if src_xaxis:
             for prop in AXIS_PROPERTIES_TO_COPY:
                 if hasattr(src_xaxis, prop):
                     value = getattr(src_xaxis, prop)
                     if value is not None: xaxis_update[prop] = value
             if hasattr(src_xaxis, 'title') and src_xaxis.title and src_xaxis.title.text:
                 xaxis_update['title_text'] = src_xaxis.title.text
        if src_yaxis:
             for prop in AXIS_PROPERTIES_TO_COPY:
                 if hasattr(src_yaxis, prop):
                     value = getattr(src_yaxis, prop)
                     if value is not None: yaxis_update[prop] = value
             if hasattr(src_yaxis, 'title') and src_yaxis.title and src_yaxis.title.text:
                 yaxis_update['title_text'] = src_yaxis.title.text
        if xaxis_update: tgt_xaxis.update(xaxis_update)
        if yaxis_update: tgt_yaxis.update(yaxis_update)
        # --- End Axis Copying ---
        
    # --- Update Overall Layout for Legend ---
    layout_update = {
        'height': height,
        'width': width,
        'title_text': title,
        'showlegend': True,
        # Add spacing between legend groups visually
        'legend_tracegroupgap': 20 # Adjust pixel gap as needed
    }

    combined_fig.update_layout(**layout_update)

    return combined_fig

