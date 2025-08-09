import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection


def generate_telemetry_plots(lap_telemetry):
    if lap_telemetry.empty:
        return None, None, None

    template = "plotly_dark"
    
    # 1. SPEED PLOT
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=lap_telemetry['Distance'], y=lap_telemetry['Speed'],
                                   mode='lines', name='Speed', line=dict(color='#FF1801')))
    fig_speed.update_layout(title="Speed Trace", xaxis_title="Distance (m)",
                            yaxis_title="Speed (Km/h)", template=template)

    # 2. THROTTLE PLOT
    fig_throttle = go.Figure()
    fig_throttle.add_trace(go.Scatter(x=lap_telemetry['Distance'], y=lap_telemetry['Throttle'],
                                      mode='lines', name='Throttle', line=dict(color='#00D2BE')))
    fig_throttle.update_layout(title="Throttle Application", xaxis_title="Distance (m)",
                               yaxis_title="Throttle (%)", template=template)

    # 3. BRAKE PLOT
    fig_brake = go.Figure()
    fig_brake.add_trace(go.Scatter(x=lap_telemetry['Distance'], y=lap_telemetry['Brake'].astype(int),
                                   mode='lines', name='Brake', line=dict(color='#E10600'), fill='tozeroy'))
    fig_brake.update_layout(title="Braking Points", xaxis_title="Distance (m)",
                            yaxis_title="Brake Applied", template=template)
    
    # 4. RPM PLOT
    fig_rpm = go.Figure()
    fig_rpm.add_trace(go.Scatter(x=lap_telemetry['Distance'], y=lap_telemetry['RPM'], mode = 'lines',name='RPM',line=dict(color='#ff4c4c')))
    fig_rpm.update_layout(title='Engine RPM',xaxis_title='Distance (m)',yaxis_title='RPM',template = template)

    # 5. GEAR PLOT
    fig_gear = go.Figure()
    fig_gear.add_trace(go.Scatter(x=lap_telemetry['Distance'], y=lap_telemetry['nGear'],
                                   mode='lines', name='Gear', line=dict(color='#ff7f7f', shape='hv')))
    fig_gear.update_layout(title="Gear Shifts", xaxis_title="Distance (m)",
                            yaxis_title="Gear", template=template, yaxis=dict(tickmode='linear', dtick=1))

    return fig_speed, fig_throttle, fig_brake, fig_rpm, fig_gear 

def generate_strategy_plot(driver_laps):
    """
    Generates a Gantt chart style plot to visualize tyre strategy.
    This version is built manually with graph_objects for reliability.
    """
    if driver_laps.empty:
        return None

    stints = driver_laps.groupby('Stint').agg(
        StintStart=('LapNumber', 'min'),
        StintEnd=('LapNumber', 'max'),
        Compound=('Compound', 'first'),
        Driver=('Driver', 'first')
    ).reset_index()

    # Define tyre colors
    compound_colors = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFF200',
        'HARD': '#F0F0F0',
        'INTERMEDIATE': '#44D744',
        'WET': '#2772FF'
    }

    fig = go.Figure()
    
    seen_compounds = set()

    for i, stint in stints.iterrows():
        compound = stint['Compound']
        color = compound_colors.get(compound, 'grey')
        
        fig.add_trace(go.Bar(
            y=[stint['Driver']], 
            x=[stint['StintEnd'] - stint['StintStart'] + 1], 
            base=stint['StintStart'], 
            orientation='h',
            marker_color=color,
            name=compound,
            text=f"Stint {stint['Stint']}",
            textposition='inside',
            insidetextanchor='middle',
            hoverinfo='text',
            hovertext=f"<b>Stint {stint['Stint']}</b><br>Laps: {stint['StintStart']}-{stint['StintEnd']}<br>Compound: {compound}",
            showlegend=(compound not in seen_compounds) 
        ))
        seen_compounds.add(compound)

    # 3. Style the layout
    fig.update_layout(
        title="Tyre Strategy and Stint Lengths",
        xaxis_title="Lap Number",
        yaxis_title="",
        yaxis_showticklabels=False,
        template="plotly_dark",
        barmode='stack', 
        legend_title_text='Tyre Compound'
    )
    
    return fig
