import streamlit as st
import fastf1
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Load session
fastf1.Cache.enable_cache('./fastf1_cache')  # Optional: Enable caching for faster load
session = fastf1.get_session(2021, 'Austrian Grand Prix', 'Q')
session.load()

# Get telemetry
lap = session.laps.pick_drivers('HAM').pick_fastest()
tel = lap.get_telemetry()

x = tel['X'].to_numpy()
y = tel['Y'].to_numpy()
gear = tel['nGear'].to_numpy().astype(int)

# Normalize gear values for color mapping (1 to 8)
gear_min, gear_max = 1, 8
norm_gear = (gear - gear_min) / (gear_max - gear_min)

# Choose a matching Streamlit color theme (e.g., 'Turbo', 'Viridis')
colorscale = 'Turbo'
colors = sample_colorscale(colorscale, norm_gear)

# Create individual segments for each gear shift section
segments = []
for i in range(len(x) - 1):
    segments.append(go.Scattergl(
        x=[x[i], x[i + 1]],
        y=[y[i], y[i + 1]],
        mode='lines',
        line=dict(color=colors[i], width=4),
        hoverinfo='none',
        showlegend=False
    ))

# Create the layout
layout = go.Layout(
    title=dict(
        text=f"<b>Fastest Lap Gear Shift Visualization</b><br>{lap['Driver']} - {session.event['EventName']} {session.event.year}",
        x=0.5
    ),
    xaxis=dict(showgrid=False, visible=False),
    yaxis=dict(showgrid=False, visible=False, scaleanchor='x', scaleratio=1),
    margin=dict(l=20, r=20, t=80, b=20),
    height=700
)

# Build and display in Streamlit
fig = go.Figure(data=segments, layout=layout)
st.plotly_chart(fig, use_container_width=True)
