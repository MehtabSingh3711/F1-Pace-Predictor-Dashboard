from datetime import timedelta
from config import DRIVERS_2024
import pandas as pd

def generate_narr(driver_laps_df, driver_code, circuit_name, session_type):
    driver_name = DRIVERS_2024.get(driver_code, "Unknown")
    
    if driver_laps_df.empty:
        return f"<h3>Analysis for {driver_name}</h3><p>No valid lap data was found for {driver_name} in the {session_type} at {circuit_name}. They may not have set a representative lap time.</p>", ""

    # Fastest Lap
    fastest_lap_row = driver_laps_df.loc[driver_laps_df['LapTime'].idxmin()]
    fastest_lap_time = fastest_lap_row['LapTime']
    fastest_lap_number = int(fastest_lap_row['LapNumber'])
    compound = fastest_lap_row['Compound']
    avg_speed = fastest_lap_row[['SpeedFL', 'SpeedST', 'SpeedI1', 'SpeedI2']].mean(skipna=True)

    # Longest Stint
    stint_lengths = driver_laps_df.groupby('Stint')['LapNumber'].count()
    longest_stint = stint_lengths.idxmax()
    longest_stint_length = stint_lengths.max()
    longest_stint_compound = driver_laps_df[driver_laps_df['Stint'] == longest_stint]['Compound'].mode().iloc[0]

    # Sector Performance
    best_s1 = driver_laps_df['Sector1Time'].min()
    best_s2 = driver_laps_df['Sector2Time'].min()
    best_s3 = driver_laps_df['Sector3Time'].min()

    s1 = best_s1.total_seconds() if pd.notnull(best_s1) else "N/A"
    s2 = best_s2.total_seconds() if pd.notnull(best_s2) else "N/A"
    s3 = best_s3.total_seconds() if pd.notnull(best_s3) else "N/A"

    # Build narrative
    narrative = f"""
    <h3>{driver_name}'s {session_type} Analysis at {circuit_name}</h3>
    <ul>
        <li>Fastest lap: <strong style="color:#e10600;">Lap {fastest_lap_number}</strong> with a time of <strong style="color:#e10600;">{fastest_lap_time}</strong> on <strong style="color:#e10600;">{compound} tyres</strong>.</li>
        <li>Average speed during that lap: <strong style="color:#e10600;">{avg_speed:.1f} km/h</strong>.</li>
        <li>Longest stint: <strong style="color:#e10600;">{longest_stint_length} laps</strong> on <strong style="color:#e10600;">{longest_stint_compound} tyres</strong>.</li>
    </ul>
    """
    sector = f"""
    <h3>Sector Insights</h3>
    <ul>
        <li>Best Time In Sector 1: <span style="color:#e10600;"><strong>{s1} s</strong></span></li>
        <li>Best Time In Sector 2: <span style="color:#e10600;"><strong>{s2} s</strong></span></li>
        <li>Best Time In Sector 3: <span style="color:#e10600;"><strong>{s3} s</strong></span></li>
    </ul>
    """
    return narrative, sector

def generate_nerd_stats(lap_df, telemetry_df, driver_code, circuit_name, session_type):
    driver_name = DRIVERS_2024.get(driver_code, "Unknown")

    if lap_df.empty or telemetry_df.empty:
        return f"<h4>Stats for Nerds</h4><p>No detailed telemetry or lap data found for {driver_name}.</p>"

    total_time = lap_df['LapTime'].sum()
    total_distance = telemetry_df['Distance'].max()
    avg_gear_shifts = telemetry_df.groupby('LapNumber')['nGear'].apply(lambda g: g.diff().fillna(0).ne(0).sum()).mean()
    top_speed = telemetry_df['Speed'].max()
    drs_laps = telemetry_df[telemetry_df['DRS'] >= 10]['LapNumber'].nunique() # DRS value is usually > 8

    sectors = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    sector_names = ['Sector 1', 'Sector 2', 'Sector 3']
    best_sector = None
    best_sector_time = float('inf')
    best_sector_lap = None
    for sector in sectors:
        if lap_df[sector].notnull().any():
            min_val = lap_df[sector].min().total_seconds()
            lap_num = lap_df[lap_df[sector] == lap_df[sector].min()]['LapNumber'].iloc[0]
            if min_val < best_sector_time:
                best_sector_time = min_val
                best_sector_lap = lap_num
                best_sector = sector_names[sectors.index(sector)]

    # Wrap values in red span
    def red(val): return f"<span style='color:#e10600;'><strong>{val}</strong></span>"

    html = f"""
    <h4>Stats for Nerds</h4>
    <ul>
        <li><strong>Total Driving Time:</strong> {red(total_time)}</li>
        <li><strong>Total Distance:</strong> {red(f'{total_distance/1000:.2f} km')}</li>
        <li><strong>Average Gear Shifts per Lap:</strong> {red(f'{avg_gear_shifts:.0f}')}</li>
        <li><strong>Top Speed:</strong> {red(f'{top_speed} km/h')}</li>
        <li><strong>DRS Used on:</strong> {red(f'{drs_laps} laps')}</li>
        <li><strong>Most Impressive Sector:</strong> {red(f'{best_sector} on Lap {best_sector_lap} ({best_sector_time:.3f}s)')}</li>
    </ul>
    """
    return html