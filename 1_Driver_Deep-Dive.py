import streamlit as st
from PIL import Image
import os
from config import CIRCUIT_IMAGE_MAP, DRIVERS_2024, TEAMS_2024, CIRCUITS_2024 , SESSION_TYPES, LOCATION_TO_EVENT_NAME_MAP
from narrative_generator import generate_narr, generate_nerd_stats
from telemetry import load_session_data, get_driver_laps, get_driver_telemetry
from plotting import generate_telemetry_plots, generate_strategy_plot
import base64
import pandas as pd
import fastf1 as ff
import joblib
import re
import xgboost as xgb
import google.generativeai as genai

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'ai_report' not in st.session_state:
    st.session_state.ai_report = None

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

def reset_analysis():
    st.session_state.analysis_run = False


def get_circuits_for_year(year):
    schedule = ff.get_event_schedule(year)
    circuits = schedule['Location'].unique().tolist()
    return circuits

@st.cache_data(show_spinner=False)
def get_drivers_for_year(year):
    session = ff.get_session(year, 1, 'R') 
    session.load(telemetry=False, laps=False, weather=False)
    results = session.results

    driver_dict = dict(zip(results['Abbreviation'], results['FullName']))
    return driver_dict

# ----------- APP SETTINGS -----------
st.set_page_config(page_title="F1 Dashboard", layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
            
        h1, h3, h4 {
            color: #FF1801;
            font-family: 'Helvetica Neue', sans-serif;
        }

        figure img {
            border-radius: 12px;
            box-shadow: 0 0 10px #FF1801;
        }
        
        [data-testid="stTab"] {
            font-size: 45px;
            font-weight: bold;
            padding: 15px 20px;
        } 
        .final p{
            color: #FF1801;    
        }
        .ai-report p {
            font-size: 1.1rem; /* This makes the paragraph text slightly larger */
            line-height: 1.6;  /* This improves readability by increasing line spacing */
        }
    </style>
""", unsafe_allow_html=True)


# ----------- TOP BAR: LOGO + TITLE -----------
col_logo, col_title, col_spacer = st.columns([1, 6, 1])
with col_logo:
    logo_path = "assets/F1_logo.png"
    if os.path.exists(logo_path):
        logo_base64 = base64.b64encode(open(logo_path, "rb").read()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{logo_base64}" width="100" style="box-shadow: none; border-radius: 0;">',
            unsafe_allow_html=True
        )

st.markdown("<hr style='border: 1px solid #FF1801;'>", unsafe_allow_html=True)


# ----------- SIDEBAR SELECTIONS -----------
st.sidebar.header("Select Race Details")
year_list = [2024, 2023, 2022, 2021, 2020, 2019, 2018]
year = st.sidebar.selectbox("Select Year", year_list, on_change=reset_analysis)
available_circuits = get_circuits_for_year(year)
circuit = st.sidebar.selectbox("Select Circuit", available_circuits, on_change=reset_analysis)

session_type = st.sidebar.selectbox("Select Session Type", SESSION_TYPES)

available_drivers = get_drivers_for_year(year)
driver_keys = list(available_drivers.keys())

selected_driver_code = st.sidebar.selectbox(
    "Select Driver",
    driver_keys,
    format_func=lambda x: available_drivers[x],
    on_change = reset_analysis
)

if st.sidebar.button("Show Analysis"):
    st.session_state.analysis_run = True


st.sidebar.header("Gemini API Key")
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = None

with st.sidebar.form("api_key_form"):
    api_key_input = st.text_input(
        "Enter your Google Gemini API Key", 
        type="password",
        help="Get your free API key from Google AI Studio."
    )
    submitted = st.form_submit_button("Validate and Save Key")

if submitted:
    if api_key_input == st.secrets.get("DEV_SECRET_CODE"):
        key_to_validate = st.secrets.get("GEMINI_API_KEY")
        st.sidebar.info("Developer key selected.")
    else:
        key_to_validate = api_key_input

    if not key_to_validate:
        st.sidebar.warning("Please enter a key.")
    else:
        try:
            genai.configure(api_key=key_to_validate)
            list(genai.list_models())
            
            st.session_state.gemini_api_key = key_to_validate
            st.sidebar.success("API Key is valid and has been saved!")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error("The provided API Key is invalid.")
            st.session_state.gemini_api_key = None

if st.session_state.analysis_run:
    all_laps, results, total_laps, weather_data = load_session_data(year, circuit, session_type)
    driver_laps = get_driver_laps(all_laps, selected_driver_code)
    driver_telemetry = get_driver_telemetry(all_laps, selected_driver_code)

    driver_img_path = f"assets/drivers/{selected_driver_code}.png"
    image_filename = CIRCUIT_IMAGE_MAP.get(circuit, "default.png")
    circuit_img_path = f"assets/circuits/{image_filename}"
    
    col1, col3 = st.columns([2, 2])
    with col1:
        st.image(driver_img_path, caption=DRIVERS_2024[selected_driver_code], use_container_width=False, width=280)
    with col3:
        st.image(circuit_img_path, caption=circuit, use_container_width=False, width=500)

    tab1, tab2 = st.tabs(["## Full Race Analysis", "## ML Pace Predictor"])

    with tab1:
        driver_result_row = results.loc[results['Abbreviation'] == selected_driver_code]
        if not driver_result_row.empty:
            driver_result = driver_result_row.iloc[0]
            position = driver_result['Position']
            status = driver_result['Status']
            position_text = f"P{int(position)}" if pd.notnull(position) else "N/A"
        else:
            position_text = "N/A"
            status = "Did not participate"

        if not driver_laps.empty:
            fastest_lap_time = driver_laps['LapTime'].min()
            fastest_lap_text = str(fastest_lap_time).split(':', 1)[1] if pd.notnull(fastest_lap_time) else "N/A"
        else:
            fastest_lap_text = "N/A"

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric(label="Finishing Position", value=position_text)
        with metric_col2:
            st.metric(label="Fastest Lap", value=fastest_lap_text)
        with metric_col3:
            st.metric(label="Status", value=status)
        with metric_col4:
            if not driver_laps.empty:
                team_name = driver_laps['Team'].iloc[0]
                team_logo_path = f"assets/teams/{team_name}.png"
                
                if os.path.exists(team_logo_path):
                    team_logo_base64 = base64.b64encode(open(team_logo_path, "rb").read()).decode()
                    st.markdown(
                        f'<img src="data:image/png;base64,{team_logo_base64}" width="100" style="box-shadow: none; border-radius: 0;">',
                        unsafe_allow_html=True
                    )
                else:
                    st.metric(team_name)
                
        st.markdown("---")
        narr, sector = generate_narr(driver_laps, selected_driver_code, circuit, session_type)
        stats = generate_nerd_stats(driver_laps, driver_telemetry, selected_driver_code, circuit, session_type)
        
        cola, colb, colc = st.columns([1.5,1.5,1.5])
        with cola:
            st.markdown(narr, unsafe_allow_html = True)
        with colb:
            st.markdown(sector, unsafe_allow_html = True)
        with colc:
            st.markdown(stats, unsafe_allow_html = True)

        st.markdown("---")
        st.markdown("### Tyre Strategy ")
        if not driver_laps.empty:
            fig_strategy = generate_strategy_plot(driver_laps)
            if fig_strategy:
                st.plotly_chart(fig_strategy, use_container_width=True)

        st.markdown("---")
        st.markdown("### Fastest Lap Telemetry Analysis")

        if not driver_laps.empty:
            fastest_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
            fastest_lap_telemetry = fastest_lap.get_telemetry().add_distance()
            fig_speed, fig_throttle, fig_brake, fig_rpm, fig_gear=  generate_telemetry_plots(fastest_lap_telemetry)
            if fig_speed:
                st.plotly_chart(fig_speed,use_container_width = True)

                col_t, col_b = st.columns(2)
                with col_t:
                    st.plotly_chart(fig_throttle,use_container_width = True)
                with col_b:
                    st.plotly_chart(fig_brake,use_container_width = True)
                
                col_r, col_g = st.columns(2)
                with col_r:
                    st.plotly_chart(fig_rpm,use_container_width = True)
                with col_g:
                    st.plotly_chart(fig_gear,use_container_width = True)
            else:
                st.warning(f"Could not generate telemetry plots for the fastest lap")
        else:
            st.warning(f"No lap data available to find fastest lap")
    
    with tab2:
        st.markdown("## Lap Time Predictor")
        col_a, col_b = st.columns(2)
        with col_a:
            lap_number = st.slider('Select Lap Number', min_value = 2, max_value = int(total_laps), value = 2)
            compound = st.selectbox('Select Tyre Compound', ['HYPERSOFT','SUPERSOFT','ULTRASOFT','SOFT','MEDIUM','INTERMEDIATE','HARD','WET'])
        with col_b:
            tyre_life = st.slider('Select Tyre Age (Laps)', min_value = 1, max_value = int(total_laps), value = 2)
            stint = st.number_input("Select Stint Number", min_value=1, max_value=5, value=2)
            
        weather = st.selectbox("Select Weather Scenario", ["Use Historical Average", "Simulate: Sunny & Hot", "Simulate: Cloudy & Cool", "Simulate: Light Rain"])
        if st.button("Predict Lap Time"):
            event_name = LOCATION_TO_EVENT_NAME_MAP.get(circuit,circuit)
            safe_event_name = re.sub(r'[\\/*?:"<>|]', "", event_name).replace(" ", "_")
            model_path = f"models/{safe_event_name}_model.joblib"
            
            try:
                model = joblib.load(model_path)
            except FileNotFoundError:
                st.error(f"Prediction model for {circuit} not found. Please ensure it has been trained.")
                st.stop()

            if isinstance(model, xgb.XGBRegressor):
                model_features = model.get_booster().feature_names
            else:
                model_features = model.feature_name_
            
            df = pd.DataFrame(0, index = [0],columns = model_features)
            df['LapNumber'] = lap_number
            df['Stint'] = stint
            df['TyreLife'] = tyre_life
            df['Year'] = year
            if weather == "Use Historical Average":
                df['AirTemp'] = weather_data['AirTemp'].mean()
                df['Humidity'] = weather_data['Humidity'].mean()
                df['Pressure'] = weather_data['Pressure'].mean()
                df['Rainfall'] = weather_data['Rainfall'].mode()[0]
                df['TrackTemp'] = weather_data['TrackTemp'].mean()
                df['WindDirection'] = weather_data['WindDirection'].mean()
                df['WindSpeed'] = weather_data['WindSpeed'].mean()
            elif weather == "Simulate: Sunny & Hot":
                df['AirTemp'] = weather_data['AirTemp'].mean() + 5
                df['Humidity'] = weather_data['Humidity'].mean() - 5
                df['Pressure'] = weather_data['Pressure'].mean() 
                df['Rainfall'] = 0
                df['TrackTemp'] = weather_data['TrackTemp'].mean() + 5
                df['WindDirection'] = weather_data['WindDirection'].mean()
                df['WindSpeed'] = weather_data['WindSpeed'].mean() - 2
            elif weather == "Simulate: Cloudy & Cool":
                df['AirTemp'] = weather_data['AirTemp'].mean() - 3
                df['Humidity'] = weather_data['Humidity'].mean() + 10
                df['Pressure'] = weather_data['Pressure'].mean() 
                df['Rainfall'] = 0
                df['TrackTemp'] = weather_data['TrackTemp'].mean() - 5
                df['WindDirection'] = weather_data['WindDirection'].mean()
                df['WindSpeed'] = weather_data['WindSpeed'].mean() * 1.5
            elif weather == "Simulate: Light Rain":
                df['AirTemp'] = weather_data['AirTemp'].mean() - 10
                df['Humidity'] = 98.0
                df['Pressure'] = weather_data['Pressure'].mean() 
                df['Rainfall'] = 1
                df['TrackTemp'] = weather_data['TrackTemp'].mean() - 15
                df['WindDirection'] = weather_data['WindDirection'].mean()
                df['WindSpeed'] = weather_data['WindSpeed'].mean() * 1.5
            
            driver = f"Driver_{selected_driver_code}"
            if driver in df.columns:
                df[driver] = 1

            comp = f"Compound_{compound}"
            if comp in df.columns:
                df[comp] = 1
            
            pred_time_s = model.predict(df)[0]
            mins = int(pred_time_s // 60)
            sec = pred_time_s % 60
            lap_time = f"{mins}:{sec:06.3f}"

            st.session_state.prediction_made = True
            st.session_state.prediction_result = {
                "time_str": lap_time,
                "driver_name": available_drivers[selected_driver_code],
                "circuit": circuit,
                "lap_number": lap_number,
                "compound": compound,
                "tyre_life": tyre_life,
                "weather": weather
            }
            st.session_state.ai_report = None

        if st.session_state.prediction_made:
            st.markdown('<h3 style="color: #FF1801;">ML Model Prediction</h3>', unsafe_allow_html=True)
            st.metric(label = "LAP TIME", value = st.session_state.prediction_result['time_str'])

            if st.button("Get the Pundit's Verdict"):
                if 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key:
                    with st.spinner("Waking Up the AI Commentator...."):
                        result = st.session_state.prediction_result
                        prompt = (f"You are a seasoned and witty F1 TV commentator. Your task is to analyze a lap time simulation and present your findings in an exciting and insightful way for a live broadcast. Use Markdown for formatting, including bold text and emojis, to make your analysis engaging.\n\n"
                                f"**The Simulation:**\n"
                                f"- Driver: {result['driver_name']}\n"
                                f"- Track: {result['circuit']}\n"
                                f"- Lap: {result['lap_number']}\n"
                                f"- Tyres: {result['compound']} ({result['tyre_life']} laps old)\n"
                                f"- Weather: '{result['weather']}'\n\n"
                                f"The simulation predicts a lap time of **{result['time_str']}**.\n\n"
                                f"**Your Commentary (in a few short, exciting paragraphs):**")
                        
                        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
                        response = gemini_model.generate_content(prompt)
                        st.session_state.ai_report = response.text
                else:
                    st.error("Please enter and validate a Gemini API key in the sidebar first.")

            if st.session_state.ai_report:
                st.markdown('<h3 style="color: #FF1801;">Google Gemini as an F1 TV Commentator</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-report">{st.session_state.ai_report}</div>', unsafe_allow_html=True)

                


                
            