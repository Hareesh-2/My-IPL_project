import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Title
st.title("IPL Match Winner Predictor")

# Load your model (update the path if needed)
# Note: You'll need to save your model first using pickle.dump()
try:
    model = pickle.load(open("ipl_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'ipl_model.pkl' exists.")
    st.stop()

# Input fields
teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 
    'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Delhi Capitals', 
    'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 
    'Gujarat Titans'
]

cities = [
    'Mumbai', 'Kolkata', 'Delhi', 'Chennai', 'Hyderabad', 
    'Bangalore', 'Ahmedabad', 'Pune', 'Jaipur', 'Chandigarh'
]

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted([t for t in teams if t != batting_team]))

city = st.selectbox("Match Location", sorted(cities))

col3, col4 = st.columns(2)
with col3:
    target = st.number_input("Target Runs", min_value=0, step=1, value=180)
with col4:
    current_score = st.number_input("Current Score", min_value=0, step=1, value=90)

col5, col6 = st.columns(2)
with col5:
    overs_completed = st.slider("Overs Completed", 0.1, 20.0, value=10.0, step=0.1)
with col6:
    wickets_left = st.slider("Wickets Left", 0, 10, value=7)

# Calculate derived fields
balls_left = int(120 - (overs_completed * 6))
runs_left = max(target - current_score, 0)
current_run_rate = (current_score * 6) / (120 - balls_left) if (120 - balls_left) > 0 else 0
required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

if st.button("Predict Winner"):
    # Prepare input for model
    input_data = pd.DataFrame({
        'BattingTeam': [batting_team],
        'BowlingTeam': [bowling_team],
        'City': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'current_run_rate': [current_run_rate],
        'required_run_rate': [required_run_rate],
        'target': [target]
    })
    
    try:
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        col7, col8 = st.columns(2)
        
        with col7:
            st.metric(f"{batting_team} Win Probability", 
                      f"{probabilities[0][1]*100:.1f}%")
        with col8:
            st.metric(f"{bowling_team} Win Probability", 
                      f"{probabilities[0][0]*100:.1f}%")
        
        # Add some visual feedback
        if probabilities[0][1] > 0.5:
            st.success(f"{batting_team} are favorites to win!")
        else:
            st.success(f"{bowling_team} are favorites to win!")
            
        # Show a progress bar
        st.progress(probabilities[0][1])
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some additional information
st.markdown("---")
st.info("""
**Note:** This predictor uses a machine learning model trained on IPL match data from 2008-2022. 
The predictions are based on the current match situation and historical performance.
""")