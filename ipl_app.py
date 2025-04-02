import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False
    st.warning("Seaborn is not installed. Some visualizations may not be available.")


st.set_page_config(page_title="IPL Match Predictor", layout="wide")

st.title("IPL Match Prediction App")

# File upload section
st.sidebar.header("Upload Data Files")
uploaded_deliveries = st.sidebar.file_uploader("Upload deliveries.csv", type="csv")
uploaded_matches = st.sidebar.file_uploader("Upload matches.csv", type="csv")

# Load data
@st.cache_data
def load_data():
    try:
        delivery = pd.read_csv('deliveries.csv')
        match = pd.read_csv('matches.csv')
        return delivery, match
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None

if uploaded_deliveries and uploaded_matches:
    delivery = pd.read_csv(uploaded_deliveries)
    match = pd.read_csv(uploaded_matches)
    st.success("Files successfully uploaded!")
else:
    # Try to load from default path
    delivery, match = load_data()
    if delivery is None or match is None:
        st.warning("Please upload the required CSV files to proceed")
        st.stop()

# Display basic information about the datasets
st.header("Dataset Information")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Matches Dataset")
    st.write(f"Shape: {match.shape}")
    st.dataframe(match.head())

with col2:
    st.subheader("Deliveries Dataset")
    st.write(f"Shape: {delivery.shape}")
    st.dataframe(delivery.head())

# Data preprocessing function
@st.cache_data
def preprocess_data(delivery, match):
    # Total score per match
    total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]
    
    # Merge with match data
    match_df = match.merge(total_score_df[['match_id', 'total_runs']], on='match_id')
    
    # Create delivery data with match info
    delivery_df = match_df.merge(delivery, on='match_id')
    
    # Filter second innings
    delivery_df = delivery_df[delivery_df['inning'] == 2]
    
    # Calculate current score, runs left, and balls left
    delivery_df['current_score'] = delivery_df.groupby(['match_id'])['total_runs_y'].cumsum()
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])
    
    # Calculate wickets
    wickets = delivery_df.groupby(['match_id', 'over', 'ball'])['player_dismissed'].apply(lambda x: str(x) != 'nan').groupby(['match_id']).cumsum().values
    delivery_df['wickets'] = 10 - wickets
    
    # Calculate current run rate and required run rate
    delivery_df['crr'] = delivery_df['current_score'] / ((120 - delivery_df['balls_left'])/6)
    delivery_df['rrr'] = delivery_df['runs_left'] * 6 / delivery_df['balls_left']
    
    # Clean data
    delivery_df = delivery_df[np.isfinite(delivery_df['rrr'])]
    
    # Create final dataset
    X = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']]
    y = delivery_df['batting_team'] == delivery_df['winner']
    
    return X, y, delivery_df

# Create and train model
@st.cache_resource
def create_model(X, y):
    # Define the preprocessing steps
    # Use sparse_output instead of sparse for newer sklearn versions
    try:
        trf = ColumnTransformer([
            ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
        ], remainder='passthrough')
    except TypeError:
        # For older versions of sklearn
        trf = ColumnTransformer([
            ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
        ], remainder='passthrough')
    
    # Create pipeline
    pipe = Pipeline(steps=[
        ('step1', trf),
        ('step2', LogisticRegression(solver='liblinear'))
    ])
    
    # Train the model
    pipe.fit(X, y)
    return pipe

# Process data and train model
try:
    X, y, delivery_df = preprocess_data(delivery, match)
    model = create_model(X, y)

    # Prediction section
    st.header("Match Prediction")

    # Get unique teams and cities
    teams = sorted(delivery_df['batting_team'].unique())
    cities = sorted(delivery_df['city'].unique())

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Select Batting Team", teams)
        bowling_team = st.selectbox("Select Bowling Team", [team for team in teams if team != batting_team])
        city = st.selectbox("Select City", cities)
        target = st.number_input("Target Score", min_value=1, value=180)

    with col2:
        current_score = st.number_input("Current Score", min_value=0, value=100)
        overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=19.5, value=10.0, step=0.1)
        wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2)

    # Calculate derived features
    balls_completed = int(overs_completed) * 6 + int((overs_completed - int(overs_completed)) * 10)
    balls_left = 120 - balls_completed
    runs_left = target - current_score

    if balls_completed > 0:
        crr = current_score / (balls_completed/6)
    else:
        crr = 0

    if balls_left > 0:
        rrr = runs_left * 6 / balls_left
    else:
        rrr = 0

    # Create input dataframe
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [10 - wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Make prediction when button is clicked
    if st.button("Predict Winner"):
        if balls_left <= 0:
            st.error("No balls left. Match is over!")
        elif runs_left <= 0:
            st.success(f"{batting_team} has already won!")
        else:
            prediction = model.predict_proba(input_df)
            win_prob = prediction[0][1] * 100
            
            st.subheader("Match Situation")
            st.write(f"Target: {target}")
            st.write(f"Current Score: {current_score}/{wickets}")
            st.write(f"Runs Left: {runs_left}")
            st.write(f"Balls Left: {balls_left}")
            st.write(f"Current Run Rate: {crr:.2f}")
            st.write(f"Required Run Rate: {rrr:.2f}")
            
            st.subheader("Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{batting_team} Win Probability", f"{win_prob:.1f}%")
            with col2:
                st.metric(f"{bowling_team} Win Probability", f"{100-win_prob:.1f}%")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([batting_team, bowling_team], [win_prob, 100-win_prob], color=['blue', 'red'])
            ax.set_ylabel('Win Probability (%)')
            ax.set_title('Match Prediction')
            st.pyplot(fig)

    # Data exploration section
    st.header("Data Exploration")
    if st.checkbox("Show Data Exploration"):
        tab1, tab2, tab3 = st.tabs(["Team Performance", "City Analysis", "Score Distribution"])
        
        with tab1:
            st.subheader("Team Performance Analysis")
            team_wins = match['winner'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            team_wins.plot(kind='bar', ax=ax)
            ax.set_title('Number of Wins by Team')
            ax.set_ylabel('Number of Wins')
            st.pyplot(fig)
        
        with tab2:
            st.subheader("City Analysis")
            city_matches = match['city'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            city_matches.plot(kind='bar', ax=ax)
            ax.set_title('Number of Matches by City (Top 10)')
            ax.set_ylabel('Number of Matches')
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Score Distribution")
            total_scores = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
            first_innings = total_scores[total_scores['inning'] == 1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if seaborn_available:
                sns.histplot(first_innings['total_runs'], bins=20, ax=ax)
            else:
                ax.hist(first_innings['total_runs'], bins=20)
            ax.set_title('First Innings Score Distribution')
            ax.set_xlabel('Total Runs')
            st.pyplot(fig)
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check your data files and try again.")

# Footer
st.markdown("---")
st.markdown("IPL Match Prediction App - Created with Streamlit")
