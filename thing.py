import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Try to import seaborn (optional dependency)
try:
    import seaborn as sns

    seaborn_available = True
except ImportError:
    seaborn_available = False

# Page configuration
st.set_page_config(
    page_title="IPL Match Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("IPL Match Prediction App")
st.markdown("""
This app predicts the winner of an IPL cricket match based on the current match situation.
""")

# Define file paths directly (update these paths according to your files)
MATCHES_FILE_PATH = r"matches(1).csv"
DELIVERIES_FILE_PATH = r"deliveries(1).csv"


# Function to load data
@st.cache_data
def load_data(deliveries_path, matches_path):
    """Load data from specified file paths."""
    try:
        if os.path.exists(deliveries_path) and os.path.exists(matches_path):
            delivery = pd.read_csv(deliveries_path)
            match = pd.read_csv(matches_path)

            # Rename 'id' column to 'match_id' in the matches dataset for consistency
            if 'id' in match.columns and 'match_id' not in match.columns:
                match = match.rename(columns={'id': 'match_id'})

            return delivery, match, True
        else:
            missing_files = []
            if not os.path.exists(deliveries_path):
                missing_files.append(f"Deliveries file: {deliveries_path}")
            if not os.path.exists(matches_path):
                missing_files.append(f"Matches file: {matches_path}")

            st.error(f"The following files were not found: {', '.join(missing_files)}")
            return None, None, False
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None, False


# Load data from the specified file paths
with st.spinner("Loading data files..."):
    delivery, match, data_loaded = load_data(DELIVERIES_FILE_PATH, MATCHES_FILE_PATH)

# If data is not loaded, stop execution
if not data_loaded:
    st.error("Could not load data files. Please check the file paths.")
    st.code(f"Deliveries file path: {DELIVERIES_FILE_PATH}")
    st.code(f"Matches file path: {MATCHES_FILE_PATH}")
    st.stop()
else:
    st.success("Data files loaded successfully!")

# Main application code continues if data is loaded
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

# Inspect the columns to identify any issues
st.header("Deliveries Dataset Columns")
st.write(delivery.columns)

st.header("Matches Dataset Columns")
st.write(match.columns)

# Data preprocessing function
@st.cache_data
def preprocess_data(delivery_df, match_df):
    """Preprocess the data for model training and prediction."""
    try:
        # Check if required columns exist
        required_delivery_cols = ['match_id', 'inning', 'batting_team', 'bowling_team',
                                  'over', 'ball', 'total_runs', 'player_dismissed']
        required_match_cols = ['match_id', 'city', 'winner']

        # Verify delivery columns
        missing_delivery_cols = [col for col in required_delivery_cols if col not in delivery_df.columns]
        if missing_delivery_cols:
            st.error(f"Missing columns in deliveries dataset: {missing_delivery_cols}")
            return None, None, None

        # Verify match columns
        missing_match_cols = [col for col in required_match_cols if col not in match_df.columns]
        if missing_match_cols:
            st.error(f"Missing columns in matches dataset: {missing_match_cols}")
            return None, None, None

        # Total score per match (first innings only)
        total_score_df = delivery_df.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
        total_score_df = total_score_df[total_score_df['inning'] == 1]

        # Merge with match data
        match_with_score = match_df.merge(total_score_df[['match_id', 'total_runs']],
                                          on='match_id', how='inner')

        # Create delivery data with match info
        delivery_with_match = delivery_df.merge(match_with_score, on='match_id', how='inner')

        # Handle potential column name issues
        if 'total_runs_x' in delivery_with_match.columns:
            total_runs_col = 'total_runs_x'
        else:
            total_runs_col = 'total_runs'

        if 'total_runs_y' in delivery_with_match.columns:
            target_col = 'total_runs_y'
        else:
            target_col = 'target'

        # Filter to only include second innings data
        second_innings = delivery_with_match[delivery_with_match['inning'] == 2].copy()

        # Standardize column names
        col_mapping = {
            'total_runs': 'delivery_runs',
            total_runs_col: 'delivery_runs',
            target_col: 'target'
        }
        second_innings.rename(columns=col_mapping, inplace=True)

        # Check if renaming was successful
        if 'delivery_runs' not in second_innings.columns:
            st.error("Column 'delivery_runs' not found after renaming.")
            return None, None, None

        # Calculate current score, runs left, and balls left
        second_innings['current_score'] = second_innings.groupby(['match_id'])['delivery_runs'].cumsum()
        second_innings['runs_left'] = second_innings['target'] - second_innings['current_score']
        second_innings['balls_left'] = 120 - (second_innings['over'] * 6 + second_innings['ball'])

        # Calculate wickets
        second_innings['is_wicket'] = second_innings['player_dismissed'].notna().astype(int)
        wickets_fallen = second_innings.groupby('match_id')['is_wicket'].cumsum()
        second_innings['wickets'] = 10 - wickets_fallen

        # Calculate current run rate and required run rate
        balls_completed = 120 - second_innings['balls_left']
        # Avoid division by zero
        balls_completed_overs = np.maximum(balls_completed / 6, 0.1)  # At least 0.1 overs to avoid div by zero
        second_innings['crr'] = second_innings['current_score'] / balls_completed_overs

        # Handle potential division by zero in required run rate
        second_innings['rrr'] = np.where(
            second_innings['balls_left'] > 0,
            (second_innings['runs_left'] * 6) / second_innings['balls_left'],
            999  # Large value for impossible situation
        )

        # Clean data - remove rows with infinite or very large required run rate
        second_innings = second_innings[second_innings['rrr'] < 100]

        # Create target variable - whether batting team won
        second_innings['result'] = second_innings['batting_team'] == second_innings['winner']

        # Create feature dataset
        X = second_innings[['batting_team', 'bowling_team', 'city',
                            'runs_left', 'balls_left', 'wickets',
                            'target', 'crr', 'rrr']]
        y = second_innings['result']

        return X, y, second_innings
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


# Create and train model
@st.cache_resource
def create_model(X, y):
    """Create and train a logistic regression model."""
    try:
        # Define the preprocessing steps with proper handling of sparse parameter
        # (compatibility with different sklearn versions)
        sparse_param = {}
        if hasattr(OneHotEncoder(), 'sparse_output'):
            sparse_param = {'sparse_output': False}
        else:
            sparse_param = {'sparse': False}

        # Create preprocessing transformer
        trf = ColumnTransformer([
            ('trf', OneHotEncoder(drop='first', **sparse_param),
             ['batting_team', 'bowling_team', 'city'])
        ], remainder='passthrough')

        # Create pipeline
        pipe = Pipeline(steps=[
            ('step1', trf),
            ('step2', LogisticRegression(solver='liblinear', max_iter=1000))
        ])

        # Train the model
        pipe.fit(X, y)
        return pipe
    except Exception as e:
        st.error(f"Error in model creation: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# Display a spinner while processing data
with st.spinner("Processing data and training model..."):
    # Process data and train model
    try:
        X, y, processed_data = preprocess_data(delivery, match)

        if X is None or len(X) == 0:
            st.error("Unable to process data. Please check your data files.")
            st.stop()

        model = create_model(X, y)

        if model is None:
            st.error("Unable to create model. Please check your data files.")
            st.stop()

        st.success("Model trained successfully!")

        # Get unique teams and cities for prediction inputs
        # Convert non-string city values to string and fill NaNs
        processed_data['city'] = processed_data['city'].astype(str).fillna('Unknown')
        teams = sorted(processed_data['batting_team'].unique())
        cities = sorted(processed_data['city'].unique())

        # Prediction section
        st.header("Match Prediction")
        st.markdown("Enter the current match situation to predict the winner:")

        col1, col2 = st.columns(2)

        with col1:
            batting_team = st.selectbox("Select Batting Team", teams)
            bowling_team = st.selectbox("Select Bowling Team",
                                        [team for team in teams if team != batting_team])
            city = st.selectbox("Select City", cities)
            target = st.number_input("Target Score", min_value=1, value=180)

        with col2:
            current_score = st.number_input("Current Score",
                                            min_value=0,
                                            max_value=target - 1,
                                            value=min(100, target - 1))

            overs_completed = st.number_input("Overs Completed",
                                              min_value=0.0,
                                              max_value=19.5,
                                              value=10.0,
                                              step=0.1,
                                              format="%.1f")

            wickets = st.number_input("Wickets Fallen",
                                      min_value=0,
                                      max_value=9,
                                      value=2)

        # Calculate derived features
        balls_completed = int(overs_completed) * 6 + int((overs_completed - int(overs_completed)) * 10)
        balls_left = 120 - balls_completed
        runs_left = target - current_score

        if balls_completed > 0:
            crr = current_score / (balls_completed / 6)
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
            'target': [target],
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
                # Make prediction
                prediction = model.predict_proba(input_df)
                win_prob = prediction[0][1] * 100

                # Display match situation
                st.subheader("Match Situation")
                situation_col1, situation_col2 = st.columns(2)

                with situation_col1:
                    st.write(f"Target: {target}")
                    st.write(f"Current Score: {current_score}/{wickets}")
                    st.write(f"Runs Left: {runs_left}")

                with situation_col2:
                    st.write(f"Balls Left: {balls_left}")
                    st.write(f"Current Run Rate: {crr:.2f}")
                    st.write(f"Required Run Rate: {rrr:.2f}")

                # Display prediction
                st.subheader("Prediction")
                pred_col1, pred_col2 = st.columns(2)

                with pred_col1:
                    st.metric(f"{batting_team} Win Probability", f"{win_prob:.1f}%")

                with pred_col2:
                    st.metric(f"{bowling_team} Win Probability", f"{100 - win_prob:.1f}%")

                # Visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                teams_to_plot = [batting_team, bowling_team]
                probs_to_plot = [win_prob, 100 - win_prob]
                colors = ['#1f77b4', '#ff7f0e']  # Blue and orange

                ax.bar(teams_to_plot, probs_to_plot, color=colors)
                ax.set_ylabel('Win Probability (%)')
                ax.set_title('Match Prediction')
                ax.set_ylim(0, 100)

                for i, v in enumerate(probs_to_plot):
                    ax.text(i, v + 2, f"{v:.1f}%", ha='center')

                st.pyplot(fig)

        # Data exploration section
        st.header("Data Exploration")
        if st.checkbox("Show Data Exploration"):
            tab1, tab2, tab3, tab4 = st.tabs(["Team Performance", "City Analysis", "Score Distribution", "Head-to-Head Analysis"])

            with tab1:
                st.subheader("Team Performance Analysis")
                if 'winner' in match.columns:
                    team_wins = match['winner'].value_counts()
                    fig, ax = plt.subplots(figsize=(12, 6))

                    bars = team_wins.plot(kind='bar', ax=ax, color='#1f77b4')
                    ax.set_title('Number of Wins by Team')
                    ax.set_ylabel('Number of Wins')
                    ax.set_xlabel('Team')

                    # Add value labels on top of bars
                    for bar in bars.patches:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                f'{int(height)}', ha='center')

                    st.pyplot(fig)
                else:
                    st.warning("Winner column not found in matches dataset")

            with tab2:
                st.subheader("City Analysis")
                if 'city' in match.columns:
                    city_matches = match['city'].value_counts().head(10)
                    fig, ax = plt.subplots(figsize=(12, 6))

                    bars = city_matches.plot(kind='bar', ax=ax, color='#ff7f0e')
                    ax.set_title('Number of Matches by City (Top 10)')
                    ax.set_ylabel('Number of Matches')
                    ax.set_xlabel('City')

                    # Add value labels on top of bars
                    for bar in bars.patches:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                f'{int(height)}', ha='center')

                    st.pyplot(fig)
                else:
                    st.warning("City column not found in matches dataset")

            with tab3:
                st.subheader("Score Distribution")
                total_scores = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
                first_innings = total_scores[total_scores['inning'] == 1]

                fig, ax = plt.subplots(figsize=(12, 6))

                if seaborn_available:
                    sns.histplot(first_innings['total_runs'], bins=20, ax=ax, kde=True)
                else:
                    ax.hist(first_innings['total_runs'], bins=20, color='#2ca02c', alpha=0.7)

                ax.set_title('First Innings Score Distribution')
                ax.set_xlabel('Total Runs')
                ax.set_ylabel('Frequency')

                st.pyplot(fig)

                # Summary statistics
                st.write("Summary Statistics for First Innings Scores:")
                st.write(first_innings['total_runs'].describe())

            with tab4:
                # Head-to-Head Analysis
                st.subheader("Head-to-Head Analysis")
                if 'winner' in match.columns:
                    head_to_head = match[((match['team1'] == batting_team) & (match['team2'] == bowling_team)) |
                                         ((match['team1'] == bowling_team) & (match['team2'] == batting_team))]
                    head_to_head_wins = head_to_head['winner'].value_counts()

                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = head_to_head_wins.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                    ax.set_title(f'Head-to-Head Wins: {batting_team} vs {bowling_team}')
                    ax.set_ylabel('Number of Wins')
                    ax.set_xlabel('Team')

                    # Add value labels on top of bars
                    for bar in bars.patches:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                f'{int(height)}', ha='center')

                    st.pyplot(fig)

                    st.write(f"Total Matches: {len(head_to_head)}")

                    # Check if 'season' column exists before attempting to display it
                    if 'season' in head_to_head.columns:
                        st.write(head_to_head[['season', 'city', 'winner']].sort_values('season'))
                    else:
                        st.write(head_to_head[['city', 'winner']].sort_values('city'))
                else:
                    st.warning("Winner column not found in matches dataset")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        st.info("Please check your data files and try again.")

# Footer
st.markdown("---")
st.markdown("IPL Match Prediction App - Created with Streamlit")