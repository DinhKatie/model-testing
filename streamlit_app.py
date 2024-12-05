import streamlit as st
import pandas as pd
import numpy as np

hospital_profiles = pd.read_csv('GroupedHospitalScores3.csv')
cheatsheet = pd.read_csv('Ask Margot Sample Data_ Reviews Spreadsheet - Nurse Cheat Sheet.csv')

hospital_profiles = hospital_profiles.set_index('Hospital')

#Fix column headers being at row 0.
cheatsheet.columns = cheatsheet.iloc[0]
cheatsheet = cheatsheet.drop(cheatsheet.index[0]).reset_index(drop=True)
cheatsheet = cheatsheet.drop(columns=['Age', 'Political Spectrum', 'Travel Nurse Experience'])

category_mapping = {
    'Mgmt & leadership': 'Mgmt & Leadership',
    'Management & Leadership': 'Mgmt & Leadership',
    'Safety & patient ratios': 'Safety & Patient Ratios',
    'Safety & patient ratios ': 'Safety & Patient Ratios',
    'Safety & Patient Ratios': 'Safety & Patient Ratios',
    'Patient acuity': 'Patient Acuity',
    'DEI, Facility Location': 'DEI/LGBTQ+ Friendliness',
    'Orientation & onboarding': 'Orientation & Onboarding',
    'Management & leadership': 'Mgmt & Leadership',
    'Orientaton & Onboarding': 'Orientation & Onboarding',
    'Safety & patient ratios , Facility Location': 'Safety & Patient Ratios',
    'Patient acuity, Mgmt & leadership': 'Patient Acuity',
    'Patient acuity, Facility Location': 'Patient Acuity',
    'Orientation & onboarding, Mgmt & leadership': 'Orientation & Onboarding',
    'Safety': "Safety & Patient Ratios"
}

# Standardize the category columns in 'cheatsheet'
cheatsheet = cheatsheet.replace(category_mapping)

categories = ['Pay','DEI/LGBTQ+ Friendliness', 'Patient Acuity',
              'Orientation & Onboarding', 'Mgmt & Leadership', 'Housing Options',
              'Facility Location', 'Safety & Patient Ratios']

# Create an empty DataFrame with all categories as columns and fill with 0's
preference_df = pd.DataFrame(0, index=cheatsheet['Nurse'], columns=categories)

# Function to assign weights to the preferences
def assign_weights(row, preference_df):
    nurse_id = row['Nurse']

    # Assign weights based on importance
    preference_df.at[nurse_id, row['Most Important - Top 1']] = 4
    preference_df.at[nurse_id, row['Most Important - Top 2']] = 3
    preference_df.at[nurse_id, row['Least Important - 2nd bottom of list']] = 2
    preference_df.at[nurse_id, row['Least Important - bottom of list']] = 1

def set_preferences(preference_df, hospital_profiles):
    nurse_scores = {}

    for nurse in preference_df.index:
        scores = []  # Store scores for each hospital for this nurse

        for _, hospital in hospital_profiles.iterrows():
            total_score = 0

            for category in categories:
                weight = preference_df.loc[nurse, category]

                # Assign influence based on weight
                if isinstance(weight, (int, float)):
                    if weight == 1:
                        influence = 0.2  # Least influence
                    elif weight == 2:
                        influence = 0.3
                    elif weight == 3:
                        influence = 1.7
                    elif weight == 4:
                        influence = 1.9  # Most influence
                    else:
                        influence = 1.0  # Neutral weight
                else:
                    influence = 1.0

                # Add the weighted value to the total score
                total_score += hospital[category] * influence

            # Append the total score for this hospital
            scores.append(total_score)

        nurse_scores[nurse] = scores

    # Convert scores dictionary to a new DataFrame
    scores_df = pd.DataFrame(nurse_scores, index=hospital_profiles.index)

    return scores_df


# Apply the weight assignment function
cheatsheet.apply(lambda row: assign_weights(row, preference_df), axis=1)
scores_df = set_preferences(preference_df, hospital_profiles)

preference_df.columns = preference_df.columns.to_series().replace(category_mapping)
preference_df = preference_df.drop(columns=['DEI', 'Management and Leadership'])

st.title("Hospital Recommendation System")

# Dropdowns for user preference selection
top_1 = st.selectbox('Most Important - Top 1', categories)
top_2 = st.selectbox('Most Important - Top 2', [cat for cat in categories if cat != top_1])
bottom_1 = st.selectbox('Least Important - Bottom 1', [cat for cat in categories if cat not in (top_1, top_2)])
bottom_2 = st.selectbox('Least Important - Bottom 2', [cat for cat in categories if cat not in (top_1, top_2, bottom_1)])

# Display the user's selection
st.write(f"Your preferences are: Top 1: {top_1}, Top 2: {top_2}, Bottom 1: {bottom_1}, Bottom 2: {bottom_2}")

# Initialize all categories with a neutral weight (e.g., 2.5 for a 0-5 scale).
preferences = {category: 2.5 for category in categories}

# Assign weights based on the user's input
preferences[top_1] = 4
preferences[top_2] = 3
preferences[bottom_1] = 1
preferences[bottom_2] = 2

# Update preference_df with user's selections
for category, weight in preferences.items():
    preference_df[category] = weight

# Calculate the scores based on updated preferences
scores_df = set_preferences(preference_df, hospital_profiles)

# Rank the hospitals based on the scores (highest score first)
ranked_scores = scores_df.mean(axis=1).sort_values(ascending=False)

# Display the ranked hospitals
st.write("Top hospitals based on your preferences:")
st.write(ranked_scores.head(10))
