import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

hospital_profiles = pd.read_csv('GroupedHospitalScores2.csv')
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
cheatsheet.head()

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
    for category in categories:
        mean_value = hospital_profiles[category].mean()

        for nurse in preference_df.index:
            weight = preference_df.loc[nurse, category]

            # Ensure weight is a scalar value
            if isinstance(weight, (int, float)):  # Check if weight is a number
                if weight == 1:
                    influence = 0.5  # Least influence
                elif weight == 2:
                    influence = 0.75
                elif weight == 3:
                    influence = 1.2
                elif weight == 4:
                    influence = 1.3   # More influence
                else:
                    influence = 1.0  # Default to neutral if weight is out of range

            # Calculate the weighted value
                weighted_value = mean_value * influence
                preference_df.at[nurse, category] = weighted_value
            else:
                preference_df.at[nurse, category] = mean_value

# Apply the weight assignment function
cheatsheet.apply(lambda row: assign_weights(row, preference_df), axis=1)
set_preferences(preference_df, hospital_profiles)

preference_df.columns = preference_df.columns.to_series().replace(category_mapping)
preference_df = preference_df.drop(columns=['DEI', 'Management and Leadership'])
preference_df.head()

# Normalize the data
scaler = StandardScaler()
hospital_ratings_scaled = scaler.fit_transform(hospital_profiles)
nurse_ratings_scaled = scaler.fit_transform(preference_df)

scaled_df = pd.DataFrame(hospital_ratings_scaled, columns=hospital_profiles.columns, index=hospital_profiles.index)
scaled_nurse_df = pd.DataFrame(nurse_ratings_scaled, columns=preference_df.columns, index=preference_df.index)

scaled_df.rename(columns={
    'Pay (1-5)': 'Pay',
    'Orientation & onboarding (1-5)': 'Orientation & Onboarding',
    'Mgmt & leadership (1-5)': 'Management and Leadership',
    'Safety & patient ratios (1-5)': 'Safety & Patient Ratios',
    'DEI/LGBTQ+ friendliness (1-5)': 'DEI',
    'Patient Acuity (1-5)': 'Patient Acuity',
    'Housing options (1-5)': 'Housing Options',
    'Facility Location (1-5)': 'Facility Location'}, inplace=True)

# Create the model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(scaled_df)

scaled_nurse_df = scaled_nurse_df[scaled_df.columns]
distances, indices = knn.kneighbors(scaled_nurse_df)

# Store closest hospitals as a list
preference_df['Closest_Hospitals'] = [list(row) for row in indices]

# Create a mapping from indices to hospital names
index_to_hospital = hospital_profiles.index.to_series().reset_index(drop=True).to_dict()

def replace_indices_with_labels(indices):
    return [index_to_hospital.get(i, "Unknown Hospital") for i in indices]

# Apply the function to the Closest_Hospitals column
preference_df['Closest_Hospitals'] = preference_df['Closest_Hospitals'].apply(replace_indices_with_labels)


categories = ['Pay', 'Orientation & Onboarding', 'Mgmt & Leadership', 
              'Safety & Patient Ratios', 'DEI/LGBTQ+ Friendliness', 
              'Patient Acuity', 'Housing Options', 'Facility Location']

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

# Convert preferences to a numpy array and scale it
user_preferences_scaled = scaler.transform(np.array(list(preferences.values())).reshape(1, -1))

    # Use the KNN model to find the nearest hospitals
distances, indices = knn.kneighbors(user_preferences_scaled)

# Map indices to hospital names
closest_hospitals = [index_to_hospital.get(i, "Unknown Hospital") for i in indices[0]]

# Display the closest hospitals
st.write("Your top 5 hospital matches based on your preferences are:")
for hospital in closest_hospitals:
    st.write(hospital)

st.write("Details for the recommended hospitals:")

for hospital in closest_hospitals:
    st.write(f"**{hospital}**")  # Hospital name
    
    # Get the actual ratings for each hospital from the original hospital profiles
    hospital_ratings = hospital_profiles.loc[hospital, categories]

    # Display the ratings for each category
    for category in categories:
        st.write(f"{category}: {round(hospital_ratings[category], 2)}")
    
    st.write("---")  # Add a separator between hospitals