import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the water quality data
url_water = 'https://raw.githubusercontent.com/michalis0/MGT-502-Data-Science-and-Machine-Learning/main/data/waterQuality1.csv'
df_water = pd.read_csv(url_water)

# Features (X) and target variable (y)
Xw = df_water.drop('is_safe', axis=1)
yw = df_water['is_safe']

# Calculate the range for each column
column_ranges = {column: df_water[column].max() - df_water[column].min() for column in Xw.columns}

# Adjust the range for sliders to be 150% bigger than the dataset range
slider_ranges = {column: (df_water[column].min() - 0.5 * column_ranges[column], df_water[column].max() + 0.5 * column_ranges[column]) for column in Xw.columns}

# Split data set into a train and a test data sets
Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=39, shuffle=True)

# Define the scaler
scaler = StandardScaler()

# Fit the scaler
scaler.fit(Xw_train)

# 1. Set up our model
modelw = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)

# 2. Fit our model
modelw.fit(Xw_train, yw_train)

# Streamlit app
st.title('Water Safety Prediction')

# Input form for user to enter chemical elements and impurities
user_input = {}
for column in Xw.columns:
    if column != 'is_safe':
        user_input[column] = st.slider(f'{column} Value', min_value=0, max_value=slider_ranges[column][1] + 0.5 * column_ranges[column], step=0.01, value=df_water[column].mean())

# Create a user input DataFrame
user_input_df = pd.DataFrame(user_input, index=[0])

# Scale the user input using the same scaler
user_input_scaled = scaler.transform(user_input_df)

# Make prediction
prediction = modelw.predict(user_input_scaled)

# Display prediction result
if prediction[0] == 0:
    st.write('Prediction: This water sample is unsafe.')
else:
    st.write('Prediction: This water sample is safe.')
