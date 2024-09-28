import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

# Custom JavaScript for alerts
custom_js = """
<script>
    // Alert on page load
    window.onload = function() {
        alert('Welcome to the Machine Learning App!');
    };

    // Function to alert the prediction result
    function showPredictionResult(species) {
        alert('The predicted species is: ' + species);
    }
</script>
"""

# Render the custom JavaScript
st.components.v1.html(custom_js)

# Data Section
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.dataframe(df)

    st.write('**X (Features)**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**y (Target)**')
    y_raw = df.species
    st.dataframe(y_raw)

# Data Visualization Section
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar with Input Features
with st.sidebar:
    st.header('Input features')
    
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('Male', 'Female'))
    
    # Create a DataFrame for the input features
    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(data, index=[0])

# Model training and inference
# Data preparation
encode = ['island', 'sex']
df_penguins = pd.get_dummies(pd.concat([input_df, X_raw], axis=0), prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.apply(lambda val: target_mapper[val])

# Train the model
clf = RandomForestClassifier()
clf.fit(X, y)

# Make predictions
prediction = clf.predict(input_row)
species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_species = species[prediction][0]

# Display prediction result and call the JavaScript function
st.subheader('Predicted Species')
st.write(predicted_species)

# Show prediction result alert
show_prediction_js = f"""
<script>
    // Call the function to alert the predicted species
    showPredictionResult('{predicted_species}');
</script>
"""
st.components.v1.html(show_prediction_js)
