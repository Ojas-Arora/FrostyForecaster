import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Penguin Predictor", page_icon=":penguin:", layout="wide")

# Page Title
st.markdown('<h2 style="color: #00CED1;">💻 Discovering Penguin Species with Machine Learning</h2>', unsafe_allow_html=True)

# Data Section
with st.expander("🗂️ Data"):
    st.markdown('<h3 style="color: #00CED1;">Data</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00CED1;">📄 <b>Raw data</b></p>', unsafe_allow_html=True)
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.dataframe(df)

    st.markdown('<p style="color: #00CED1;">🔢 <b>X (Features)</b></p>', unsafe_allow_html=True)
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.markdown('<p style="color: #00CED1;">🎯 <b>y (Target)</b></p>', unsafe_allow_html=True)
    y_raw = df.species
    st.dataframe(y_raw)

# Data Visualization Section
with st.expander("📊 Data visualization"):
    st.markdown('<h3 style="color: #00CED1;">Data visualization</h3>', unsafe_allow_html=True)
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features in sidebar
with st.sidebar:
    st.markdown('<h3 style="color: #00CED1;">🛠️ Input features</h3>', unsafe_allow_html=True)
    island = st.selectbox('🏝️ Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('📏 Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('📏 Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('📏 Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('⚖️ Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('🚻 Gender', ('Male', 'Female'))

    # Create a DataFrame for the input features
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# Input Features Display
with st.expander("📥 Input features"):
    st.markdown('<h3 style="color: #00CED1;">Input features</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00CED1;">📝 <b>Input penguin data</b></p>', unsafe_allow_html=True)
    st.dataframe(input_df)
    st.markdown('<p style="color: #00CED1;">🧮 <b>Combined penguins data</b></p>', unsafe_allow_html=True)
    st.dataframe(input_penguins)

# Data Preparation Section
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y (Target)
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander("⚙️ Data preparation"):
    st.markdown('<h3 style="color: #00CED1;">Data preparation</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00CED1;">🔢 <b>Encoded X (input penguin)</b></p>', unsafe_allow_html=True)
    st.dataframe(input_row)
    st.markdown('<p style="color: #00CED1;">🎯 <b>Encoded y (target)</b></p>', unsafe_allow_html=True)
    st.dataframe(y)

# Model Training and Prediction Section
clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']

# Prediction Display Section
st.markdown('<h2 style="color: #00CED1;">🔮 Predicted Species Probability</h2>', unsafe_allow_html=True)
st.dataframe(df_prediction_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn(
                     'Adelie',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Chinstrap': st.column_config.ProgressColumn(
                     'Chinstrap',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Gentoo': st.column_config.ProgressColumn(
                     'Gentoo',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
             }, hide_index=True)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_species = penguins_species[prediction][0]

# Custom HTML to style the success message in dark turquoise
st.markdown(f'<h3 style="color: #00CED1;">🎉 The predicted species is: {predicted_species}</h3>', unsafe_allow_html=True)


# Custom JavaScript for alerts
custom_js = """
<script>
    // Alert on page load
    window.onload = function() {
        alert('🌟 Welcome to the Penguin Predictor! Explore the fascinating world of penguin species with our machine learning tool!');
    };
</script>
"""

# Render the custom JavaScript
st.components.v1.html(custom_js)
