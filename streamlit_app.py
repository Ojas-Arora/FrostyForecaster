import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Penguin Predictor", page_icon=":penguin:", layout="wide")

# Custom CSS to change sidebar color
st.markdown(
    """
    <style>
    .sidebar {
        background-color: #00CED1; /* Dark Turquoise */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Page Title
st.header(" 💻 Discovering Penguin Species with Machine Learning")


# Data Section
with st.expander('🗂️ Data'):
    st.write('📄 **Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.dataframe(df)

    st.write('🔢 **X (Features)**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('🎯 **y (Target)**')
    y_raw = df.species
    st.dataframe(y_raw)

# Data Visualization Section
with st.expander('📊 Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features in sidebar
with st.sidebar:
    st.header('🛠️ Input features')
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
with st.expander('📥 Input features'):
    st.write('📝 **Input penguin data**')
    st.dataframe(input_df)
    st.write('🧮 **Combined penguins data**')
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

with st.expander('⚙️ Data preparation'):
    st.write('🔢 **Encoded X (input penguin)**')
    st.dataframe(input_row)
    st.write('🎯 **Encoded y (target)**')
    st.dataframe(y)

# Model Training and Prediction Section
clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']

# Prediction Display Section
st.subheader('🔮 Predicted Species Probability')
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
st.success(f'🎉 The predicted species is: {penguins_species[prediction][0]}')

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
