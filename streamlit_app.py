import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Custom HTML for style
html_style = """
<style>
    h1 {
        color: #ff6347;
        font-family: 'Arial', sans-serif;
    }
    .data-section {
        padding: 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .expander-title {
        font-weight: bold;
        color: #4682b4;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .input-box {
        margin-bottom: 15px;
    }
</style>
"""

# Embedding custom JavaScript (for small alert)
custom_js = """
<script>
    alert('Welcome to the Machine Learning App!');
</script>
"""

# Render the custom HTML and CSS
st.markdown(html_style, unsafe_allow_html=True)

# Display the JavaScript
st.components.v1.html(custom_js)

st.title('🤖 Machine Learning App')

# Data Section
with st.expander('<span class="expander-title">🗂️ Data</span>', expanded=True):
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

# Sidebar with Input Features
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛠️ Input features</div>', unsafe_allow_html=True)
    
    island = st.selectbox('🏝️ Island', ('Biscoe', 'Dream', 'Torgersen'), key='island')
    bill_length_mm = st.slider('📏 Bill length (mm)', 32.1, 59.6, 43.9, key='bill_length')
    bill_depth_mm = st.slider('📏 Bill depth (mm)', 13.1, 21.5, 17.2, key='bill_depth')
    flipper_length_mm = st.slider('📏 Flipper length (mm)', 172.0, 231.0, 201.0, key='flipper_length')
    body_mass_g = st.slider('⚖️ Body mass (g)', 2700.0, 6300.0, 4207.0, key='body_mass')
    gender = st.selectbox('🚻 Gender', ('Male', 'Female'), key='gender')

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
with st.expander('<span class="expander-title">📥 Input features</span>', expanded=True):
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
st.dataframe(df_prediction_proba)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f'🎉 The predicted species is: {penguins_species[prediction][0]}')

# Render any additional JS content (if needed)
st.components.v1.html("<script>alert('Prediction completed!');</script>")
