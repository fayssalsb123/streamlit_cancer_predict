import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

def get_clean_data():

    df = pd.read_csv('data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # convert Malignant to 1 and benign to 0
    return df

def add_sidebar(data):

    st.sidebar.header("Cell Nuclei Measurements")
    input_dict = {}
    column_names = data.columns[1:]
    sliders_labels = [(f"{column} (mean)", column) for column in column_names]

    for label, key in sliders_labels:

        input_dict[key] = st.sidebar.slider(

            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())

        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():

        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    values1 = [input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
               input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
               input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
               input_data['fractal_dimension_mean']]

    values2 = [input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
               input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
               input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']]

    values3 = [input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
               input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
               input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
               input_data['fractal_dimension_worst']]

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(

        r=values1,
        theta=categories,
        fill='toself',
        name='mean value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=categories,
        fill='toself',
        name='standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values3,
        theta=categories,
        fill='toself',
        name='worst value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )
    return fig

# Move st.set_page_config to the beginning of the script
st.set_page_config(
    page_title="Breast cancer Predictor",
    page_icon="femal-doctor",
    layout="wide",
    initial_sidebar_state="expanded"
)



import pickle
import numpy as np

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)  # Reshape to a 2D array
    scaled_input_array = scaler.transform(input_array)
    
    # Assuming the model.predict() method is used for predictions
    prediction = model.predict(scaled_input_array)
    st.subheader("Cell cluster prediction ")
    st.write("the cell cluster is:")
    if prediction[0]==0:
        st.write("Benign")
    else:
        st.write("malignant")
    
    st.write("Probability of being benign:",model.predict_proba(scaled_input_array)[0][0])
    st.write("Probability of being malignant:",model.predict_proba(scaled_input_array)[0][1])
    st.write("this app can assist medical professional in amking a diagnosis, but should not be used as a substitute for professional diagnosis")





def main():
    

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose from your tissue sample.")

    data = get_clean_data()
    input_data = add_sidebar(data)

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()


