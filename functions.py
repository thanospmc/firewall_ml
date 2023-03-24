import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import dill
import matplotlib

def predict_TL(freq_type, models, scaler, X):
    # Scaling the input data
    print(f"Scaling {freq_type} model")
    X = scaler.transform(X)
    
    # Predicting the transmission loss 
    print(f"Predicting {freq_type} model")
    y_pred = models[freq_type].predict(X)
    return y_pred

#@st.cache(show_spinner=False, suppress_st_warning=True)
def load_models(model_path, model_no, freq_types):
    models = dict()
    for freq_type in freq_types:
        print(f"Loading {freq_type} model")
        mlmodel_path = model_path / f"lightgbm_{freq_type}_model_{model_no}.dill"
        with open(mlmodel_path, 'rb') as f:
            models[freq_type] = dill.load(f)
    return models

#@st.cache(show_spinner=False, suppress_st_warning=True)
def load_freqs(freq_path):
    freqs = {"narrow" : np.loadtxt(freq_path / 'freqs_narrow.csv'),
             "octave" : np.loadtxt(freq_path /'freqs_octave_band.csv'),
            "third_octave" : np.loadtxt(freq_path / 'freqs_third_octave_band.csv')
            } 
    return freqs

#@st.cache(show_spinner=False, suppress_st_warning=True)
def load_scaler(scaler_path):
    # load scaler
    with open(scaler_path, 'rb') as s:
        scaler = dill.load(s)
    print("Imported scaler")
    return scaler

def initialise_parameters():
    # Initialise parameters
    param_list = ["hl_thickness", 
                  "hl_density", 
                  "hl_modulus", 
                  "hl_poisson", 
                  "porous_thickness", 
                  "porous_density", 
                  "porous_modulus", 
                  "porous_porosity", 
                  "porous_tortuosity", 
                  "porous_flow_res", 
                  "porous_poisson"]
    param_dict =  {param: np.zeros((1,1)) for param in param_list}
    return  param_dict

def initialise_freq_types():
    freq_types = ["narrow", "octave", "third_octave"]
    return freq_types

def create_form(param_dict):
    with st.form('parameters', clear_on_submit=False):
        st.subheader('Specify the treatment properties on the patches')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Porous material properties')
            param_dict["porous_thickness"][0] = st.slider('Thickness [$m$]', 0.01, 0.08, 0.04, step=0.005, key=1)
            param_dict["porous_density"][0] = st.slider('Density [$kg/m^3$]', 395, 2160, 500, step=5, key=2)
            param_dict["porous_porosity"][0] = st.slider('Porosity [-]', 0.922, 0.98, 0.95, step=0.001, key=3)
            param_dict["porous_modulus"][0] = st.slider('Elastic Modulus [$Pa$]', 2600, 214000, 50000, step=50, key=4)
            param_dict["porous_tortuosity"][0] = st.slider('Tortuosity', 1.0, 1.88, 1.2, step=0.01, key=5)
            param_dict["porous_flow_res"][0] = st.slider('Flow resistivity [-]', 2000, 135000, 50000, step=10,key=6)
            param_dict["porous_poisson"][0] = st.slider('Poisson [-]', 0.02, 0.4, 0.24, step=0.005, key=7)
        with col2:
            st.subheader('Heavy layer material properties')
            param_dict["hl_thickness"][0] = st.slider('Thickness [$m$]', 0.001, 0.006, 0.004, step=0.0005, key=11)
            param_dict["hl_density"][0] = st.slider('Density [$kg/m^3$]', 10000, 20000, 15000, step=100, key=12)
            param_dict["hl_modulus"][0] = st.slider('Elastic Modulus [$Pa$]', 7000000, 130000000, 50000000, step=1000, key=13)
            param_dict["hl_poisson"][0] = st.slider('Poisson [-]', 0.3, 0.4, 0.35, step=0.005, key=14)
        
        submit_button=st.form_submit_button("Submit")
        if submit_button:
            st.session_state.count += 1

def prepare_predict(param_dict, freq_types, models, scaler):
    # Predicting the transmission loss
    X = np.hstack(list(param_dict.values()))

    y_pred = dict()
    for ft in freq_types:
        y_pred[ft] = np.transpose(predict_TL(ft, models, scaler, X)).reshape(-1)
    return y_pred

def create_plots(y_pred, freqs, freq_types):
    # Plotting the results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    for i, ft in enumerate(freq_types):
        ax[i].plot(freqs[ft], y_pred[ft], color='blue', linewidth=2, label='ML prediction, current')
        ax[i].set_title(f"{ft} band")
        ax[i].set_xlabel("Frequency [Hz]")
        ax[i].set_ylabel("Transmission loss [dB]")
        ax[i].set_xscale('log')
        ax[i].set_ylim(0, 120)
        ax[i].legend()
        ax[i].set_xticks([125, 500, 1000])
        ax[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
    ax[0].set_xlim(100, 1400)
    ax[1].set_xlim(100, 1250)
    ax[2].set_xlim(100, 1001)

    plt.tight_layout()

    #if st.session_state.count>0:
    #    ax[0].plot(freqs['narrow'], st.session_state.old_y_narrow, color='red', linestyle='dashed', linewidth=1, label='ML prediction, previous')
    #    ax[1].plot(freqs['octave'], st.session_state.old_y_octave, color='red', linestyle='dashed', linewidth=1, label='ML prediction, previous')
    #    ax[2].plot(freqs['third_octave'], st.session_state.old_y_third, color='red', linestyle='dashed', linewidth=1, label='ML prediction, previous')

    return fig

def initialise_connect_dict():
    connect_dict = {"porous_thickness": "Porous thickness", 
                    "porous_density": "Porous density", 
                    "porous_porosity": "Porosity",
                    "porous_modulus": "Porous Young's modulus",
                    "porous_tortuosity": "Tortuosity",
                    "porous_flow_res": "Flow resistivity",
                    "porous_poisson": "Porous Poisson's ratio",
                    "hl_thickness": "Heavy layer thickness",
                    "hl_density": "Heavy layer density",
                    "hl_modulus": "Heavy layer Young's modulus",
                    "hl_poisson": "Heavy layer Poisson's ratio"
                    }
    return connect_dict

def export_data(export_path, freqs, y_pred, param_dict, connect_dict, model_no):
    model_name = f"model_{model_no}.txt"
    model_path = export_path / model_name
    
    param_name = "parameters.txt"
    param_path = export_path / param_name

    narrow_name = "narrow_band.csv"
    narrow_path = export_path / narrow_name

    octave_name = "octave_band.csv"
    octave_path = export_path / octave_name

    third_octave_name = "third_octave_band.csv"
    third_octave_path = export_path / third_octave_name

    print("Exporting data...")
    with open(param_path, 'w') as f:
        f.write(f"Parameters for the porous and heavy layer materials for Model {model_no}\n")
        for key in param_dict.keys():
            f.write(f"{connect_dict[key]}, {param_dict[key][0][0]}\n")
    
    with open(narrow_path, 'w') as f:
        f.write("Frequency [Hz], Transmission loss [dB]\n")
        for i in range(len(freqs['narrow'])):
            f.write(f"{freqs['narrow'][i]}, {y_pred['narrow'][i]}\n")
    
    with open(octave_path, 'w') as f:
        f.write("Frequency [Hz], Transmission loss [dB]\n")
        for i in range(len(freqs['octave'])):
            f.write(f"{freqs['octave'][i]}, {y_pred['octave'][i]}\n")
    
    with open(third_octave_path, 'w') as f:
        f.write("Frequency [Hz], Transmission loss [dB]\n")
        for i in range(len(freqs['third_octave'])):
            f.write(f"{freqs['third_octave'][i]}, {y_pred['third_octave'][i]}\n")

    with open(model_path, 'w') as f:
        f.write(f"Data exported for Model {model_no}\n")

    print("Data exported!\n")
