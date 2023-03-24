import streamlit as st
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import functions as f
import os
from datetime import datetime

from pathlib import Path

plt.rcParams.update({'font.size': 18})
model_no = 4

st.title(f'Instant firewall transmission loss calculation for Model {model_no}')

if 'count' not in st.session_state:
    st.session_state.count = 0
if 'old_y_narrow' not in st.session_state:
    st.session_state.old_y_narrow = 0 
if 'old_y_octave' not in st.session_state:
    st.session_state.old_y_octave = 0
if 'old_y_third' not in st.session_state:
    st.session_state.old_y_third = 0

main_path = Path('.')
image = Image.open(main_path / 'Data' / 'Images' / f'Model_{model_no}.png')
st.image(image, width=400, caption=f'Model {model_no} geometry')

param_dict = f.initialise_parameters()
freq_types = f.initialise_freq_types()
connect_dict =f.initialise_connect_dict()

# load files
freq_path = main_path / 'Data' / 'Freqs' 
freqs = f.load_freqs(freq_path=freq_path)
print("Imported frequencies")

model_path = main_path / 'Data' / f'Model_{model_no}'
models = f.load_models(model_path=model_path, model_no=model_no, freq_types=freq_types)

scaler_path = main_path / 'Data' / 'Features' / 'scaler.dill'
scaler = f.load_scaler(scaler_path=scaler_path)

# Create form
_ = f.create_form(param_dict)

# Prepare vector and predict
preds = f.prepare_predict(param_dict, freq_types, models, scaler)

# Plot data
fig = f.create_plots(preds, freqs, freq_types)
st.pyplot(fig)

now = datetime.now()
date_string = now.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-6]

# Export data
export_path = main_path / 'Data' / 'Exported_data' / f'Exported_data_{date_string}'

if st.button('Export data'):
    if export_path.exists():
        f.export_data(export_path, freqs, preds, param_dict, connect_dict, model_no=model_no)
    else:
        os.mkdir(export_path)
        f.export_data(export_path, freqs, preds, param_dict, connect_dict, model_no=model_no)
    st.write('Data exported to folder: ', export_path.resolve())