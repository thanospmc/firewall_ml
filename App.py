import streamlit as st
from PIL import Image
from pathlib import Path

main_path = Path(".")

st.set_page_config(
    page_title="Instant firewall transmission loss calculation",
    page_icon="☕",
)

st.title('Instant firewall transmission loss calculation☕')
st.header('Machine Learning Based Transmission Loss')

st.sidebar.success("Select a model to begin.")

st.image(Image.open(main_path / 'Data' / 'Images' / 'all_models.png'), use_column_width=True)

st.markdown(
    """
    Welcome to this app that calculates the transmission loss of a firewall using machine learning.
    
    **To get started, 👈select a model from the sidebar**. You can see the available models above.
     To learn more about the app, read on👇.

    ### What is this app?
    This app is a machine learning based transmission loss calculator for firewalls. 
    It uses LightGBM models trained on a dataset of 150 samples of 7 different firewall geometries,
    to predict the transmission loss of a firewall. The model is trained on 3 different frequency bands: 
    narrow band, octave band and third octave band. The model is trained on a number of parameters for porous
    and heavy layer materials. 
    
    The model is trained on the following parameters:
    - Porous layer thickness
    - Porous layer porosity
    - Porous layer density
    - Porous layer tortuosity
    - Porous layer flow resistivity
    - Porous layer Young's modulus
    - Porous layer Poisson's ratio
    - Heavy layer thickness
    - Heavy layer density
    - Heavy layer Young's modulus
    - Heavy layer Poisson's ratio

    You can play with the parameters via the sliders and once you have selected the values you want, click the "Submit" 
    button to get the transmission loss plotted in different frequency bands. Finally, you can download the results to do
    further post processing or evaluate the performance of such a firewall with Actran.

    Let's get started!
"""
)

c1, c2 = st.columns(2)
with c1:
    st.info('**Creator: [@ThanosPoulos](https://linkedin.com/in/thanospoulos)**', icon="💡")
with c2:
    st.info('**GitHub: [@thanospmc](https://github.com/thanospmc)**', icon="💻")