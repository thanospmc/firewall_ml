# Instant firewall transmission loss calculation☕
## Machine Learning-Based Transmission Loss

Streamlit app for predicting the transmission loss of various firewall geometries with machine learning

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
