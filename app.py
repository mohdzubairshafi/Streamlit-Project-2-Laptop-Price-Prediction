import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("Laptop Predictor")

# brand
company = st.selectbox("Brand", df["Company"].unique())

# cpu
cpu = st.selectbox("CPU", df["Processor Type"].unique())
os = st.selectbox("OS", df["Operating System"].unique())

# Ram
ram = st.selectbox("RAM(in GB)", df['RAM Size'].unique())
ram_type = st.selectbox('Type',df['RAM Type'].unique())




disk_size = st.selectbox("Disk size", df['Disc Size'].unique())

disk_type = st.selectbox("Disk Type",df['Disc Type'].unique())


if st.button("Predict Price"):
    # query
    query = np.array([company, cpu, os, ram, ram_type, disk_size, disk_type])

    query = query.reshape(1, 7)
    st.title(
        "The predicted price of this configuration is:₹ "
        + str(int(np.exp(pipe.predict(query)[0])))
    )
