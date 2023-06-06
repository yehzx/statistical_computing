import streamlit as sl
x = sl.slider("Select a value")
sl.write(x, "squared is", x * x)