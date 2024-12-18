import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def dcm_pricing_spread():
    st.header("DCM Pricing Spread")
    # Вставьте код из приложения DCM Pricing Spread

def dcm_matchbox():
    st.header("DCM Matchbox")
    # Вставьте код из приложения DCM Matchbox

def dcm_terminal():
    st.header("DCM MatVal")
    # Вставьте код из приложения DCM Terminal

def main():
    st.title("MatVal")
    menu = st.sidebar.selectbox("Выберите приложение", ("DCM Pricing Spread", "DCM Matchbox", "DCM MatVal"))

    if menu == "DCM Pricing Spread":
        dcm_pricing_spread()
    elif menu == "DCM Matchbox":
        dcm_matchbox()
    elif menu == "DCM MatVal":
        dcm_terminal()

if __name__ == "__main__":
    main()
