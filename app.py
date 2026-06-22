import os
import sys

if __name__ == "__main__":
    # Check if we are running inside streamlit
    try:
        import streamlit as st
        is_streamlit = st.runtime.exists()
    except (ImportError, AttributeError):
        is_streamlit = False

    if is_streamlit:
        # Prevent loop: load the actual app code directly
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))
        import app
    else:
        # Start the streamlit app located in app/app.py
        os.system("streamlit run app/app.py")
