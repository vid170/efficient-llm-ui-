import streamlit as st
import pandas as pd
import numpy as np
from utils import get_model_output
st.title('Efficient LLM')

prompt: str = st.chat_input("Enter a prompt here")

USER = "user"
ASSISTANT = "assistant"

if prompt:
    output, elapsed_time=get_model_output(prompt)
    st.chat_message(USER).write(prompt)
    st.chat_message(ASSISTANT).write(f"{output}\n Time Taken: {elapsed_time}")

