import streamlit as st
from utils import get_model_output
st.title('Efficient LLM')

prompt: str = st.chat_input("Enter a prompt here")

USER = "user"
ASSISTANT = "assistant"

if prompt:
    output= "One day, she was playing with her toys. While playing, she saw a big pin on the yard. It wanted to safe again looking at it. Lily wanted to help her mother, but so she talked about the pink jump and better.\"Wow, Lily!\" said the pink is a pink. \"What rainbows are!""\"The next was safe,\" said the dog's happy steps. \"Thank you, Lily,\" said Lily.\"I found a big ice-cream,\" the pink grew drain.\"Yes, let's play!\" said Lily. But the warn's white was rough. \"My tooughs are looking in the sky!\" Lily smiled."
    st.chat_message(USER).write(prompt)
    st.chat_message(ASSISTANT).write(f"{output}")

