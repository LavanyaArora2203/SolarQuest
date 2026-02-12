import streamlit as st
from chatbot import get_chat_response   # Import your chatbot function

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot Demo")
st.caption("Streamlit Frontend + GPT-5 Nano Backend")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Display previous chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input box
user_input = st.chat_input("Type your message...")

if user_input:

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from backend chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_chat_response(st.session_state.messages)
            st.markdown(reply)

    # Save assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
