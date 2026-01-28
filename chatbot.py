from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

# Hugging Face open-source model

model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_new_tokens=512
)

chat_history=[
    SystemMessage(content="You are a helpful Ai assistant")
]

while True:
    user_input=input("You:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_history)


