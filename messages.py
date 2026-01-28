from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_new_tokens=512
)

messages={
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content='Tell me about langchain')
}

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

