from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="anpigon/qwen2.5-7b-instruct-kowiki:q6_k",
    temperature=0.7,
)

print(llm.invoke("안녕하세요"))