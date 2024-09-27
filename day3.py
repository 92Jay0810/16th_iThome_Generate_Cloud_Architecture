import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")


text = "hi!"

parser = StrOutputParser()

chain = model | parser

result = chain.invoke(text)

print(result)
