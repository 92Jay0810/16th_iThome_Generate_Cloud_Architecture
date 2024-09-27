import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

'''
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 需要輸出解析器
parser = StrOutputParser()
# 使用 | 組合提示模型和解析器
chain = prompt_template | model | parser
text = chain.invoke({"language": "Chinese", "text": "hi"})
'''
text = model.invoke("hi!")

print(text)
