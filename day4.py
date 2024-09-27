import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

'''system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 需要輸出解析器
parser = StrOutputParser()
# 使用 | 組合提示模型和解析器
chain = prompt_template | model | parser
text = chain.invoke({"language": "Chinese", "text": "hi"})
print(text)'''

# 定義系統訊息，設置對話角色和情境
system_template = "創建一個對話場景包含 {character1} 和 {character2}."


prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{dialogue_start}")]
)

# 需要輸出解析器
parser = StrOutputParser()

# 使用 | 組合提示模型和解析器
chain = prompt_template | model | parser

# 輸入對話角色和開始的對話
text = chain.invoke({"character1": "偵探", "character2": "嫌疑犯",
                    "dialogue_start": "你為什麼會在犯罪現場？"})
print(text)
