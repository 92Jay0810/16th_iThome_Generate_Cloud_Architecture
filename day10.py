from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")


# 2. 載入CSV檔案
loader = CSVLoader(file_path="day10.csv", encoding="utf-8")
docs = loader.load()

# 3. 切分成數個chunk
all_text = "".join(doc.page_content for doc in docs)
print("文件長度： ", end="")
print(len(all_text))
print("chunk長度： ", end="")
chunk_size = 400
print(chunk_size)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=50)
splits = text_splitter.split_documents(docs)


# 4. 轉換成embedding，儲存進Chroma向量資料庫.
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddingmodel)
retriever = vectorstore.as_retriever()


# 5. 做成chain.
template = """使用以下的上下文來回答最後的問題。
如果你不知道答案，就直接說你不知道，不要編造答案。
最多用三句話，並保持回答簡明扼要。

{context}

使用者提供的額外資訊：
{user_input}
"""

custom_rag_prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "user_input": lambda x: x}
    | custom_rag_prompt
    | model
    | StrOutputParser()
)

# 6. 呼叫
result = rag_chain.invoke(
    "低收入戶人數上從過去的年份到最新的幾個年份有什麼樣的變化嗎?")
print(result)
