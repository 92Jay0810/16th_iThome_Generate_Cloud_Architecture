from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# model

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")


# 2. 載入文件

Loader = PyPDFLoader("day12-data.pdf")
docs = Loader.load()

# 3. 切分成數個chunk，自定切割大小，自訂overlap
all_text = "".join(doc.page_content for doc in docs)
print("文件長度： ", end="")
print(len(all_text))
print("chunk長度： ", end="")
chunk_size = 400
print(chunk_size)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=100)
splits = text_splitter.split_documents(docs)


# 4. 轉換成embedding，儲存進Chroma向量資料庫.
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddingmodel)

retriever = vectorstore.as_retriever()


# 5. 做成chain.
template = """你是一個文件分析大師，可以根據使用者提供的資訊來對文件內進行分析


以下為文件的內容
              
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
    """請幫我總結文件的內容""")
print(result)
