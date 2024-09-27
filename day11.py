from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_core.document_loaders import BaseLoader
from typing import Iterator
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# model

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")


# 2. 製作loader並載入
class CustomDocumentLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number,
                              "source": self.file_path},
                )
                line_number += 1


loader = CustomDocumentLoader("day11-data.py")
docs = list(loader.lazy_load())

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
文件有可能是程式碼 或者是單純的txt檔案

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
    """文件內是程式碼檔案，請讀取完後製作更多的範例來讓我學習""")
print(result)
