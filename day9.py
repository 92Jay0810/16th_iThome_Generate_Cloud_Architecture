from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")


# 2. 抓取網頁的特定class的內容，並載入
loader = WebBaseLoader(
    web_paths=("https://house.udn.com/house/story/123590/7815040",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-content__title",
                    "article-content__editor")
        )
    ),
)
docs = loader.load()

# 3. 切分成數個chunk
all_text = "".join(doc.page_content for doc in docs)
print("文件長度： ", end="")
print(len(all_text))
print("chunk長度： ", end="")
chunk_size = 300
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
回答結束時，總是要說「感謝提問！」

{context}

使用者提供的額外資訊：
{user_input}

"""
custom_rag_prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# retriever  為何要配合format_docs，retriever 從向量資料庫中，搜尋到最相似的片段後，
# 將這些片段當成參數輸入給format_docs，格式化成字串
rag_chain = (
    {"context": retriever | format_docs, "user_input": lambda x: x}
    | custom_rag_prompt
    | model
    | StrOutputParser()
)


# 6. 呼叫
result = rag_chain.invoke("新增哪6棟摩天大樓?")
print(result)
