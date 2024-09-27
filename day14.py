# streamlit
import streamlit as st
# langChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_core.document_loaders import BaseLoader
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 啟動命令 streamlit run filename.py

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")


# 2. 製作loader
class CustomDocumentLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        if self.file_path.endswith('.csv'):
            loader = CSVLoader(self.file_path)
            for doc in loader.lazy_load():
                yield doc
        elif self.file_path.endswith('.pdf'):
            loader = PyPDFLoader(self.file_path)
            for doc in loader.lazy_load():
                yield doc
        else:
            with open(self.file_path, encoding="utf-8") as f:
                line_number = 0
                for line in f:
                    yield Document(
                        page_content=line,
                        metadata={"line_number": line_number,
                                  "source": self.file_path},
                    )
                    line_number += 1


def process_documents(files, user_input):
    all_docs = []

    for uploaded_file in files:
        # 暫存上檔案
        temp_file_path = f'''temp_uploaded_file_{
            uploaded_file.name.split('.')[-1]}'''
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getvalue())

        # 1. 載入檔案
        loader = CustomDocumentLoader(temp_file_path)
        docs = list(loader.lazy_load())
        all_docs.extend(docs)

        # 刪除暫存檔案
        os.remove(temp_file_path)

    # 2. 切分成數個chunk，自定切割大小，自訂overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. 轉換成embedding，儲存進Chroma向量資料庫
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddingmodel)
    retriever = vectorstore.as_retriever()

    # 4. 做成chain.
    template = """
    你是一個人工智慧輔助系統，可以根據文件的內容進行簡單的總結，並且根據使用者需求來進行分析
    以下為文件的內容:
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

    # 5. 呼叫
    result = rag_chain.invoke(user_input)
    return result


def run_streamlit():
    st.set_page_config(
        page_title="文件分析與總結器", page_icon="📄")
    st.title("文件分析與總結器")
    st.write("")
    st.write("### Step 1: 上傳文件")
    uploaded_files = st.file_uploader(
        "Choose a file", type=["txt", "py", "csv", "pdf"], label_visibility="visible", accept_multiple_files=True)
    st.write("")
    st.write("### Step 2: 輸入額外資訊，讓人工智慧處理")
    user_input = st.text_input(
        "Enter Additional Information", "")
    st.write("")
    st.write("### Step 3: 提交")
    submit_button = st.button("Submit", use_container_width=True)
    if submit_button:
        if uploaded_files is not None:
            file_names = ", ".join([file.name for file in uploaded_files])
            st.write(f"**檔案名稱:** {file_names}")
            st.write(f"**額外資訊:** {user_input}")

            # Process the document
            summary = process_documents(uploaded_files, user_input)

            st.write("")
            st.write("### Summary:")
            st.write(summary)
        else:
            st.error("請先上傳檔案")


if __name__ == "__main__":
    run_streamlit()
