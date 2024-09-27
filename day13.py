# streamlit
import streamlit as st
# langChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 啟動命令 streamlit run filename.py

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")


# define process file
def process_document(file_content, user_input, file_name):
    # 暫存上檔案，並且得到上傳檔案的副檔名
    temp_file_path = f"temp_uploaded_file.{file_name.split('.')[-1]}"
    # 寫成暫存檔案
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    # 1. 載入CSV檔案
    loader = CSVLoader(file_path=temp_file_path, encoding="utf-8")
    docs = list(loader.lazy_load())

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
    os.remove(temp_file_path)

    return result


# frontend
def run_streamlit():
    st.set_page_config(
        page_title="文件分析與總結器", page_icon="📄")
    st.title("文件分析與總結器")
    st.write("")
    st.write("### Step 1: 上傳文件")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv"], label_visibility="visible")
    st.write("")
    st.write("### Step 2: 輸入額外資訊，讓人工智慧處理")
    user_input = st.text_input(
        "Enter Additional Information", "")
    st.write("")
    st.write("### Step 3: 提交")
    submit_button = st.button("Submit", use_container_width=True)
    if submit_button:
        if uploaded_file is not None:
            st.write(f"**檔案名稱:** {uploaded_file.name}")
            st.write(f"**額外資訊:** {user_input}")

            # Process the document
            file_content = uploaded_file.getvalue()
            summary = process_document(
                file_content, user_input, uploaded_file.name)

            st.write("")
            st.write("### Summary:")
            st.write(summary)
        else:
            st.error("請先上傳檔案")


if __name__ == "__main__":
    run_streamlit()
