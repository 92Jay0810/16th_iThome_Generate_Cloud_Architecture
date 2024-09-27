# langChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

embeddingmodel = OpenAIEmbeddings(model="text-embedding-ada-002")

# 2. DocumentLoader


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


#  3. LoaderFile And Split And Save
print("chunk長度： ", end="")
chunk_size = 1200
print(chunk_size)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=100)

loader = CustomDocumentLoader("Dac_Guides/Diagram.txt")
docs = list(loader.lazy_load())
all_text = "".join(doc.page_content for doc in docs)
print("文件1長度： ", end="")
print(len(all_text))
splits = text_splitter.split_documents(docs)

loader2 = CustomDocumentLoader("Dac_Guides/Node.txt")
docs2 = list(loader2.lazy_load())
all_text2 = "".join(doc2.page_content for doc2 in docs2)
print("文件2長度： ", end="")
print(len(all_text2))
splits2 = text_splitter.split_documents(docs2)

loader3 = CustomDocumentLoader("Dac_Guides/Cluster.txt")
docs3 = list(loader3.lazy_load())
all_text3 = "".join(doc3.page_content for doc3 in docs3)
print("文件3長度： ", end="")
print(len(all_text3))
splits3 = text_splitter.split_documents(docs3)

loader4 = CustomDocumentLoader("Dac_Guides/Edges.txt")
docs4 = list(loader4.lazy_load())
all_text4 = "".join(doc4.page_content for doc4 in docs4)
print("文件4長度： ", end="")
print(len(all_text4))
splits4 = text_splitter.split_documents(docs4)

loader5 = CustomDocumentLoader("Dac_Nodes/GCPNode.txt")
docs5 = list(loader5.lazy_load())
all_text5 = "".join(doc5.page_content for doc5 in docs5)
print("文件5長度： ", end="")
print(len(all_text5))
splits5 = text_splitter.split_documents(docs5)


vectorstore = Chroma.from_documents(
    documents=splits+splits2+splits3+splits4+splits5, embedding=embeddingmodel)
retriever = vectorstore.as_retriever()

# 4. Create chain
template = """你將成為一個雲端架構圖設計師，使用 Python 的 diagrams 函式庫，生成GCP的雲端架構圖。
請依照使用者的需求，設計一個安全、高效的雲端架構，並確保使用相關的雲端服務，如 VPC、負載均衡、Kubernetes、儲存服務、資料庫等。
你可以自由發揮，但每個服務的選擇都需要合理解釋，並考慮擴展性、安全性和效能優化。

以下是一個基本範例，你可以基於此來自訂使用者的需求：

```python

from diagrams import Cluster, Diagram
from diagrams.gcp.network import VPC, LoadBalancing, Armor, CDN, DNS,VPN
from diagrams.gcp.compute import KubernetesEngine
from diagrams.gcp.database import Firestore, SQL, Memorystore
from diagrams.gcp.storage import Filestore, Storage
from diagrams.gcp.operations import Monitoring
from diagrams.gcp.security import Iam, KeyManagementService, SecurityCommandCenter    
from diagrams.onprem.client import Users

# 使用者可調整這個架構設計
with Diagram("Secure Website System Architecture", show=False):
    # 定義使用者、網路、計算、資料庫、儲存、運營和安全等模組
    # 添加每個服務的設計思路
```


請你自動根據使用者給定的架構需求，進行設計，並解釋每個服務的選擇理由。


有以下簡單的規則請注意
- 將每個雲端服務分別定義，並使用合適的 `Cluster` 來組織它們。
- 每個 `Cluster` 的 `graph_attr` 應該設置顏色。
- 請用 ``` ``` 來包裹程式碼，並確保裡面只有程式碼。
- security這類別是自由的!，不需要連結到其他服務

以下是參考資料：        
{context}


最後是重要的使用者要求:
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


rchain_response = rag_chain.invoke("我想要一個保證安全性的網站系統雲端架構圖")
print(rchain_response)
