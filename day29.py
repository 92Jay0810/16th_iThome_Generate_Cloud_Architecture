# langChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import AmazonKendraRetriever

# streamlit
import streamlit as st
import re

# langGraph
from typing_extensions import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# run command streamlit run day29.py

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model_name="gpt-4o")


# 2. retriever
retriever = AmazonKendraRetriever(
    index_id="",
    min_score_confidence=0.5)


# 3. relevance Prompt templates
relevance_template = """
你是一個判斷使用者輸入是否與雲端架構相關的專家。
請判斷以下使用者輸入是否與雲端架構設計相關：

使用者輸入：{user_input}

如果相關，請回答 "relevance_Yes"。如果不相關，請回答 "relevance_NO"，並簡短解釋原因。
"""

# 4. architecture Prompt template
architecture_template = """你將成為一個雲端架構圖設計師，使用 Python 的 diagrams 函式庫，生成AWS的雲端架構圖。
請依照使用者的需求，設計一個安全、高效的雲端架構，並確保使用相關的雲端服務，如 VPC、負載均衡、Kubernetes、儲存服務、資料庫等。
你可以自由發揮，但每個服務的選擇都需要合理解釋，並考慮擴展性、安全性和效能優化。

以下是一個基本範例，你可以基於此來自訂使用者的需求：

```python

from diagrams import Cluster, Diagram
from diagrams.aws.network import VPC, ElasticLoadBalancing, CloudFront, Route53, ClientVpn
from diagrams.aws.compute import EKS
from diagrams.aws.database import Aurora, RDS, Elasticache
from diagrams.aws.storage import FSx, S3
from diagrams.aws.management import Cloudwatch
from diagrams.aws.security import IAM, KMS, SecurityHub, Shield
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
# 5. 創造不同的chain
relevance_prompt = PromptTemplate.from_template(relevance_template)
architecture_prompt = PromptTemplate.from_template(architecture_template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


relevance_chain = relevance_prompt | model | StrOutputParser()
architecture_chain = (
    {"context": retriever | format_docs, "user_input": lambda x: x}
    | architecture_prompt
    | model
    | StrOutputParser()
)

# 6.1 Define state


class State(TypedDict):
    messages: list[HumanMessage | AIMessage]
    next: str
# 6.2 Define nodes


def relevance_check(state: State) -> State:
    user_input = state["messages"][-1].content
    relevance_result = relevance_chain.invoke({"user_input": user_input})
    if "relevance_Yes" in relevance_result:
        state["next"] = "generate_architecture"
        state["messages"].append(AIMessage(content="您的輸入與雲端架構相關。我將為您生成架構圖。"))
    else:
        state["next"] = "irrelevant_response"
        state["messages"].append(AIMessage(content=relevance_result))
    return state


def generate_architecture(state: State) -> Dict:
    user_input = state["messages"][-2].content
    architecture_result = architecture_chain.invoke(user_input)
    state["messages"].append(AIMessage(content=architecture_result))
    return state


def irrelevant_response(state: State) -> Dict:
    state["messages"].append(
        AIMessage(content="您的輸入似乎與雲端架構設計無關。請提供與雲端架構相關的需求，我才能為您生成架構圖。"))
    return state


# 6.3 Define workflow
workflow = StateGraph(State)
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("generate_architecture", generate_architecture)
workflow.add_node("irrelevant_response", irrelevant_response)
workflow.set_entry_point("relevance_check")
workflow.add_conditional_edges(
    "relevance_check",
    lambda x: x["next"],
    {
        "generate_architecture": "generate_architecture",
        "irrelevant_response": "irrelevant_response"
    }
)
workflow.add_edge("generate_architecture", END)
workflow.add_edge("irrelevant_response", END)
chain = workflow.compile()


# 7. Streamlit 介面設計
st.title("MutiAgent雲端架構圖生成器 :cloud:")
st.write(
    "輸入您的雲端架構需求，系統將自動生成相應的程式碼，讓您的雲端設計更快捷！",
    unsafe_allow_html=True,
)
st.markdown("---")
st.subheader("請輸入架構需求 :pencil:")
user_input = st.text_input("", placeholder="例如：我想要一個保證安全性的網站系統雲端架構圖",)
st.markdown("---")

# 按鈕觸發生成
if st.button("生成雲端架構圖", help="點擊按鈕生成架構圖"):
    if user_input:
        with st.spinner("生成中，請稍候..."):
            try:
                # Invoke the LangGraph chain
                result = chain.invoke(
                    {"messages": [HumanMessage(content=user_input)]})
                final_message = result["messages"][-1].content

                if "您的輸入似乎與雲端架構設計無關" in final_message:
                    st.warning(final_message)
                else:
                    # 提取 ``` 內的程式碼
                    code_pattern = r"```python(.*?)```"
                    match = re.search(code_pattern, final_message, re.DOTALL)

                    # 保留程式碼以外的說明文字
                    explanation = re.sub(
                        code_pattern, "", final_message, flags=re.DOTALL).strip()
                    if explanation:
                        st.markdown("### 架構設計說明")
                        st.write(explanation)

                    if match:
                        # 取得程式碼
                        diagram_code = match.group(1).strip()

                        # 建立執行環境並執行程式碼
                        exec(diagram_code)

                        # 顯示生成的架構圖
                        st.image(f"secure_website_system_architecture.png",
                                 caption="生成的架構圖", use_column_width=True)
                        st.success("架構圖生成成功！")
                    else:
                        st.error("未能擷取到程式碼，請檢查輸入內容或格式。")
            except Exception as e:
                st.error(f"生成程式碼時發生錯誤：{e}")
    else:
        st.warning("請輸入需求後再進行生成")
# draw
try:
    image_data = chain.get_graph().draw_mermaid_png()  # 二進制資料
    with open('day29_MutiAgent_ArchitectureGenerate_workflow.png', 'wb') as f:
        f.write(image_data)
except Exception as e:
    print(f"意外: {e}")
    pass
