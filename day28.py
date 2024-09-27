from langchain_openai import ChatOpenAI
import os
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model_name="gpt-4o")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
# draw
try:
    image_data = graph.get_graph().draw_mermaid_png()  # 二進制資料
    with open('day28_LangGraph_workflow.png', 'wb') as f:
        f.write(image_data)
except Exception as e:
    print(f"意外: {e}")
    pass
