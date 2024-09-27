import os
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")


# 2. pydantic class
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="笑話的設定")
    punchline: str = Field(description="笑話有趣的重點")
    rating: int = Field(
        description="這個笑話的評分，從1到10")


# structured_llm = model.with_structured_output(Joke)

# structure_joke = structured_llm.invoke("告訴我一個關於貓的笑話")

structured_llm = model.with_structured_output(method="json_mode")

structure_joke = structured_llm.invoke(
    "告訴我一個關於貓的笑話，用JSON回應並且有`setup`和`punchline`和`rating`這些關鍵字")

print(structure_joke)
