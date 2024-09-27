import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# 1. Create model
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4o")

# 2. Create prompt template
system_template = "創建一個對話場景包含 {character1} 和 {character2}."


prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{dialogue_start}")]
)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)


@app.get("/")
async def read_root():
    return RedirectResponse(url="/chain/playground")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
