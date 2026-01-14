from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_brain import workflow
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
import sqlite3
import uvicorn
from langgraph.checkpoint.sqlite import SqliteSaver

app = FastAPI(title="SmartCloud SaaS API")

# Enable CORS for Angular integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    cpu: float
    ram: float
    company_id: str

@app.post("/chat")
async def chat_with_agent(req: ChatRequest):
    config = {"configurable": {"thread_id": req.company_id}}
    
    # We use a context manager for SQLite to prevent locking errors on Windows
    conn = sqlite3.connect("saas_storage.db", check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        # Re-compile with the saver for each request to ensure persistence
        agent = workflow.with_config(checkpointer=saver)
        
        inputs = {
            "messages": [HumanMessage(content=req.message)],
            "client_data": {"cpu": req.cpu, "ram": req.ram}
        }
        
        result = agent.invoke(inputs, config=config)
        return {
            "status": "success",
            "company": req.company_id,
            "response": result["messages"][-1].content
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
