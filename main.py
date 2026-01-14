from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_brain import workflow
# UPDATED IMPORT: This matches the 'langgraph-checkpoint-sqlite' library
from langgraph.checkpoint.sqlite import SqliteSaver 
from langchain_core.messages import HumanMessage
import sqlite3
import uvicorn

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
    # This thread_id keeps different companies' chats separate
    config = {"configurable": {"thread_id": req.company_id}}
    
    # Connect to the database file
    conn = sqlite3.connect("saas_storage.db", check_same_thread=False)
    
    try:
        # 1. Initialize the saver
        saver = SqliteSaver(conn)
        
        # 2. Compile the agent WITH the saver inside the request
        # This is the most reliable way to ensure memory works on the cloud
        agent = workflow.compile(checkpointer=saver)
        
        inputs = {
            "messages": [HumanMessage(content=req.message)],
            "client_data": {"cpu": req.cpu, "ram": req.ram}
        }
        
        # 3. Get response
        result = agent.invoke(inputs, config=config)
        
        return {
            "status": "success",
            "company": req.company_id,
            "response": result["messages"][-1].content
        }
    except Exception as e:
        print(f"Error: {e}") # This will show in Render logs
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

if __name__ == "__main__":
    # Render uses an environment variable for PORT, 8000 is for local testing
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
