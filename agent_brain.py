import os
import operator
from typing import TypedDict, Annotated, List
from langchain_groq import ChatGroq 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- CONFIG ---
MY_GROQ_KEY = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    client_data: dict 

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=MY_GROQ_KEY, temperature=0.1)

# --- NODES ---
def supervisor_router(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], AIMessage):
        return {"next_agent": "FINISH"}
    prompt = "Route to MONITORING_AGENT for hardware stats, or TASK_AGENT for chat. Reply ONLY with the name."
    response = llm.invoke(prompt)
    decision = response.content.upper()
    if "MONITORING" in decision: return {"next_agent": "MONITORING_AGENT"}
    if "TASK" in decision: return {"next_agent": "TASK_AGENT"}
    return {"next_agent": "FINISH"}

def monitoring_agent(state: AgentState):
    stats = state.get("client_data", {})
    cpu = stats.get("cpu", 0)
    ram = stats.get("ram", 0)
    status = "Healthy" if cpu < 80 else "Critical"
    report = f"System Status: {status}. Metrics -> CPU: {cpu}%, RAM: {ram}%."
    return {"messages": [AIMessage(content=report)]}

def task_agent(state: AgentState):
    response = llm.invoke(state['messages'])
    return {"messages": [AIMessage(content=response.content)]}

# --- GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_router)
builder.add_node("monitoring_agent", monitoring_agent)
builder.add_node("task_agent", task_agent)

builder.set_entry_point("supervisor")
builder.add_conditional_edges("supervisor", lambda x: x["next_agent"], 
                              {"MONITORING_AGENT": "monitoring_agent", "TASK_AGENT": "task_agent", "FINISH": END})
builder.add_edge("monitoring_agent", "supervisor")
builder.add_edge("task_agent", "supervisor")

# IMPORTANT: We export the builder, not the compiled workflow
workflow_builder = builder
