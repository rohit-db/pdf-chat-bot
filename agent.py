import functools
import os
from typing import Any, Generator, Literal, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool
)
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel

# Configuration
GENIE_SPACE_ID = "01f025452ab7187abf84076f27b5d249"
GENIE_AGENT_DESCRIPTION = (
    "This genie agent can answer questions for querying usage data from billing documents. "
    "The pdf documents have been parsed specifically into predefined structured output so that billing details and usage details can be extracted. "
    "Especially if there are questions around total usage by months, this agent can help."
)
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
VECTOR_SEARCH_INDEX_NAME = "rohitb_demo.pdf_chat.chunked_pdf_docs_index"
VECTOR_SEARCH_INDEX_DESCRIPTION = (
    "The index contains extracted entities from pdf documents from all the bills. "
    "The bills are invoices from AT&T by month and include details like usage, terms and conditions etc."
)
RAG_AGENT_DESCRIPTION = (
    "This agent specializes in extracting relevant information from PDF bills stored. "
    "This can be used especially when there're specific pin point questions about a bill and not when there're questions like aggregation."
)
MAX_ITERATIONS = 3

# Functions to create agents and tools
def create_genie_agent():
    return GenieAgent(
        genie_space_id=GENIE_SPACE_ID,
        genie_agent_name="Genie",
        description=GENIE_AGENT_DESCRIPTION,
        client=WorkspaceClient(
            host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
            token=os.getenv("DATABRICKS_GENIE_PAT"),
        ),
    )

def create_llm():
    return ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

def create_tools():
    tools = []
    uc_toolkit = UCFunctionToolkit(function_names=[])
    tools.extend(uc_toolkit.tools)
    vector_search_tool = VectorSearchRetrieverTool(
        index_name=VECTOR_SEARCH_INDEX_NAME,
        tool_description=VECTOR_SEARCH_INDEX_DESCRIPTION
    )
    tools.append(vector_search_tool)
    return tools

def create_rag_agent(llm, tools):
    return create_react_agent(llm, tools=tools)

def create_supervisor_agent(llm, worker_descriptions):
    formatted_descriptions = "\n".join(
        f"- {name}: {desc}" for name, desc in worker_descriptions.items()
    )
    system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
    options = ["FINISH"] + list(worker_descriptions.keys())
    FINISH = {"next_node": "FINISH"}

    def supervisor_agent(state):
        count = state.get("iteration_count", 0) + 1
        if count > MAX_ITERATIONS:
            return FINISH
        
        class nextNode(BaseModel):
            next_node: Literal[tuple(options)]

        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
        supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
        next_node = supervisor_chain.invoke(state).next_node
        
        if state.get("next_node") == next_node:
            return FINISH
        return {
            "iteration_count": count,
            "next_node": next_node
        }
    
    return supervisor_agent

def create_workflow(genie_agent, rag_agent, supervisor_agent):
    def agent_node(state, agent, name):
        result = agent.invoke(state)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": result["messages"][-1].content,
                    "name": name,
                }
            ]
        }

    def final_answer(state):
        prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."
        preprocessor = RunnableLambda(
            lambda state: state["messages"] + [{"role": "user", "content": prompt}]
        )
        final_answer_chain = preprocessor | llm
        return {"messages": [final_answer_chain.invoke(state)]}

    class AgentState(ChatAgentState):
        next_node: str
        iteration_count: int

    rag_node = functools.partial(agent_node, agent=rag_agent, name="RAG")
    genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")

    workflow = StateGraph(AgentState)
    workflow.add_node("Genie", genie_node)
    workflow.add_node("RAG", rag_node)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("final_answer", final_answer)

    workflow.set_entry_point("supervisor")
    for worker in worker_descriptions.keys():
        workflow.add_edge(worker, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_node"],
        {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
    )
    workflow.add_edge("final_answer", END)
    return workflow.compile()

# Main script
genie_agent = create_genie_agent()
llm = create_llm()
tools = create_tools()
rag_agent = create_rag_agent(llm, tools)

worker_descriptions = {
    "Genie": GENIE_AGENT_DESCRIPTION,
    "RAG": RAG_AGENT_DESCRIPTION,
}

supervisor_agent = create_supervisor_agent(llm, worker_descriptions)
multi_agent = create_workflow(genie_agent, rag_agent, supervisor_agent)

class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg})
                    for msg in node_data.get("messages", [])
                )

mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)