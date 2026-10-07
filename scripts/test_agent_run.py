import asyncio
import json
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.llm import create_llm_client
from ai_software_engineering_agent.tools import create_default_tool_registry
from ai_software_engineering_agent.langgraph_agent import LangGraphAgent
from ai_software_engineering_agent.rag import RAGService
from ai_software_engineering_agent.vector_store import create_vector_store
from ai_software_engineering_agent.embeddings import create_embedding_client

async def main():
    settings = Settings.from_environment()
    llm = create_llm_client(settings)
    emb = create_embedding_client(settings)
    vs = create_vector_store(settings)
    rag = RAGService(vector_store=vs, embedding_client=emb, llm_client=llm, settings=settings)
    tools = create_default_tool_registry(rag, repo_id="yield-curve-lab")
    agent = LangGraphAgent(llm_client=llm, tool_registry=tools, settings=settings)

    # Let's run step by step or inspect graph
    async for event in agent.graph.astream({
        "messages": [
            {"role": "system", "content": agent._build_system_prompt()},
            {"role": "user", "content": "How is price_bond implemented?"},
        ],
        "task": "How is price_bond implemented?",
        "current_step": 0,
        "max_steps": 5,
        "status": "running",
        "final_answer": None,
        "citations": [],
        "trace_steps": [],
        "start_time": 0.0,
    }, config={"configurable": {"thread_id": "test_1"}}):
        for node_name, state_update in event.items():
            print(f"=== Node: {node_name} ===")
            print("Keys:", state_update.keys())
            if "messages" in state_update:
                for m in state_update["messages"]:
                    print(f"  Msg type: {type(m).__name__}, content: {repr(m.content)[:100]}, tool_calls: {getattr(m, 'tool_calls', None)}")
            if "citations" in state_update:
                print("  Citations count:", len(state_update["citations"]))
            if "status" in state_update:
                print("  Status:", state_update["status"])
            if "final_answer" in state_update:
                print("  Final answer:", repr(state_update["final_answer"])[:100])

asyncio.run(main())
