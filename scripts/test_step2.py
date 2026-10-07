import asyncio
import json
import os
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.llm import create_llm_client, LLMRequest
from ai_software_engineering_agent.tools import create_default_tool_registry
from ai_software_engineering_agent.rag import RAGService
from ai_software_engineering_agent.vector_store import create_vector_store
from ai_software_engineering_agent.embeddings import create_embedding_client
from ai_software_engineering_agent.langgraph_agent import LangGraphAgent
from openai import OpenAI

async def main():
    settings = Settings.from_environment()
    llm = create_llm_client(settings)
    emb = create_embedding_client(settings)
    vs = create_vector_store(settings)
    rag = RAGService(vector_store=vs, embedding_client=emb, llm_client=llm, settings=settings)
    tools = create_default_tool_registry(rag, repo_id="yield-curve-lab")
    agent = LangGraphAgent(llm_client=llm, tool_registry=tools, settings=settings)

    prompt = agent._build_system_prompt()
    openai_tools = tools.to_openai_tools()

    # Step 1
    req1 = LLMRequest(
        prompt="How is price_bond implemented?",
        system_instruction=prompt,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "How is price_bond implemented?"},
        ],
        tools=openai_tools,
    )
    r1 = await llm.generate(req1)
    tc = r1.tool_calls[0]

    # Execute tool
    from ai_software_engineering_agent.agent_state import AgentState, ToolCall
    st = AgentState(task="How is price_bond implemented?", max_steps=5)
    obs = await tools.execute(ToolCall(tool_name=tc.name, arguments=tc.arguments, call_id=tc.call_id), st)

    # Call OpenAI SDK directly to see exact raw message
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    raw_resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "How is price_bond implemented?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": obs.content,
            },
        ],
        tools=openai_tools,
        max_completion_tokens=8192,
        extra_body={"reasoning_effort": "low", "include_reasoning": False},
    )
    print("RAW CHOICE:", raw_resp.choices[0])

asyncio.run(main())
