import asyncio
import json
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.llm import create_llm_client, LLMRequest

async def main():
    s = Settings.from_environment()
    client = create_llm_client(s)
    req = LLMRequest(
        prompt="How is price_bond implemented in this codebase?",
        system_instruction="You are an AI SWE agent.",
        tools=[{
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search code",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        }]
    )
    r1 = await client.generate(req)
    print("R1 tool_calls:", r1.tool_calls)
    if not r1.tool_calls:
        return
    tc = r1.tool_calls[0]
    req2 = LLMRequest(
        prompt="How is price_bond implemented in this codebase?",
        system_instruction="You are an AI SWE agent.",
        messages=[
            {"role": "user", "content": "How is price_bond implemented in this codebase?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": tc.call_id, "type": "function", "function": {"name": "search_code", "arguments": json.dumps(tc.arguments)}}]},
            {"role": "tool", "tool_call_id": tc.call_id, "content": "Found in bond_engine/bond_pricer.py:29-36:\ndef price_bond(bond, yc):\n    price = sum(...)"}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search code",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        }]
    )
    r2 = await client.generate(req2)
    print("R2 text:", repr(r2.text), "tool_calls:", r2.tool_calls)

asyncio.run(main())
