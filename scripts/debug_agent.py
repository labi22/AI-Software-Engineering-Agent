import os
import asyncio
import json

from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.llm import create_llm_client, LLMRequest

settings = Settings(
    llm_provider="groq",
    groq_api_key=os.environ["GROQ_API_KEY"],
    llm_model="openai/gpt-oss-20b",
)
client = create_llm_client(settings)

async def test():
    req = LLMRequest(
        prompt="How is price_bond implemented?",
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
    res = await client.generate(req)
    print("First res: tool_calls=", res.tool_calls, "text=", repr(res.text))

    req2 = LLMRequest(
        prompt="How is price_bond implemented?",
        system_instruction="You are an AI SWE agent.",
        messages=[
            {"role": "system", "content": "You are an AI SWE agent."},
            {"role": "user", "content": "How is price_bond implemented?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "arguments": json.dumps({"query": "price_bond"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "bond_engine/bond_pricer.py:29-36:\ndef price_bond(bond, yc):\n    price = sum(...)",
            },
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search code",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        }],
    )
    res2 = await client.generate(req2)
    print("Second res: tool_calls=", res2.tool_calls, "text=", repr(res2.text))

asyncio.run(test())
