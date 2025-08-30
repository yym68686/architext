import os
import json
import asyncio
from openai import AsyncOpenAI

# 从我们设计的 architext 库中导入消息类
from architext.core import (
    Messages,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCalls,
    ToolResults,
    Texts,
)

def _add_tool(a: int, b: int) -> int:
    """(工具函数) 计算两个整数的和。"""
    print(f"Executing tool: add(a={a}, b={b})")
    return a + b

async def main():
    """
    一个简化的、函数式的流程，用于处理单个包含工具调用的用户查询。
    """
    print("Starting simplified Tool Use demonstration...")

    # --- 1. 初始化 ---
    # 确保环境变量已设置
    if not os.getenv("API_KEY"):
        print("\nERROR: API_KEY environment variable not set.")
        return

    client = AsyncOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
    model = os.getenv("MODEL", "gpt-4o-mini")

    # 定义工具
    tool_executors = { "add": _add_tool }
    tools_definition = [{
        "type": "function", "function": {
            "name": "add", "description": "Calculate the sum of two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first integer."},
                    "b": {"type": "integer", "description": "The second integer."},
                }, "required": ["a", "b"],
            },
        },
    }]

    # --- 2. 处理查询 ---
    # 初始消息
    messages = Messages(
        SystemMessage(Texts("system_prompt", "You are a helpful assistant. You must use the provided tools to answer questions.")),
        UserMessage(Texts("user_question", "What is the sum of 5 and 10?"))
    )

    # 第一次 API 调用
    print("\n--- [Step 1] Calling OpenAI with tools...")
    response = await client.chat.completions.create(
        model=model,
        messages=await messages.render_latest(),
        tools=tools_definition,
        tool_choice="auto",
    )
    response_message = response.choices[0].message

    # 检查是否需要工具调用
    if not response_message.tool_calls:
        final_content = response_message.content or ""
        messages.append(AssistantMessage(Texts("assistant_response", final_content)))
    else:
        # 执行工具调用
        print("--- [Step 2] Assistant requested tool calls. Executing them...")
        messages.append(ToolCalls(response_message.tool_calls))

        for tool_call in response_message.tool_calls:
            if tool_call.function is None: continue

            executor = tool_executors.get(tool_call.function.name)
            if not executor: continue

            try:
                args = json.loads(tool_call.function.arguments)
                result = executor(**args)
                messages.append(ToolResults(tool_call_id=tool_call.id, content=str(result)))
                print(f"  - Executed '{tool_call.function.name}'. Result: {result}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  - Error processing tool call '{tool_call.function.name}': {e}")

        # 第二次 API 调用
        print("--- [Step 3] Calling OpenAI with tool results for final answer...")
        final_response = await client.chat.completions.create(
            model=model,
            messages=await messages.render_latest(),
        )
        final_content = final_response.choices[0].message.content or ""
        messages.append(AssistantMessage(Texts("final_response", final_content)))

    # --- 3. 显示结果 ---
    print("\n--- Final request body sent to OpenAI: ---")
    print(json.dumps(await messages.render_latest(), indent=2, ensure_ascii=False))

    print("\n--- Final Assistant Answer ---")
    print(final_content)
    print("\nDemonstration finished.")

if __name__ == "__main__":
    asyncio.run(main())

"""
[
  {
    "role": "system",
    "content": "You are a helpful assistant. You must use the provided tools to answer questions."
  },
  {
    "role": "user",
    "content": "What is the sum of 5 and 10?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_rddWXkDikIxllRgbPrR6XjtMVSBPv",
        "type": "function",
        "function": {
          "name": "add",
          "arguments": "{\"b\": 10, \"a\": 5}"
        }
      }
    ],
    "content": null
  },
  {
    "role": "tool",
    "tool_call_id": "call_rddWXkDikIxllRgbPrR6XjtMVSBPv",
    "content": "15"
  },
  {
    "role": "assistant",
    "content": "The sum of 5 and 10 is 15."
  }
]
"""