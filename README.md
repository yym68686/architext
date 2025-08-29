
# Architext

[English](./README.md) | [中文](./README_CN.md)

[![PyPI version](https://badge.fury.io/py/architext.svg)](https://badge.fury.io/py/architext)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**Architext: The Context Engineering framework for building smarter, more reliable AI Agents.**

---

**Architext** (from "Architecture" + "Text") is a Python library designed for Large Language Model (LLM) applications, focusing on **Context Engineering**. It provides an elegant, powerful, and object-oriented set of tools that allow you to precisely and dynamically construct and reorganize the input context for LLMs, much like a software engineer designs an architecture.

Say goodbye to scattered string concatenation and complex construction logic, and enter a new era where context is treated as an **operable, composable, and evolvable engineered entity**.

## 🤔 What is Context Engineering?

When building complex AI Agents, the **quality and structure** of the context provided to the LLM (i.e., the `messages` list) directly determine its performance ceiling. Context Engineering is an emerging discipline that focuses on:

*   **Structuring**: How to organize information from various data sources (files, code, databases, APIs) into a structure that LLMs can most easily understand?
*   **Dynamism**: How to dynamically add, remove, or rearrange context content as the conversation progresses to maintain its relevance and timeliness?
*   **Optimization**: How to intelligently filter and present the most valuable information within a limited context window to maximize performance and minimize cost?

`Architext` is designed to solve these engineering challenges.

## ✨ Architext: Born for Context Engineering

The core philosophy of `Architext` is to elevate the context construction process from ad-hoc "craftsmanship" to systematic "engineering."

*   **Modularize Context**: Each information source (e.g., file content, tool lists) is an independent `ContextProvider` responsible for producing standardized `ContentBlock`s.
*   **Messages as Mutable Structures**: Messages are no longer static text but a container of objects that can be manipulated in real-time. You can perform precise `pop`, `insert`, and `append` operations on its internal content blocks as if they were precision components.
*   **Think like an Architect**: You can lay out the structure of `SystemMessage` and `UserMessage` as clearly as designing a software architecture, and dynamically adjust it through a unified interface to handle different task scenarios.

## 🚀 Core Features

*   **Object-Oriented Context Modeling**: Treat `SystemMessage`, `UserMessage`, etc., as first-class, operable citizens.
*   **Atomic Content Blocks (`ContentBlock`)**: Decompose context into the smallest units that can be independently manipulated and moved.
*   **List-like Dynamic Operations**: Achieve real-time, precise control over context content using methods like `pop()` and `insert()`.
*   **Provider-Driven Architecture**: Easily connect to any data source through an extensible `ContextProvider` system.
*   **Intelligent Caching and On-Demand Refresh**: Built-in efficient caching mechanism that only refreshes when the data source changes, significantly improving performance.
*   **Unified Pass-Through Interface**: Directly access and control any underlying `ContextProvider` through the top-level `Messages` object for centralized state management.

## 📦 Installation

```bash
pip install architext
```

## 🚀 Quick Start: A Context Engineering Practice

The core of `Architext` lies in its intuitive and flexible API. The following series of independent examples demonstrate how to use it for efficient context engineering.

### Example 1: Basic Layout and Initial Rendering

This is the most basic usage. We declaratively build a conversation structure containing `System` and `User` messages.

```python
# --- Example 1: Basic Layout ---
import asyncio
from architext import Messages, SystemMessage, UserMessage, Texts, Tools

async def example_1():
    # 1. Define your context providers
    tools_provider = Tools(tools_json=[{"name": "run_test"}])
    system_prompt = Texts("system_prompt", "You are a professional AI code reviewer.")

    # 2. Declaratively build the message list
    messages = Messages(
        SystemMessage(system_prompt, tools_provider),
        UserMessage(Texts("user_input", "Please help me review the following Python code."))
    )

    # 3. Render the final messages list
    print("--- Example 1: Render Result ---")
    # .render_latest() automatically refreshes and then renders
    for msg in await messages.render_latest():
        print(msg)

asyncio.run(example_1())
```

**Expected Output:**
```
--- Example 1: Render Result ---
{'role': 'system', 'content': 'You are a professional AI code reviewer.\n\n<tools>[{\'name\': \'run_test\'}]</tools>'}
{'role': 'user', 'content': 'Please help me review the following Python code.'}
```

---

### Example 2: Pass-Through Updates and Automatic Refresh

The power of `Architext` is that you can update the underlying context sources at any time, and the system will automatically and efficiently refresh the content on the next render.

```python
# --- Example 2: Pass-Through Update ---
import asyncio
from architext import Messages, UserMessage, Files

async def example_2():
    # 1. Initialize a message with a files provider
    files_provider = Files()
    messages = Messages(
        UserMessage(files_provider)
    )

    # 2. At this moment, the file content is empty, so the render result is an empty list
    print("--- Initial State (File content is empty) ---")
    print(await messages.render_latest())

    # 3. Update the file content via the pass-through interface
    # This automatically marks the files_provider as "dirty"
    print("\n>>> Updating files via messages.provider...")
    file_instance = messages.provider("files")
    if file_instance:
        file_instance.update("main.py", "def main():\n    pass")

    # 4. Render again, Architext will automatically refresh the dirty provider
    print("\n--- Render After Update ---")
    for msg in await messages.render_latest():
        print(msg)

asyncio.run(example_2())
```

**Expected Output:**
```
--- Initial State (File content is empty) ---
[]

>>> Updating files via messages.provider...

--- Render After Update ---
{'role': 'user', 'content': "<files>\n<file path='main.py'>def main():\n    pass...</file>\n</files>"}
```

---

### Example 3: Dynamic Context Refactoring (`pop` and `insert`)

This is the core practice of **Context Engineering**. You can dynamically move a content block from one message to another, just like manipulating a list, to adapt to different task requirements.

```python
# --- Example 3: Dynamic Refactoring ---
import asyncio
from architext import Messages, SystemMessage, UserMessage, Texts, Tools

async def example_3():
    # 1. Initial layout: tools are in SystemMessage
    tools_provider = Tools(tools_json=[{"name": "run_test"}])
    messages = Messages(
        SystemMessage(tools_provider),
        UserMessage(Texts("user_input", "Analyze the code and run the tests."))
    )
    print("--- Initial Layout ---")
    for msg in await messages.render_latest(): print(msg)

    # 2. Runtime decision: Move the tools context to the user message for stronger instruction
    print("\n>>> Refactoring context: Moving 'tools' block to UserMessage...")

    # a. Globally pop the 'tools' provider from any message
    popped_tools_provider = messages.pop("tools")

    # b. Precisely locate the UserMessage by index (messages[1]) and insert it
    if popped_tools_provider:
        messages[1].insert(0, popped_tools_provider)

    # 3. View the refactored result
    print("\n--- Final Layout After Refactoring ---")
    # No refresh needed, so we use the synchronous .render()
    for msg in messages.render(): print(msg)

asyncio.run(example_3())
```

**Expected Output:**
```
--- Initial Layout ---
{'role': 'system', 'content': "<tools>[{'name': 'run_test'}]</tools>"}
{'role': 'user', 'content': 'Analyze the code and run the tests.'}

>>> Refactoring context: Moving 'tools' block to UserMessage...

--- Final Layout After Refactoring ---
{'role': 'system', 'content': ''}
{'role': 'user', 'content': "<tools>[{'name': 'run_test'}]</tools>\n\nAnalyze the code and run the tests."}
```
*(Note: The content of SystemMessage becomes empty because its only block was moved, so it might be filtered out in the final rendering)*

---

### Example 4: Multimodal Context (Text + Image)

`Architext` natively supports multimodal context construction, automatically formatting the output to match APIs like OpenAI's.

```python
# --- Example 4: Multimodal ---
import asyncio
from architext import Messages, UserMessage, Texts, Images

async def example_4():
    # Create a dummy image for the example
    with open("example_image.png", "w") as f: f.write("dummy")

    messages = Messages(
        UserMessage(
            Texts("prompt", "What is in this image?"),
            Images("example_image.png") # name is optional
        )
    )

    print("--- Multimodal Render Result ---")
    for msg in await messages.render_latest():
        # Hide the long base64 string for readability
        for part in msg['content']:
            if part['type'] == 'image_url':
                part['image_url']['url'] = part['image_url']['url'][:80] + "..."
        print(msg)

    # Clean up the dummy file
    import os
    os.remove("example_image.png")

asyncio.run(example_4())
```

**Expected Output:**
```
--- Multimodal Render Result ---
{'role': 'user', 'content': [{'type': 'text', 'text': 'What is in this image?'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,ZHVtbXk=...'}}]}
```

## 🤝 Contributing

Context Engineering is an exciting new field. We welcome contributions of all forms to jointly explore building smarter, more efficient AI Agents. Whether it's reporting a bug, proposing a new feature, or submitting code, your participation is crucial.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
