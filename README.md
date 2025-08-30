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

*   **Declarative & Dynamic**: Seamlessly construct prompts with Python's f-strings, embedding dynamic, stateful components directly within your text.
*   **Context as a Mutable Structure**: Messages are no longer static text but a container of `Provider` objects that can be manipulated in real-time. You can perform precise `pop`, `insert`, `append`, and even slicing operations.
*   **Granular State Management**: Each piece of context is a `Provider` that can be individually updated, cached, and even hidden from rendering without being removed.
*   **Think like an Architect**: You can lay out the structure of `SystemMessage` and `UserMessage` as clearly as designing a software architecture, and dynamically adjust it through a unified interface to handle different task scenarios.

## 🚀 Core Features

*   **Intuitive F-String Integration**: Build complex prompts naturally with f-strings, embedding providers like `Texts()`, `Files()`, and `Tools()` directly.
*   **Object-Oriented Context Modeling**: Treat `SystemMessage`, `UserMessage`, etc., as first-class, operable Python objects.
*   **Provider-Driven Architecture**: Extensible `ContextProvider` system (`Texts`, `Files`, `Images`, `Tools`) to connect any data source.
*   **Dynamic Content with `lambda`**: `Texts(lambda: ...)` providers can execute code to generate content on-the-fly during rendering.
*   **Powerful List-like Operations**: Manipulate messages with `pop()`, `insert()`, `append()`, indexing (`messages[0]`), and slicing (`messages[1:3]`).
*   **Visibility Control**: Toggle providers on and off with `.visible = False` without removing them, enabling dynamic context filtering.
*   **Bulk Operations**: Use `ProviderGroup` to manage multiple providers with the same name simultaneously (e.g., `messages.provider("explanation").visible = False`).
*   **Intelligent Caching**: Built-in mechanism automatically refreshes content only when the source changes, boosting performance.
*   **Unified Pass-Through Interface**: Access and update any provider from the top-level `Messages` object via `messages.provider("name")`.
*   **Native Multimodal Support**: Effortlessly create messages containing both text and images.

## 📦 Installation

```bash
pip install architext
```

## 🚀 Quick Start: A Context Engineering Practice

The following examples showcase Architext's most powerful features, ordered from the most unique to the foundational.

### Example 1: The Magic of F-String Prompt Construction (Highlight Feature)

Forget manual string joining. Build prompts declaratively and dynamically using the tools you already know and love: f-strings.

```python
import asyncio
from architext import Messages, UserMessage, Texts, Tools, Files
from datetime import datetime

async def example_1():
    # Define providers that will be embedded in the f-string
    os_provider = Texts("MacOS Sonoma", name="os_version")
    tools_provider = Tools([{"name": "read_file"}])
    files_provider = Files(["main.py", "utils.py"])
    time_provider = Texts(lambda: datetime.now().isoformat()) # Dynamic content!

    # Create dummy files for the example
    with open("main.py", "w") as f: f.write("print('hello')")
    with open("utils.py", "w") as f: f.write("def helper(): pass")

    # Construct the entire message with a single f-string!
    # Architext automatically detects and manages the embedded providers.
    prompt = f"""
    System Information:
    - OS: {os_provider}
    - Time: {time_provider}

    Available Tools: {tools_provider}

    File Contents:
    {files_provider}

    User Request:
    Based on the files, what is the primary function of this project?
    """

    messages = Messages(UserMessage(prompt))

    # Render the fully constructed message
    print("--- F-String Render Result ---")
    for msg in await messages.render_latest():
        print(msg['content'])

    # Clean up dummy files
    import os
    os.remove("main.py")
    os.remove("utils.py")

asyncio.run(example_1())
```

**Expected Output:** The f-string is fully resolved with content from all providers, including the dynamically generated timestamp and file contents.

---

### Example 2: Dynamic Context Refactoring & Visibility Control

Adapt the context in real-time based on application logic. Here, we move a tool definition and then hide multiple "explanation" providers at once.

```python
import asyncio
from architext import Messages, SystemMessage, UserMessage, Texts, Tools

async def example_2():
    messages = Messages(
        SystemMessage(
            Texts("You are an AI assistant.", name="intro"),
            Tools([{"name": "run_code"}]) # Initially in SystemMessage
        ),
        UserMessage(
            Texts("First explanation.", name="explanation"),
            Texts("Please run the code.", name="request"),
            Texts("Second explanation.", name="explanation")
        )
    )

    # --- Part A: Move a provider ---
    print(">>> Refactoring: Moving 'tools' to UserMessage for emphasis...")

    # 1. Globally pop the provider from wherever it is
    tools_provider = messages.pop("tools")
    # 2. Insert it into a specific message and position
    if tools_provider:
        messages[1].insert(1, tools_provider)

    print("\n--- After Moving 'tools' ---")
    for msg in await messages.render_latest(): print(msg)

    # --- Part B: Bulk-hide providers ---
    print("\n>>> Hiding all 'explanation' providers...")

    # 1. Get a group of all providers named "explanation"
    explanation_group = messages.provider("explanation")
    # 2. Set visibility for the entire group
    explanation_group.visible = False

    print("\n--- After Hiding Explanations ---")
    for msg in await messages.render_latest(): print(msg)

asyncio.run(example_2())
```

**Expected Output:** You will see the `<tools>` block move from the system to the user message. Then, in the final output, the "First explanation" and "Second explanation" texts will disappear, while the rest of the content remains.

---

### Example 3: Multimodal and Tool-Use Conversations

Architext natively supports complex message structures required for multimodal interactions and tool-use cycles.

```python
import asyncio
from architext import Messages, UserMessage, AssistantMessage, Texts, Images, ToolCalls, ToolResults

async def example_3():
    # --- Multimodal Example ---
    with open("dummy_image.png", "w") as f: f.write("dummy")

    multimodal_messages = Messages(
        UserMessage(
            "What is in this image?",
            Images("dummy_image.png")
        )
    )
    print("--- Multimodal Render Result ---")
    for msg in await multimodal_messages.render_latest(): print(msg)

    # --- Tool-Use Example ---
    tool_use_messages = Messages(
        UserMessage("What is 5 + 10?"),
        # Represents the model's request to call a tool
        ToolCalls([{'id': 'call_123', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 5, "b": 10}'}}]),
        # Represents the result you provide back to the model
        ToolResults(tool_call_id="call_123", content="15"),
        AssistantMessage("The sum is 15.")
    )
    print("\n--- Tool-Use Render Result ---")
    for msg in await tool_use_messages.render_latest(): print(msg)

    import os
    os.remove("dummy_image.png")

asyncio.run(example_3())
```

**Expected Output:** Both examples will render into the precise dictionary format expected by modern LLM APIs (like OpenAI's), handling list-based content for multimodal messages and `tool_calls`/`tool` roles correctly.

---

### Example 4: Pass-Through Updates and Automatic Refresh

Update any piece of context from anywhere, and Architext will ensure the changes are reflected in the next render.

```python
import asyncio
from architext import Messages, UserMessage, Files

async def example_4():
    # 1. Initialize with a Files provider
    messages = Messages(UserMessage(Files(name="code_files")))

    # 2. Initially, content is empty
    print("--- Initial State (No files loaded) ---")
    print(await messages.render_latest())

    # 3. Get a handle to the provider and update it
    print("\n>>> Updating files via messages.provider('code_files')...")
    files_provider = messages.provider("code_files")
    if files_provider:
        files_provider.update("main.py", "def main():\n    print('Hello')")

    # 4. Render again. Architext detects the stale provider and refreshes it.
    print("\n--- Render After Update ---")
    for msg in await messages.render_latest(): print(msg)

asyncio.run(example_4())
```

**Expected Output:** The first render will be empty. After the update, the second render will correctly show the content of `main.py`.

## 🤝 Contributing

Context Engineering is an exciting new field. We welcome contributions of all forms to jointly explore building smarter, more efficient AI Agents. Whether it's reporting a bug, proposing a new feature, or submitting code, your participation is crucial.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
