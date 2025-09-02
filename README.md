# Architext

[English](./README.md) | [中文](./README_CN.md)

[![PyPI version](https://img.shields.io/pypi/v/architext)](https://pypi.org/project/architext/)
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
*   **Powerful List-like Operations**: Manipulate messages with `pop()`, `insert()`, `append()`, indexing (`messages[0]`), slicing (`messages[1:3]`), and even slice assignment (`messages[1:] = ...`).
*   **Pythonic & Idiomatic**: Enjoy a natural coding experience. Messages can be concatenated with `+`, content accessed via dictionary-style keys (`msg['content']`), and internal providers accessed via list-style indexing (`msg[0]`).
*   **Visibility Control**: Toggle providers on and off with `.visible = False` without removing them, enabling dynamic context filtering.
*   **Bulk Operations**: Use `ProviderGroup` to manage multiple providers with the same name simultaneously (e.g., `messages.provider("explanation").visible = False`).
*   **Intelligent Caching**: Built-in mechanism automatically refreshes content only when the source changes, boosting performance.
*   **Unified Pass-Through Interface**: Access and update any provider from the top-level `Messages` object via `messages.provider("name")`.
*   **Native Multimodal Support**: Effortlessly create messages containing both text and images.

## 📦 Installation

```bash
pip install architext
```

## 🚀 Quick Start: Real-World Scenarios

The following scenarios demonstrate how Architext solves common, yet complex, context engineering challenges with remarkable simplicity.

### Scenario 1: Dynamic Context Across Environments

An agent developed on Windows needs to run on a Mac. Manually updating a hardcoded system prompt is tedious and error-prone. Architext makes this dynamic.

```python
import json
import time
import asyncio
import platform
from datetime import datetime
from architext import Messages, SystemMessage, Texts

async def example_1():
    # Lambda functions are re-evaluated every time `render_latest` is called.
    messages = Messages(
        SystemMessage(f"OS: {Texts(lambda: platform.platform())}, Time: {Texts(lambda: datetime.now().isoformat())}")
    )

    print("--- First Render (e.g., on MacOS) ---")
    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2))

    time.sleep(1)

    print("\n--- Second Render (Time updated) ---")
    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2))

asyncio.run(example_1())
```
**Why it's powerful:** No manual intervention needed. `platform.platform()` and `datetime.now()` are evaluated at render time. This transforms static string concatenation into declarative, dynamic context construction. You declare *what* information you need, and Architext injects the latest state at runtime.

### Scenario 2: Intelligent File Management

When an agent processes files, you often have to manually inject the latest file content into the prompt. Architext automates this.

```python
import json
import asyncio
from architext import Messages, UserMessage, Files, SystemMessage

async def example_2():
    with open("main.py", "w") as f: f.write("print('hello')")

    messages = Messages(
        SystemMessage("Analyze this file:", Files(name="code_files")),
        UserMessage("hi")
    )

    # The agent "reads" the file. We just need to tell the provider its path.
    messages.provider("code_files").update("main.py")

    # `render_latest()` automatically reads the file from disk.
    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2))

    import os
    os.remove("main.py")

asyncio.run(example_2())
```
**Why it's powerful:** `messages.render_latest()` always gets the most up-to-date file content, even if the file is modified on disk *during* the agent's run. It handles reading, formatting, and injection automatically.

### Scenario 3: Effortless Context Refactoring

Need to move a block of context, like file contents, from a system message to a user message? Traditionally, this is a nightmare of string manipulation, especially with multimodal content. With Architext, it's two lines of code.

```python
import json
import asyncio
from architext import Messages, UserMessage, Files, SystemMessage, Images

async def example_3():
    with open("main.py", "w") as f: f.write("print('hello')")
    with open("image.png", "w") as f: f.write("dummy")

    messages = Messages(
        SystemMessage("Code:", Files("main.py", name="code_files")),
        UserMessage("hi", Images("image.png"))
    )

    print("--- Before Moving ---")
    print(json.dumps(await messages.render_latest(), indent=2))

    # Move the entire Files block to the user message
    files_provider = messages.pop("code_files")
    messages[1].append(files_provider) # Append to the end

    print("\n--- After Moving ---")
    print(json.dumps(await messages.render_latest(), indent=2))

    # Prepending is just as easy: messages[1] = files_provider + messages[1]

    import os
    os.remove("main.py")
    os.remove("image.png")

asyncio.run(example_3())
```
**Why it's powerful:** `messages.pop("code_files")` finds and removes the provider by name, regardless of its location. Architext automatically handles the complexity of multimodal message structures, letting you focus on logic, not data format.

### Scenario 4: Granular Visibility Control for Prompt Optimization

To prevent model output truncation, a common trick is to add an instruction to the *final* user prompt. Managing this manually is complex. Architext provides precise visibility control.

```python
import json
import asyncio
from architext import Messages, SystemMessage, Texts, UserMessage, AssistantMessage

async def example_4():
    # Add the same named provider to multiple messages
    done_marker = Texts("\n\nYour message **must** end with [done].", name="done_marker")

    messages = Messages(
        SystemMessage("You are helpful."),
        UserMessage("hi", done_marker),
        AssistantMessage("hello"),
        UserMessage("hi again", done_marker),
    )

    # 1. Hide all instances of the "done_marker" provider
    messages.provider("done_marker").visible = False
    # 2. Make only the very last instance visible
    messages.provider("done_marker")[-1].visible = True

    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2))

asyncio.run(example_4())
```
**Why it's powerful:** By naming providers, you can target them for bulk operations. A single line hides all instances, and another selectively re-enables just the one you need. This is a powerful pattern for conditional prompting, A/B testing, or managing system instructions across a long conversation.

## 🤝 Contributing

Context Engineering is an exciting new field. We welcome contributions of all forms to jointly explore building smarter, more efficient AI Agents. Whether it's reporting a bug, proposing a new feature, or submitting code, your participation is crucial.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
