# Architext

[English](./README.md) | [中文](./README_CN.md)

[![PyPI version](https://img.shields.io/pypi/v/architext)](https://pypi.org/project/architext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**Architext: 为构建更智能、更可靠的 AI Agent 而生的上下文工程框架。**

---

**Architext** (源自 "Architecture" + "Text") 是一个为大语言模型（LLM）应用设计的、专注于**上下文工程 (Context Engineering)** 的 Python 库。它提供了一套优雅、强大且面向对象的工具，让您能够像软件工程师设计架构一样，精确、动态地构建和重组 LLM 的输入上下文。

告别散乱的字符串拼接和复杂的构建逻辑，进入一个将上下文视为**可操作、可组合、可演进的工程化实体**的新时代。

## 🤔 什么是上下文工程 (Context Engineering)？

在构建复杂的 AI Agent 时，提供给 LLM 的上下文（即 `messages` 列表）的**质量和结构**直接决定了其性能的上限。上下文工程是一门新兴的学科，它关注于：

*   **结构化（Structuring）**: 如何将来自不同数据源（文件、代码、数据库、API）的信息，组织成 LLM 最易于理解的结构？
*   **动态化（Dynamism）**: 如何根据对话的进展，实时地添加、移除或重排上下文内容，以保持其相关性和时效性？
*   **优化（Optimization）**: 如何在有限的上下文窗口内，智能地筛选和呈现最高价值的信息，以最大化性能并最小化成本？

`Architext` 正是为解决这些工程化挑战而生。

## ✨ Architext: 为上下文工程而生

`Architext` 的核心理念是将上下文的构建过程从临时的“手工艺”提升为系统化的“工程学”。

*   **声明式与动态化**: 使用 Python f-string 无缝构建提示，将动态、有状态的组件直接嵌入文本中。
*   **视上下文为可变结构**: 消息不再是静态文本，而是一个可被实时操作的 `Provider` 对象容器。您可以执行精确的 `pop`、`insert`、`append` 甚至切片操作。
*   **精细的状态管理**: 每一段上下文都是一个 `Provider`，可以被独立更新、缓存，甚至在不移除的情况下隐藏。
*   **以架构师的思维构建**: 您可以像设计软件架构一样清晰地布局 `SystemMessage` 和 `UserMessage` 的结构，并通过统一接口动态调整以应对不同任务场景。

## 🚀 核心特性 (Features)

*   **直观的 F-String 集成**: 使用 f-string 自然地构建复杂提示，直接嵌入 `Texts()`、`Files()`、`Tools()` 等提供者。
*   **面向对象的上下文建模**: 将 `SystemMessage`、`UserMessage` 等视为可操作的 Python 一等公民。
*   **提供者驱动架构**: 可扩展的 `ContextProvider` 体系 (`Texts`, `Files`, `Images`, `Tools`)，用于连接任何数据源。
*   **使用 `lambda` 实现动态内容**: `Texts(lambda: ...)` 提供者可以在渲染时即时执行代码生成内容。
*   **强大的列表式操作**: 使用 `pop()`、`insert()`、`append()`、索引 (`messages[0]`)、切片 (`messages[1:3]`) 甚至切片赋值 (`messages[1:] = ...`) 来操作消息。
*   **Pythonic & 风格统一**: 享受自然的编码体验。消息可以通过 `+` 进行拼接，内容可以通过字典风格的键 (`msg['content']`) 访问，内部的 provider 也可以通过列表风格的索引 (`msg[0]`) 访问。
*   **可见性控制**: 通过 `.visible = False` 切换提供者的渲染状态而无需移除它们，实现动态上下文过滤。
*   **批量操作**: 使用 `ProviderGroup` 同时管理多个同名提供者 (例如 `messages.provider("explanation").visible = False`)。
*   **智能缓存**: 内置机制仅在数据源变化时自动刷新内容，提升性能。
*   **统一的穿透式接口**: 通过顶层 `Messages` 对象的 `messages.provider("name")` 访问和更新任何提供者。
*   **原生多模态支持**: 轻松创建包含文本和图片的消息。

## 📦 安装 (Installation)

```bash
pip install architext
```

## 🚀 快速上手: 真实世界场景

以下场景展示了 Architext 如何以惊人的简洁性解决常见但复杂的上下文工程挑战。

### 场景 1: 跨环境的动态上下文

一个在 Windows 上开发的 Agent 需要在 Mac 上运行。手动更新硬编码的系统提示既繁琐又容易出错。Architext 使其动态化。

```python
import json
import time
import asyncio
import platform
from datetime import datetime
from architext import Messages, SystemMessage, Texts

async def example_1():
    # Lambda 函数在每次调用 `render_latest` 时都会被重新求值。
    messages = Messages(
        SystemMessage(f"操作系统: {Texts(lambda: platform.platform())}, 时间: {Texts(lambda: datetime.now().isoformat())}")
    )

    print("--- 第一次渲染 (例如, 在 MacOS 上) ---")
    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2, ensure_ascii=False))

    time.sleep(1)

    print("\n--- 第二次渲染 (时间已更新) ---")
    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2, ensure_ascii=False))

asyncio.run(example_1())
```
**它为何强大:** 无需任何手动干预。`platform.platform()` 和 `datetime.now()` 在渲染时被求值。这彻底将静态的字符串拼接革命为声明式的、动态的上下文构建。你只需声明*需要什么*信息，Architext 会在运行时为你注入最新状态。

### 场景 2: 智能文件管理

当 Agent 处理文件时，你常常需要手动将最新的文件内容注入到提示中。Architext 自动化了这一过程。

```python
import json
import asyncio
from architext import Messages, UserMessage, Files, SystemMessage

async def example_2():
    with open("main.py", "w", encoding="utf-8") as f: f.write("print('你好')")

    messages = Messages(
        SystemMessage("分析这个文件:", Files(name="code_files")),
        UserMessage("hi")
    )

    # Agent "读取" 了文件。我们只需告诉 provider 它的路径。
    messages.provider("code_files").update("main.py")

    # `render_latest()` 会自动从磁盘读取文件。
    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2, ensure_ascii=False))

    import os
    os.remove("main.py")

asyncio.run(example_2())
```
**它为何强大:** 即便文件在 Agent 运行期间被修改，`messages.render_latest()` 始终能获取到最新的文件内容。它自动处理了文件的读取、格式化和注入。

### 场景 3: 轻松的上下文重构

需要将一段上下文（如文件内容）从系统消息移动到用户消息？传统方法是字符串操作的噩梦，尤其是在处理多模态内容时。使用 Architext，只需两行代码。

```python
import json
import asyncio
from architext import Messages, UserMessage, Files, SystemMessage, Images

async def example_3():
    with open("main.py", "w", encoding="utf-8") as f: f.write("print('你好')")
    with open("image.png", "w", encoding="utf-8") as f: f.write("dummy")

    messages = Messages(
        SystemMessage("代码:", Files("main.py", name="code_files")),
        UserMessage("hi", Images("image.png"))
    )

    print("--- 移动前 ---")
    print(json.dumps(await messages.render_latest(), indent=2, ensure_ascii=False))

    # 将整个 Files 块移动到用户消息
    files_provider = messages.pop("code_files")
    messages[1].append(files_provider) # 追加到末尾

    print("\n--- 移动后 ---")
    print(json.dumps(await messages.render_latest(), indent=2, ensure_ascii=False))

    # 添加到开头也同样简单: messages[1] = files_provider + messages[1]

    import os
    os.remove("main.py")
    os.remove("image.png")

asyncio.run(example_3())
```
**它为何强大:** `messages.pop("code_files")` 通过名称查找并移除 provider，无论它在何处。Architext 自动处理了多模态消息结构的复杂性，让你专注于逻辑，而非数据格式。

### 场景 4: 用于提示优化的精细可见性控制

为了防止模型输出被截断，一个常见的技巧是在*最后一条*用户提示中添加指令。手动管理这个过程非常复杂。Architext 提供了精确的可见性控制。

```python
import json
import asyncio
from architext import Messages, SystemMessage, Texts, UserMessage, AssistantMessage

async def example_4():
    # 将同一个命名的 provider 添加到多个消息中
    done_marker = Texts("\n\n你的消息 **必须** 以 [done] 结尾。", name="done_marker")

    messages = Messages(
        SystemMessage("你是一个乐于助人的助手。"),
        UserMessage("hi", done_marker),
        AssistantMessage("hello"),
        UserMessage("hi again", done_marker),
    )

    # 1. 隐藏所有 "done_marker" provider 的实例
    messages.provider("done_marker").visible = False
    # 2. 仅使最后一个实例可见
    messages.provider("done_marker")[-1].visible = True

    new_messages = await messages.render_latest()
    print(json.dumps(new_messages, indent=2, ensure_ascii=False))

asyncio.run(example_4())
```
**它为何强大:** 通过命名 provider，你可以对它们进行批量操作。一行代码隐藏所有实例，另一行代码选择性地重新启用你需要的那个。这是一个强大的模式，可用于条件化提示、A/B 测试或在长对话中管理系统指令。

## 🤝 贡献 (Contributing)

上下文工程是一个激动人心的新领域。我们欢迎任何形式的贡献，共同探索构建更智能、更高效的 AI Agent。无论是报告 Bug、提出新功能，还是提交代码，您的参与都至关重要。

## 📄 许可证 (License)

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。
