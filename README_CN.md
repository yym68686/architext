
# Architext

[English](./README.md) | [中文](./README_CN.md)

[![PyPI version](https://badge.fury.io/py/architext.svg)](https://badge.fury.io/py/architext)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**Architext: The Context Engineering framework for building smarter, more reliable AI Agents.**

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

*   **将上下文模块化 (Modularize Context)**: 每个信息源（如文件内容、工具列表）都是一个独立的 `ContextProvider`，负责生产标准化的 `ContentBlock`。
*   **视消息为可变结构 (Messages as Mutable Structures)**: 消息不再是静态的文本，而是一个可被实时操作的对象容器。您可以像操作精密组件一样，对其内部的内容块进行精确的 `pop`, `insert`, `append`。
*   **以架构师的思维构建提示 (Think like an Architect)**: 您可以像设计软件架构一样，清晰地布局 `SystemMessage` 和 `UserMessage` 的结构，并通过统一的接口动态调整，以应对不同的任务场景。

## 🚀 核心特性 (Features)

*   **面向对象的上下文建模**: 将 `SystemMessage`, `UserMessage` 等视为可操作的一等公民。
*   **原子化的内容块 (`ContentBlock`)**: 将上下文分解为可独立操作和移动的最小单元。
*   **列表式动态操作**: 通过 `pop()`, `insert()` 等方法，实现对上下文内容的实时、精确控制。
*   **提供者驱动架构**: 通过可扩展的 `ContextProvider` 体系，轻松接入任何数据源。
*   **智能缓存与按需刷新**: 内置高效的缓存机制，仅在数据源变化时才刷新，显著提升性能。
*   **统一的穿透式接口**: 通过顶层 `Messages` 对象，直接访问和控制任何底层的 `ContextProvider`，实现状态的集中管理。

## 📦 安装 (Installation)

```bash
pip install architext
```

## 🚀 快速上手：一次上下文工程实践

`Architext` 的核心在于其直观、灵活的 API。下面通过一系列独立的示例，展示如何利用它进行高效的上下文工程。

### 示例 1: 基础布局与首次渲染

这是最基础的用法。我们声明式地构建一个包含 `System` 和 `User` 消息的对话结构。

```python
# --- 示例 1: 基础布局 ---
import asyncio
from architext import Messages, SystemMessage, UserMessage, Texts, Tools

async def example_1():
    # 1. 定义你的上下文提供者
    tools_provider = Tools(tools_json=[{"name": "run_test"}])
    system_prompt = Texts("system_prompt", "你是一个专业的AI代码审查员。")

    # 2. 声明式地构建消息列表
    messages = Messages(
        SystemMessage(system_prompt, tools_provider),
        UserMessage(Texts("user_input", "请帮我审查以下Python代码。"))
    )

    # 3. 渲染最终的 messages 列表
    print("--- 示例 1: 渲染结果 ---")
    # .render_latest() 会自动刷新并渲染
    for msg in await messages.render_latest():
        print(msg)

asyncio.run(example_1())
```

**预期输出:**
```
--- 示例 1: 渲染结果 ---
{'role': 'system', 'content': '你是一个专业的AI代码审查员。\n\n<tools>[{\'name\': \'run_test\'}]</tools>'}
{'role': 'user', 'content': '请帮我审查以下Python代码。'}
```

---

### 示例 2: 穿透更新与自动刷新

`Architext` 的强大之处在于您可以随时更新底层的上下文源，而系统会在下次渲染时自动、高效地刷新内容。

```python
# --- 示例 2: 穿透更新 ---
import asyncio
from architext import Messages, UserMessage, Files

async def example_2():
    # 1. 初始化一个包含文件提供者的消息
    files_provider = Files()
    messages = Messages(
        UserMessage(files_provider)
    )

    # 2. 此刻文件内容为空，渲染结果为空列表
    print("--- 初始状态 (文件内容为空) ---")
    print(await messages.render_latest())

    # 3. 通过穿透接口更新文件内容
    # 这会自动将 files_provider 标记为“过期”
    print("\n>>> 通过 messages.provider 更新文件...")
    file_instance = messages.provider("files")
    if file_instance:
        file_instance.update("main.py", "def main():\n    pass")

    # 4. 再次渲染，Architext 会自动刷新已过期的 provider
    print("\n--- 更新后再次渲染 ---")
    for msg in await messages.render_latest():
        print(msg)

asyncio.run(example_2())
```

**预期输出:**
```
--- 初始状态 (文件内容为空) ---
[]

>>> 通过 messages.provider 更新文件...

--- 更新后再次渲染 ---
{'role': 'user', 'content': "<files>\n<file path='main.py'>def main():\n    pass...</file>\n</files>"}
```

---

### 示例 3: 动态重构上下文 (`pop` 和 `insert`)

这是**上下文工程**的核心实践。您可以像操作列表一样，动态地将一个内容块从一条消息移动到另一条消息，以适应不同的任务需求。

```python
# --- 示例 3: 动态重构 ---
import asyncio
from architext import Messages, SystemMessage, UserMessage, Texts, Tools

async def example_3():
    # 1. 初始布局：工具在 SystemMessage 中
    tools_provider = Tools(tools_json=[{"name": "run_test"}])
    messages = Messages(
        SystemMessage(tools_provider),
        UserMessage(Texts("user_input", "分析代码并运行测试。"))
    )
    print("--- 初始布局 ---")
    for msg in await messages.render_latest(): print(msg)

    # 2. 运行时决策：为了更强的指令性，将工具上下文移动到用户消息中
    print("\n>>> 重构上下文：将 'tools' 块移动到 UserMessage...")

    # a. 从任何消息中全局弹出 'tools' 提供者
    popped_tools_provider = messages.pop("tools")

    # b. 通过索引精确定位到 UserMessage (messages[1])，并插入它
    if popped_tools_provider:
        messages[1].insert(0, popped_tools_provider)

    # 3. 查看重构后的结果
    print("\n--- 重构后的最终布局 ---")
    # 此处无需刷新，所以我们使用同步的 .render()
    for msg in messages.render(): print(msg)

asyncio.run(example_3())
```

**预期输出:**
```
--- 初始布局 ---
{'role': 'system', 'content': "<tools>[{'name': 'run_test'}]</tools>"}
{'role': 'user', 'content': '分析代码并运行测试。'}

>>> 重构上下文：将 'tools' 块移动到 UserMessage...

--- 重构后的最终布局 ---
{'role': 'system', 'content': ''}
{'role': 'user', 'content': "<tools>[{'name': 'run_test'}]</tools>\n\n分析代码并运行测试。"}
```
*(注意: SystemMessage 的内容变为空，因为它唯一的块被移走了，所以在最终渲染时可能会被过滤掉)*

---

### 示例 4: 多模态上下文 (文本 + 图片)

`Architext` 原生支持多模态上下文的构建，能自动格式化输出以匹配 OpenAI 等主流 API。

```python
# --- 示例 4: 多模态 ---
import asyncio
from architext import Messages, UserMessage, Texts, Images

async def example_4():
    # 为示例创建一个虚拟图片文件
    with open("example_image.png", "w") as f: f.write("dummy")

    messages = Messages(
        UserMessage(
            Texts("prompt", "这张图片里有什么？"),
            Images("image_input", "example_image.png")
        )
    )

    print("--- 多模态渲染结果 ---")
    for msg in await messages.render_latest():
        # 为保持简洁，隐藏冗长的 base64 字符串
        for part in msg['content']:
            if part['type'] == 'image_url':
                part['image_url']['url'] = part['image_url']['url'][:80] + "..."
        print(msg)

    # 清理虚拟文件
    import os
    os.remove("example_image.png")

asyncio.run(example_4())
```

**预期输出:**
```
--- 多模态渲染结果 ---
{'role': 'user', 'content': [{'type': 'text', 'text': '这张图片里有什么？'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,ZHVtbXk=...'}}]}
```

## 🤝 贡献 (Contributing)

上下文工程是一个激动人心的新领域。我们欢迎任何形式的贡献，共同探索构建更智能、更高效的 AI Agent。无论是报告 Bug、提出新功能，还是提交代码，您的参与都至关重要。

## 📄 许可证 (License)

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。
