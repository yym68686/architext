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

## 🚀 快速上手：一次上下文工程实践

以下示例按从最独特到基础的顺序，展示了 Architext 最强大的功能。

### 示例 1: F-String 提示构建的魔力 (亮点功能)

忘掉手动拼接字符串。使用您早已熟悉的工具——F-string——以声明式和动态的方式构建提示。

```python
import asyncio
from architext import Messages, UserMessage, Texts, Tools, Files
from datetime import datetime

async def example_1():
    # 定义将嵌入 f-string 的提供者
    os_provider = Texts("MacOS Sonoma", name="os_version")
    tools_provider = Tools([{"name": "read_file"}])
    files_provider = Files(["main.py", "utils.py"])
    time_provider = Texts(lambda: datetime.now().isoformat()) # 动态内容！

    # 为示例创建虚拟文件
    with open("main.py", "w") as f: f.write("print('hello')")
    with open("utils.py", "w") as f: f.write("def helper(): pass")

    # 用一个 f-string 构建完整的消息！
    # Architext 会自动检测并管理嵌入的提供者。
    prompt = f"""
    系统信息:
    - 操作系统: {os_provider}
    - 当前时间: {time_provider}

    可用工具: {tools_provider}

    文件内容:
    {files_provider}

    用户请求:
    根据文件内容，分析这个项目的主要功能是什么？
    """

    messages = Messages(UserMessage(prompt))

    # 渲染完全构建好的消息
    print("--- F-String 渲染结果 ---")
    for msg in await messages.render_latest():
        print(msg['content'])

    # 清理虚拟文件
    import os
    os.remove("main.py")
    os.remove("utils.py")

asyncio.run(example_1())
```

**预期输出:** F-string 会被所有提供者的内容完全解析，包括动态生成的时间戳和文件内容。

---

### 示例 2: 动态上下文重构与可见性控制

根据应用逻辑实时调整上下文。在这里，我们移动一个工具定义，然后一次性隐藏多个“解释”提供者。

```python
import asyncio
from architext import Messages, SystemMessage, UserMessage, Texts, Tools

async def example_2():
    messages = Messages(
        SystemMessage(
            Texts("你是一个AI助手。", name="intro"),
            Tools([{"name": "run_code"}]) # 初始在 SystemMessage 中
        ),
        UserMessage(
            Texts("第一个解释。", name="explanation"),
            Texts("请运行代码。", name="request"),
            Texts("第二个解释。", name="explanation")
        )
    )

    # --- A 部分: 移动提供者 ---
    print(">>> 重构: 为了强调，将 'tools' 移动到 UserMessage...")

    # 1. 从任何位置全局弹出提供者
    tools_provider = messages.pop("tools")
    # 2. 将其插入到指定消息的指定位置
    if tools_provider:
        messages[1].insert(1, tools_provider)

    print("\n--- 移动 'tools' 之后 ---")
    for msg in await messages.render_latest(): print(msg)

    # --- B 部分: 批量隐藏提供者 ---
    print("\n>>> 隐藏所有 'explanation' 提供者...")

    # 1. 获取所有名为 "explanation" 的提供者组
    explanation_group = messages.provider("explanation")
    # 2. 为整个组设置可见性
    explanation_group.visible = False

    print("\n--- 隐藏解释之后 ---")
    for msg in await messages.render_latest(): print(msg)

asyncio.run(example_2())
```

**预期输出:** 您将看到 `<tools>` 块从系统消息移动到用户消息。然后，在最终输出中，“第一个解释”和“第二个解释”的文本将会消失，而其余内容保持不变。

---

### 示例 3: 多模态与工具使用对话

Architext 原生支持多模态交互和工具使用流程所需的复杂消息结构。

```python
import asyncio
from dataclasses import dataclass, field
from architext import Messages, UserMessage, AssistantMessage, Texts, Images, ToolCalls, ToolResults

# 使用 dataclass 模拟来自 OpenAI 等库的 tool_call 对象
@dataclass
class MockFunction:
    name: str
    arguments: str

@dataclass
class MockToolCall:
    id: str
    type: str = "function"
    function: MockFunction = field(default_factory=lambda: MockFunction("", ""))

async def example_3():
    # --- 多模态示例 ---
    with open("dummy_image.png", "w") as f: f.write("dummy")

    multimodal_messages = Messages(
        UserMessage(
            "这张图片里有什么？",
            Images("dummy_image.png")
        )
    )
    print("--- 多模态渲染结果 ---")
    for msg in await multimodal_messages.render_latest(): print(msg)

    # --- 工具使用示例 ---
    # 模拟一个来自模型的 tool call 请求
    tool_call_request = [
        MockToolCall(id="call_123", function=MockFunction(name="add", arguments='{"a": 5, "b": 10}'))
    ]

    tool_use_messages = Messages(
        UserMessage("5 + 10 是多少?"),
        # 代表模型请求调用工具
        ToolCalls(tool_call_request),
        # 代表您返回给模型的结果
        ToolResults(tool_call_id="call_123", content="15"),
        AssistantMessage("它们的和是 15。")
    )
    print("\n--- 工具使用渲染结果 ---")
    for msg in await tool_use_messages.render_latest(): print(msg)

    import os
    os.remove("dummy_image.png")

asyncio.run(example_3())
```

**预期输出:** 两个示例都将渲染成现代 LLM API (如 OpenAI) 所期望的精确字典格式，正确处理多模态消息的列表式内容以及 `tool_calls`/`tool` 角色。

---

### 示例 4: 穿透式更新与自动刷新

从任何地方更新任何上下文片段，Architext 将确保这些更改在下一次渲染时得到反映。

```python
import asyncio
from architext import Messages, UserMessage, Files

async def example_4():
    # 1. 初始化一个 Files 提供者
    messages = Messages(UserMessage(Files(name="code_files")))

    # 2. 初始时，内容为空
    print("--- 初始状态 (未加载文件) ---")
    print(await messages.render_latest())

    # 3. 获取提供者的句柄并更新它
    print("\n>>> 通过 messages.provider('code_files') 更新文件...")
    files_provider = messages.provider("code_files")
    if files_provider:
        # 在内存中为一个新文件更新内容
        files_provider.update("main.py", "def main():\\n    print('Hello')")

    # 4. 再次渲染。Architext 检测到过期的提供者并刷新它。
    print("\n--- 更新后渲染 ---")
    for msg in await messages.render_latest(): print(msg)

asyncio.run(example_4())
```

**预期输出:** 第一次渲染结果将为空。更新后，第二次渲染将正确显示 `main.py` 的内容。

## 🤝 贡献 (Contributing)

上下文工程是一个激动人心的新领域。我们欢迎任何形式的贡献，共同探索构建更智能、更高效的 AI Agent。无论是报告 Bug、提出新功能，还是提交代码，您的参与都至关重要。

## 📄 许可证 (License)

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。
