# Architext

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

下面的例子将演示如何使用 `Architext` 对 LLM 的输入进行一次动态的“上下文重构”。

```python
import asyncio
# 假设这些类位于您的 architext 库中
from architext import Messages, SystemMessage, UserMessage, Texts, Files, Tools

async def main():
    # 1. 定义你的上下文源 (Context Providers)
    system_prompt = Texts("system_prompt", "你是一个代码分析助手。")
    tools = Tools(tools_json=[{"name": "run_test"}])
    files = Files()

    # 2. 声明式地构建初始消息布局
    messages = Messages(
        SystemMessage(system_prompt, tools), # 系统消息包含基础提示和工具
        UserMessage(files) # 用户消息最初只包含文件内容
    )
    # 为文件提供者注入初始内容
    files.update("main.py", "def hello(): print('world')")

    # 3. 初始状态：让 LLM 分析文件
    print("--- 初始上下文结构 ---")
    messages.append(UserMessage(Texts("prompt", "分析这个文件。")))
    for msg in await messages.render():
        print(msg)

    # 假设 LLM 回复需要运行测试...

    # 4. 上下文重构：为了执行测试，将工具移动到用户消息中，以获得更强的指令性
    print("\n--- 上下文重构：移动'tools'块 ---")

    # a. 从 SystemMessage 全局弹出 'tools' provider
    tools_provider = messages.pop("tools")

    # b. 创建一个新的 UserMessage，并将弹出的 provider 和新的指令插入其中
    if tools_provider:
        new_user_message = UserMessage(
            tools_provider, # 将工具上下文放在前面
            Texts("prompt", "现在，请使用以上工具运行测试。")
        )
        messages.append(new_user_message)

    # 5. 渲染重构后的最终上下文
    print("\n--- 重构后的最终上下文 ---")
    for msg in await messages.render():
        print(msg)

if __name__ == "__main__":
    asyncio.run(main())
```
在这个例子中，我们动态地将 `tools` 上下文从 `SystemMessage` 移动到了一个新的 `UserMessage` 中，这种精确到消息级别的实时重构正是**上下文工程**的核心实践。

## 🤝 贡献 (Contributing)

上下文工程是一个激动人心的新领域。我们欢迎任何形式的贡献，共同探索构建更智能、更高效的 AI Agent。无论是报告 Bug、提出新功能，还是提交代码，您的参与都至关重要。

## 📄 许可证 (License)

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。