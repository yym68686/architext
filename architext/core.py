import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

# 1. 核心数据结构: ContentBlock
class ContentBlock:
    def __init__(self, name: str, content: str, provider: Optional['ContextProvider'] = None):
        self.name = name; self.content = content; self.provider = provider
    def __repr__(self): return f"Block(name='{self.name}')"

# 2. 上下文提供者 (带缓存)
class ContextProvider(ABC):
    def __init__(self, name: str):
        self.name = name; self._cached_content: Optional[str] = None; self._is_stale: bool = True
    def mark_stale(self): self._is_stale = True
    async def _refresh(self):
        if self._is_stale:
            # 注意：我们将在这个方法上使用 mock，所以实际的 print 不再重要
            # print(f"信息: 正在为上下文提供者 '{self.name}' 刷新内容...")
            self._cached_content = await self._fetch_content()
            self._is_stale = False
        # else:
            # print(f"调试: 上下文提供者 '{self.name}' 正在使用缓存内容。")
    @abstractmethod
    async def _fetch_content(self) -> Optional[str]: raise NotImplementedError
    async def render(self) -> Optional[ContentBlock]:
        await self._refresh()
        if self._cached_content is not None: return ContentBlock(self.name, self._cached_content, self)
        return None

class Texts(ContextProvider):
    def __init__(self, name: str, text: str): super().__init__(name); self._text = text
    async def _fetch_content(self) -> str: return self._text

class Tools(ContextProvider):
    def __init__(self, tools_json: List[Dict]): super().__init__("tools"); self._tools_json = tools_json
    async def _fetch_content(self) -> str: return f"<tools>{str(self._tools_json)}</tools>"

class Files(ContextProvider):
    def __init__(self): super().__init__("files"); self._files: Dict[str, str] = {}
    def update(self, path: str, content: str): self._files[path] = content; self.mark_stale()
    async def _fetch_content(self) -> str:
        if not self._files: return None
        return "<files>\n" + "\n".join([f"<file path='{p}'>{c[:50]}...</file>" for p, c in self._files.items()]) + "\n</files>"

# 3. 消息内容类与消息类
class MessageContent:
    def __init__(self, items: List[ContextProvider]): self._items: List[ContextProvider] = items
    async def render(self) -> str:
        tasks = [item.render() for item in self._items]
        blocks = await asyncio.gather(*tasks)
        return "\n\n".join(b.content for b in blocks if b and b.content)
    def pop(self, name: str) -> Optional[ContextProvider]:
        for i, item in enumerate(self._items):
            if hasattr(item, 'name') and item.name == name: return self._items.pop(i)
        return None
    def insert(self, index: int, item: ContextProvider): self._items.insert(index, item)
    def append(self, item: ContextProvider): self._items.append(item)
    def providers(self) -> List[ContextProvider]: return self._items
    def __repr__(self): return f"Content(items={[item.name for item in self._items if hasattr(item, 'name')]})"

class Message(ABC):
    def __init__(self, role: str, *initial_items: ContextProvider): self.role = role; self.content = MessageContent(list(initial_items))
    async def to_dict(self) -> Optional[Dict[str, Any]]:
        rendered_content = await self.content.render()
        if not rendered_content: return None
        return {"role": self.role, "content": rendered_content}

class SystemMessage(Message):
    def __init__(self, *items): super().__init__("system", *items)
class UserMessage(Message):
    def __init__(self, *items): super().__init__("user", *items)

# 4. 上下文构建器 (内部使用)
class ContextBuilder:
    def __init__(self, providers: List[ContextProvider]): self.providers = {p.name: p for p in providers}
    def get_provider(self, name: str) -> Optional[ContextProvider]: return self.providers.get(name)

# 5. 顶层容器: Messages
class Messages:
    def __init__(self, *initial_messages: Message):
        self._messages: List[Message] = list(initial_messages)
        all_providers = []
        for msg in self._messages: all_providers.extend(msg.content.providers())
        self._context_builder = ContextBuilder(all_providers)
    def provider(self, name: str) -> Optional[ContextProvider]: return self._context_builder.get_provider(name)
    def pop(self, name: str) -> Optional[ContextProvider]:
        for message in self._messages:
            popped_item = message.content.pop(name)
            if popped_item: return popped_item
        return None
    async def render(self) -> List[Dict[str, Any]]:
        tasks = [msg.to_dict() for msg in self._messages]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res]
    def append(self, message: Message):
        self._messages.append(message)
        for p in message.content.providers():
            if p.name not in self._context_builder.providers: self._context_builder.providers[p.name] = p
    def __getitem__(self, index: int) -> Message: return self._messages[index]
    def __len__(self) -> int: return len(self._messages)
    def __iter__(self): return iter(self._messages)


# ==============================================================================
# 6. 演示
# ==============================================================================
async def run_demo():
    # --- 1. 初始化提供者 ---
    system_prompt_provider = Texts("system_prompt", "你是一个AI助手。")
    tools_provider = Tools(tools_json=[{"name": "read_file"}])
    files_provider = Files()

    # --- 2. 演示新功能：优雅地构建 Messages ---
    print("\n>>> 场景 A: 使用新的、优雅的构造函数直接初始化 Messages")
    messages = Messages(
        SystemMessage(system_prompt_provider, tools_provider),
        UserMessage(files_provider, Texts("user_input", "这是我的初始问题。"))
    )

    print("\n--- 渲染后的初始 Messages (首次渲染，全部刷新) ---")
    for msg_dict in await messages.render(): print(msg_dict)
    print("-" * 40)

    # --- 3. 演示穿透更新 ---
    print("\n>>> 场景 B: 穿透更新 File Provider，渲染时自动刷新")

    # 直接通过 messages 对象穿透访问并更新 files provider
    files_provider_instance = messages.provider("files")
    if isinstance(files_provider_instance, Files):
        files_provider_instance.update("file1.py", "这是新的文件内容！")

    print("\n--- 再次渲染 Messages (只有文件提供者会刷新) ---")
    for msg_dict in await messages.render(): print(msg_dict)
    print("-" * 40)

    # --- 4. 演示全局 Pop 和通过索引 Insert ---
    print("\n>>> 场景 C: 全局 Pop 工具提供者，并 Insert 到 UserMessage 中")

    # a. 全局弹出 'tools' Provider
    popped_tools_provider = messages.pop("tools")

    # b. 将弹出的 Provider 插入到第一个 UserMessage (索引为1) 的开头
    if popped_tools_provider:
        # 通过索引精确定位
        messages[1].content.insert(0, popped_tools_provider)
        print(f"\n已成功将 '{popped_tools_provider.name}' 提供者移动到用户消息。")

    print("\n--- Pop 和 Insert 后渲染的 Messages (验证移动效果) ---")
    for msg_dict in await messages.render(): print(msg_dict)
    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(run_demo())
