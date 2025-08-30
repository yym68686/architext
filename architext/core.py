import pickle
import base64
import asyncio
import logging
import mimetypes
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

# 1. 核心数据结构: ContentBlock
@dataclass
class ContentBlock:
    name: str
    content: str

# 2. 上下文提供者 (带缓存)
class ContextProvider(ABC):
    def __init__(self, name: str):
        self.name = name; self._cached_content: Optional[str] = None; self._is_stale: bool = True
    def mark_stale(self): self._is_stale = True
    async def refresh(self):
        if self._is_stale:
            self._cached_content = await self._fetch_content()
            self._is_stale = False
    @abstractmethod
    async def _fetch_content(self) -> Optional[str]: raise NotImplementedError
    @abstractmethod
    def update(self, *args, **kwargs): raise NotImplementedError
    def get_content_block(self) -> Optional[ContentBlock]:
        if self._cached_content is not None: return ContentBlock(self.name, self._cached_content)
        return None

class Texts(ContextProvider):
    def __init__(self, name: str, text: str): super().__init__(name); self._text = text
    def update(self, text: str):
        self._text = text
        self.mark_stale()
    async def _fetch_content(self) -> str: return self._text

class Tools(ContextProvider):
    def __init__(self, tools_json: List[Dict]): super().__init__("tools"); self._tools_json = tools_json
    def update(self, tools_json: List[Dict]):
        self._tools_json = tools_json
        self.mark_stale()
    async def _fetch_content(self) -> str: return f"<tools>{str(self._tools_json)}</tools>"

class Files(ContextProvider):
    def __init__(self): super().__init__("files"); self._files: Dict[str, str] = {}
    def update(self, path: str, content: str): self._files[path] = content; self.mark_stale()
    async def _fetch_content(self) -> str:
        if not self._files: return None
        return "<files>\n" + "\n".join([f"<file path='{p}'>{c}...</file>" for p, c in self._files.items()]) + "\n</files>"

class Images(ContextProvider):
    def __init__(self, image_path: str, name: Optional[str] = None):
        super().__init__(name or image_path)
        self.image_path = image_path
    def update(self, image_path: str):
        self.image_path = image_path
        self.mark_stale()
    async def _fetch_content(self) -> Optional[str]:
        try:
            with open(self.image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(self.image_path)
                if not mime_type: mime_type = "application/octet-stream" # Fallback
                return f"data:{mime_type};base64,{encoded_string}"
        except FileNotFoundError:
            logging.warning(f"Image file not found: {self.image_path}. Skipping.")
            return None # Or handle error appropriately

# 3. 消息类 (已合并 MessageContent)
class Message(ABC):
    def __init__(self, role: str, *initial_items: ContextProvider):
        self.role = role
        self._items: List[ContextProvider] = list(initial_items)
        self._parent_messages: Optional['Messages'] = None

    def _render_content(self) -> str:
        blocks = [item.get_content_block() for item in self._items]
        return "\n\n".join(b.content for b in blocks if b and b.content)

    def pop(self, name: str) -> Optional[ContextProvider]:
        popped_item = None
        for i, item in enumerate(self._items):
            if hasattr(item, 'name') and item.name == name:
                popped_item = self._items.pop(i)
                break
        if popped_item and self._parent_messages:
            self._parent_messages._notify_provider_removed(popped_item)
        return popped_item

    def insert(self, index: int, item: ContextProvider):
        self._items.insert(index, item)
        if self._parent_messages:
            self._parent_messages._notify_provider_added(item, self)

    def append(self, item: ContextProvider):
        self._items.append(item)
        if self._parent_messages:
            self._parent_messages._notify_provider_added(item, self)

    def providers(self) -> List[ContextProvider]: return self._items
    def __repr__(self): return f"Message(role='{self.role}', items={[i.name for i in self._items]})"
    def to_dict(self) -> Optional[Dict[str, Any]]:
        is_multimodal = any(isinstance(p, Images) for p in self._items)

        if not is_multimodal:
            rendered_content = self._render_content()
            if not rendered_content: return None
            return {"role": self.role, "content": rendered_content}
        else:
            content_list = []
            for item in self._items:
                block = item.get_content_block()
                if not block or not block.content: continue
                if isinstance(item, Images):
                    content_list.append({"type": "image_url", "image_url": {"url": block.content}})
                else:
                    content_list.append({"type": "text", "text": block.content})
            if not content_list: return None
            return {"role": self.role, "content": content_list}

class SystemMessage(Message):
    def __init__(self, *items): super().__init__("system", *items)
class UserMessage(Message):
    def __init__(self, *items): super().__init__("user", *items)
class AssistantMessage(Message):
    def __init__(self, *items): super().__init__("assistant", *items)

class ToolCalls(Message):
    """Represents an assistant message that requests tool calls."""
    def __init__(self, tool_calls: List[Any]):
        super().__init__("assistant")
        self.tool_calls = tool_calls

    def to_dict(self) -> Dict[str, Any]:
        # Duck-typing serialization for OpenAI's tool_call objects
        serialized_calls = []
        for tc in self.tool_calls:
            try:
                # Attempt to serialize based on openai-python > 1.0 tool_call structure
                func = tc.function
                serialized_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": { "name": func.name, "arguments": func.arguments }
                })
            except AttributeError:
                if isinstance(tc, dict):
                    serialized_calls.append(tc) # Assume it's already a serializable dict
                else:
                    raise TypeError(f"Unsupported tool_call type: {type(tc)}. It should be an OpenAI tool_call object or a dict.")

        return {
            "role": self.role,
            "tool_calls": serialized_calls,
            "content": None
        }

class ToolResults(Message):
    """Represents a tool message with the result of a single tool call."""
    def __init__(self, tool_call_id: str, content: str):
        super().__init__("tool")
        self.tool_call_id = tool_call_id
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self.content
        }

# 4. 顶层容器: Messages
class Messages:
    def __init__(self, *initial_messages: Message):
        from typing import Tuple
        self._messages: List[Message] = []
        self._providers_index: Dict[str, Tuple[ContextProvider, Message]] = {}
        if initial_messages:
            for msg in initial_messages:
                self.append(msg)

    def _notify_provider_added(self, provider: ContextProvider, message: Message):
        if provider.name not in self._providers_index:
            self._providers_index[provider.name] = (provider, message)

    def _notify_provider_removed(self, provider: ContextProvider):
        if provider.name in self._providers_index:
            del self._providers_index[provider.name]

    def provider(self, name: str) -> Optional[ContextProvider]:
        indexed = self._providers_index.get(name)
        return indexed[0] if indexed else None

    def pop(self, key: Union[str, int]) -> Union[Optional[ContextProvider], Optional[Message]]:
        if isinstance(key, str):
            indexed = self._providers_index.get(key)
            if not indexed:
                return None
            _provider, parent_message = indexed
            return parent_message.pop(key)
        elif isinstance(key, int):
            try:
                popped_message = self._messages.pop(key)
                popped_message._parent_messages = None
                for provider in popped_message.providers():
                    self._notify_provider_removed(provider)
                return popped_message
            except IndexError:
                return None

        return None

    async def refresh(self):
        tasks = [provider.refresh() for provider, _ in self._providers_index.values()]
        await asyncio.gather(*tasks)

    def render(self) -> List[Dict[str, Any]]:
        results = [msg.to_dict() for msg in self._messages]
        return [res for res in results if res]

    async def render_latest(self) -> List[Dict[str, Any]]:
        await self.refresh()
        return self.render()

    def append(self, message: Message):
        if self._messages and self._messages[-1].role == message.role:
            last_message = self._messages[-1]
            for provider in message.providers():
                last_message.append(provider)
        else:
            message._parent_messages = self
            self._messages.append(message)
            for p in message.providers():
                self._notify_provider_added(p, message)

    def save(self, file_path: str):
        """
        Saves the entire Messages object to a file using pickle.
        Warning: Deserializing data with pickle from an untrusted source is insecure.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> Optional['Messages']:
        """
        Loads a Messages object from a file using pickle.
        Returns the loaded object, or None if the file is not found or an error occurs.
        Warning: Only load files from a trusted source.
        """
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logging.warning(f"File not found at {file_path}, returning empty Messages.")
            return cls()
        except (pickle.UnpicklingError, EOFError) as e:
            logging.error(f"Could not deserialize file {file_path}: {e}")
            return cls()

    def __getitem__(self, index: int) -> Message: return self._messages[index]
    def __len__(self) -> int: return len(self._messages)
    def __iter__(self): return iter(self._messages)
