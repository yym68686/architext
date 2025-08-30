import pickle
import base64
import asyncio
import logging
import hashlib
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
            self._cached_content = await self.render()
            self._is_stale = False
    @abstractmethod
    async def render(self) -> Optional[str]: raise NotImplementedError
    @abstractmethod
    def update(self, *args, **kwargs): raise NotImplementedError
    def get_content_block(self) -> Optional[ContentBlock]:
        if self._cached_content is not None: return ContentBlock(self.name, self._cached_content)
        return None

class Texts(ContextProvider):
    def __init__(self, text: str, name: Optional[str] = None):
        self._text = text
        if name is None:
            h = hashlib.sha1(self._text.encode()).hexdigest()
            _name = f"text_{h[:8]}"
        else:
            _name = name
        super().__init__(_name)

    def update(self, text: str):
        self._text = text
        self.mark_stale()

    async def render(self) -> str: return self._text

class Tools(ContextProvider):
    def __init__(self, tools_json: List[Dict]): super().__init__("tools"); self._tools_json = tools_json
    def update(self, tools_json: List[Dict]):
        self._tools_json = tools_json
        self.mark_stale()
    async def render(self) -> str: return f"<tools>{str(self._tools_json)}</tools>"

class Files(ContextProvider):
    def __init__(self, *paths: Union[str, List[str]]):
        super().__init__("files")
        self._files: Dict[str, str] = {}

        file_paths: List[str] = []
        if paths:
            # Handle the case where the first argument is a list of paths, e.g., Files(['a', 'b'])
            if len(paths) == 1 and isinstance(paths[0], list):
                file_paths.extend(paths[0])
            # Handle the case where arguments are individual string paths, e.g., Files('a', 'b')
            else:
                file_paths.extend(paths)

        if file_paths:
            for path in file_paths:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._files[path] = f.read()
                except FileNotFoundError:
                    logging.warning(f"File not found during initialization: {path}. Skipping.")
                except Exception as e:
                    logging.error(f"Error reading file {path} during initialization: {e}")

    def reload(self, path: Optional[str] = None) -> bool:
        """
        Reloads file contents from disk.

        If a path is provided, it reloads that specific file.
        If no path is provided, it reloads all files currently tracked by the provider.

        Args:
            path (Optional[str]): The path to the file to reload. Defaults to None.

        Returns:
            bool: True if all requested reloads were successful, False otherwise.
        """
        paths_to_reload = []
        if path:
            if path in self._files:
                paths_to_reload.append(path)
            else:
                logging.warning(f"Path '{path}' not tracked by this Files provider. Cannot reload.")
                return False
        else:
            paths_to_reload = list(self._files.keys())

        if not paths_to_reload:
            logging.info("No files to reload.")
            return True

        success = True
        for p in paths_to_reload:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    self._files[p] = f.read()
                logging.info(f"Successfully reloaded file: {p}")
            except FileNotFoundError:
                logging.error(f"File not found during reload: {p}. Keeping stale content.")
                success = False
            except Exception as e:
                logging.error(f"Error reloading file {p}: {e}. Keeping stale content.")
                success = False

        if success:
            self.mark_stale()

        return success

    def update(self, path: str, content: str): self._files[path] = content; self.mark_stale()
    async def render(self) -> str:
        if not self._files: return None
        return "<latest_file_content>" + "\n".join([f"<file><file_path>{p}</file_path><file_content>{c}</file_content></file>" for p, c in self._files.items()]) + "\n</latest_file_content>"

class Images(ContextProvider):
    def __init__(self, url: str, name: Optional[str] = None):
        super().__init__(name or url)
        self.url = url
    def update(self, url: str):
        self.url = url
        self.mark_stale()
    async def render(self) -> Optional[str]:
        if self.url.startswith("data:"):
            return self.url
        try:
            with open(self.url, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(self.url)
                if not mime_type: mime_type = "application/octet-stream" # Fallback
                return f"data:{mime_type};base64,{encoded_string}"
        except FileNotFoundError:
            logging.warning(f"Image file not found: {self.url}. Skipping.")
            return None # Or handle error appropriately

# 3. 消息类 (已合并 MessageContent)
class Message(ABC):
    def __init__(self, role: str, *initial_items: Union[ContextProvider, str, list]):
        self.role = role
        processed_items = []
        for item in initial_items:
            if isinstance(item, str):
                processed_items.append(Texts(text=item))
            elif isinstance(item, Message):
                processed_items.extend(item.providers())
            elif isinstance(item, ContextProvider):
                processed_items.append(item)
            elif isinstance(item, list):
                for sub_item in item:
                    if not isinstance(sub_item, dict) or 'type' not in sub_item:
                        raise ValueError("List items must be dicts with a 'type' key.")

                    item_type = sub_item['type']
                    if item_type == 'text':
                        processed_items.append(Texts(text=sub_item.get('text', '')))
                    elif item_type == 'image_url':
                        image_url = sub_item.get('image_url', {}).get('url')
                        if image_url:
                            processed_items.append(Images(url=image_url))
                    else:
                        raise ValueError(f"Unsupported item type in list: {item_type}")
            else:
                raise TypeError(f"Unsupported item type: {type(item)}. Must be str, ContextProvider, or list.")
        self._items: List[ContextProvider] = processed_items
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

    def __add__(self, other):
        if isinstance(other, str):
            new_items = self._items + [Texts(text=other)]
            return type(self)(*new_items)
        if isinstance(other, Message):
            new_items = self._items + other.providers()
            return type(self)(*new_items)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            new_items = [Texts(text=other)] + self._items
            return type(self)(*new_items)
        if isinstance(other, Message):
            new_items = other.providers() + self._items
            return type(self)(*new_items)
        return NotImplemented

    def __getitem__(self, key: str) -> Any:
        """
        使得 Message 对象支持字典风格的访问 (e.g., message['content'])。
        """
        if key == 'role':
            return self.role
        elif key == 'content':
            # 直接调用 to_dict 并提取 'content'，确保逻辑一致
            rendered_dict = self.to_dict()
            return rendered_dict.get('content') if rendered_dict else None
        # 对于 tool_calls 等特殊属性，也通过 to_dict 获取
        elif hasattr(self, key):
            rendered_dict = self.to_dict()
            if rendered_dict and key in rendered_dict:
                return rendered_dict[key]

        # 如果在对象本身或其 to_dict() 中都找不到，则引发 KeyError
        if hasattr(self, key):
             return getattr(self, key)
        raise KeyError(f"'{key}'")

    def __repr__(self): return f"Message(role='{self.role}', items={[i.name for i in self._items]})"
    def __bool__(self) -> bool:
        return bool(self._items)
    def get(self, key: str, default: Any = None) -> Any:
        """提供类似字典的 .get() 方法来访问属性。"""
        return getattr(self, key, default)
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

class RoleMessage:
    """A factory class that creates a specific message type based on the role."""
    def __new__(cls, role: str, *items):
        if role == 'system':
            return SystemMessage(*items)
        elif role == 'user':
            return UserMessage(*items)
        elif role == 'assistant':
            return AssistantMessage(*items)
        else:
            raise ValueError(f"Invalid role: {role}. Must be 'system', 'user', or 'assistant'.")

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

    def pop(self, key: Optional[Union[str, int]] = None) -> Union[Optional[ContextProvider], Optional[Message]]:
        # If no key is provided, pop the last message.
        if key is None:
            key = len(self._messages) - 1

        if isinstance(key, str):
            indexed = self._providers_index.get(key)
            if not indexed:
                return None
            _provider, parent_message = indexed
            return parent_message.pop(key)
        elif isinstance(key, int):
            try:
                if key < 0: # Handle negative indices like -1
                    key += len(self._messages)
                if not (0 <= key < len(self._messages)):
                    return None
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
