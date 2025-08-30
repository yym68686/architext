import pickle
import base64
import asyncio
import logging
import hashlib
import mimetypes
import uuid
import threading
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable

# A wrapper to manage multiple providers with the same name
class ProviderGroup:
    """A container for multiple providers that share the same name, allowing for bulk operations."""
    def __init__(self, providers: List['ContextProvider']):
        self._providers = providers
    def __getitem__(self, key: int) -> 'ContextProvider':
        """Allows accessing providers by index, e.g., group[-1]."""
        return self._providers[key]
    def __iter__(self):
        """Allows iterating over the providers."""
        return iter(self._providers)
    def __len__(self) -> int:
        """Returns the number of providers in the group."""
        return len(self._providers)
    @property
    def visible(self) -> List[bool]:
        """Gets the visibility of all providers in the group."""
        return [p.visible for p in self._providers]
    @visible.setter
    def visible(self, value: bool):
        """Sets the visibility for all providers in the group."""
        for p in self._providers:
            p.visible = value

# Global, thread-safe registry for providers created within f-strings
_fstring_provider_registry = {}
_registry_lock = threading.Lock()

def _register_provider(provider: 'ContextProvider') -> str:
    """Registers a provider and returns a unique placeholder."""
    with _registry_lock:
        provider_id = f"__provider_placeholder_{uuid.uuid4().hex}__"
        _fstring_provider_registry[provider_id] = provider
        return provider_id

def _retrieve_provider(placeholder: str) -> Optional['ContextProvider']:
    """Retrieves a provider from the registry."""
    with _registry_lock:
        return _fstring_provider_registry.pop(placeholder, None)

# 1. 核心数据结构: ContentBlock
@dataclass
class ContentBlock:
    name: str
    content: str

# 2. 上下文提供者 (带缓存)
class ContextProvider(ABC):
    def __init__(self, name: str):
        self.name = name
        self._cached_content: Optional[str] = None
        self._is_stale: bool = True
        self._visible: bool = True

    def __str__(self):
        # This allows the object to be captured when used inside an f-string.
        return _register_provider(self)

    def mark_stale(self): self._is_stale = True

    @property
    def visible(self) -> bool:
        """Gets the visibility of the provider."""
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        """Sets the visibility of the provider."""
        if self._visible != value:
            self._visible = value
            # Content needs to be re-evaluated, but the source data hasn't changed,
            # so just marking it stale is enough for the renderer to reconsider it.
            self.mark_stale()
    async def refresh(self):
        if self._is_stale:
            self._cached_content = await self.render()
            self._is_stale = False
    @abstractmethod
    async def render(self) -> Optional[str]: raise NotImplementedError
    @abstractmethod
    def update(self, *args, **kwargs): raise NotImplementedError
    def get_content_block(self) -> Optional[ContentBlock]:
        if self.visible and self._cached_content is not None:
            return ContentBlock(self.name, self._cached_content)
        return None

class Texts(ContextProvider):
    def __init__(self, text: Optional[Union[str, Callable[[], str]]] = None, name: Optional[str] = None):
        if text is None and name is None:
            raise ValueError("Either 'text' or 'name' must be provided.")

        # Ensure that non-callable inputs are treated as strings
        if not callable(text):
            self._text = str(text) if text is not None else None
        else:
            self._text = text

        self._is_dynamic = callable(self._text)

        if name is None:
            if self._is_dynamic:
                import uuid
                _name = f"dynamic_text_{uuid.uuid4().hex[:8]}"
            else:
                # Handle the case where text is None during initialization
                h = hashlib.sha1(self._text.encode() if self._text else b'').hexdigest()
                _name = f"text_{h[:8]}"
        else:
            _name = name
        super().__init__(_name)

    async def refresh(self):
        if self._is_dynamic:
            self._is_stale = True
        await super().refresh()

    def update(self, text: Union[str, Callable[[], str]]):
        self._text = text
        self._is_dynamic = callable(self._text)
        self.mark_stale()

    @property
    def content(self) -> Optional[str]:
        """
        Synchronously retrieves the raw text content as a property.
        If the content is dynamic (a callable), it executes the callable.
        """
        if self._is_dynamic:
            # Ensure dynamic content returns a string, even if empty
            result = self._text()
            return result if result is not None else ""
        # Ensure static content returns a string, even if empty
        return self._text if self._text is not None else ""

    async def render(self) -> Optional[str]:
        return self.content

    def __eq__(self, other):
        if not isinstance(other, Texts):
            return NotImplemented
        # If either object is dynamic, they are only equal if they are the exact same object.
        if self._is_dynamic or (hasattr(other, '_is_dynamic') and other._is_dynamic):
            return self is other
        # For static content, compare the actual content.
        return self.content == other.content

class Tools(ContextProvider):
    def __init__(self, tools_json: Optional[List[Dict]] = None, name: str = "tools"):
        super().__init__(name)
        self._tools_json = tools_json or []
    def update(self, tools_json: List[Dict]):
        self._tools_json = tools_json
        self.mark_stale()
    async def render(self) -> Optional[str]:
        if not self._tools_json:
            return None
        return f"<tools>{str(self._tools_json)}</tools>"

    def __eq__(self, other):
        if not isinstance(other, Tools):
            return NotImplemented
        return self._tools_json == other._tools_json

class Files(ContextProvider):
    def __init__(self, *paths: Union[str, List[str]], name: str = "files"):
        super().__init__(name)
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

    async def refresh(self):
        """
        Overrides the default refresh behavior. It synchronizes the content of
        all tracked files with the file system. If a file is not found, its
        content is updated to reflect the error.
        """
        is_changed = False
        for path in list(self._files.keys()):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    new_content = f.read()
                if self._files.get(path) != new_content:
                    self._files[path] = new_content
                    is_changed = True
            except FileNotFoundError:
                error_msg = f"[Error: File not found at path '{path}']"
                if self._files.get(path) != error_msg:
                    self._files[path] = error_msg
                    is_changed = True
            except Exception as e:
                error_msg = f"[Error: Could not read file at path '{path}': {e}]"
                if self._files.get(path) != error_msg:
                    self._files[path] = error_msg
                    is_changed = True

        if is_changed:
            self.mark_stale()

        await super().refresh()

    def update(self, path: str, content: Optional[str] = None):
        """
        Updates a single file. If content is provided, it updates the file in
        memory. If content is None, it reads the file from disk.
        """
        if content is not None:
            self._files[path] = content
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._files[path] = f.read()
            except FileNotFoundError:
                logging.error(f"File not found for update: {path}.")
                self._files[path] = f"[Error: File not found at path '{path}']"
            except Exception as e:
                logging.error(f"Error reading file for update {path}: {e}.")
                self._files[path] = f"[Error: Could not read file at path '{path}': {e}]"
        self.mark_stale()
    async def render(self) -> str:
        if not self._files: return None
        return "<latest_file_content>" + "\n".join([f"<file><file_path>{p}</file_path><file_content>{c}</file_content></file>" for p, c in self._files.items()]) + "\n</latest_file_content>"

    def __eq__(self, other):
        if not isinstance(other, Files):
            return NotImplemented
        return self._files == other._files

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

    def __eq__(self, other):
        if not isinstance(other, Images):
            return NotImplemented
        return self.url == other.url

# 3. 消息类 (已合并 MessageContent)
class Message(ABC):
    def __init__(self, role: str, *initial_items: Union[ContextProvider, str, list]):
        self.role = role
        processed_items = []
        for item in initial_items:
            if isinstance(item, str):
                # Check if the string contains placeholders from f-string rendering
                import re
                placeholder_pattern = re.compile(r'(__provider_placeholder_[a-f0-9]{32}__)')
                parts = placeholder_pattern.split(item)

                if len(parts) > 1: # Placeholders were found
                    for part in parts:
                        if not part: continue
                        if placeholder_pattern.match(part):
                            provider = _retrieve_provider(part)
                            if provider:
                                processed_items.append(provider)
                        else:
                            processed_items.append(Texts(text=part))
                else: # No placeholders, just a regular string
                    processed_items.append(Texts(text=item))

            elif isinstance(item, Message):
                processed_items.extend(item.provider())
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

    @property
    def content(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """
        Renders the message content.
        For simple text messages, returns a string.
        For multimodal messages, returns a list of content blocks.
        """
        rendered_dict = self.to_dict()
        return rendered_dict.get('content') if rendered_dict else None

    def _render_content(self) -> str:
        final_parts = []
        for item in self._items:
            block = item.get_content_block()
            if block and block.content is not None:
                final_parts.append(block.content)

        return "".join(final_parts)

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

    def provider(self, name: Optional[str] = None) -> Optional[Union[ContextProvider, ProviderGroup, List[ContextProvider]]]:
        if name is None:
            return self._items

        named_providers = [p for p in self._items if hasattr(p, 'name') and p.name == name]

        if not named_providers:
            return None
        if len(named_providers) == 1:
            return named_providers[0]
        return ProviderGroup(named_providers)

    def __add__(self, other):
        if isinstance(other, str):
            new_items = self._items + [Texts(text=other)]
            return type(self)(*new_items)
        if isinstance(other, Message):
            new_items = self._items + other.provider()
            return type(self)(*new_items)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            new_items = [Texts(text=other)] + self._items
            return type(self)(*new_items)
        if isinstance(other, Message):
            new_items = other.provider() + self._items
            return type(self)(*new_items)
        return NotImplemented

    def __getitem__(self, key: Union[str, int]) -> Any:
        """
        使得 Message 对象支持字典风格的访问 (e.g., message['content'])
        和列表风格的索引访问 (e.g., message[-1])。
        """
        if isinstance(key, str):
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
        elif isinstance(key, int):
            return self._items[key]
        else:
            raise TypeError(f"Message indices must be integers or strings, not {type(key).__name__}")

    def __len__(self) -> int:
        """返回消息中 provider 的数量。"""
        return len(self._items)

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
        # We pass a Texts provider to the parent so it can be rendered,
        # but the primary way to access content for ToolResults is via its dict representation.
        super().__init__("tool", Texts(text=content))
        self.tool_call_id = tool_call_id
        self._content = content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self._content
        }

# 4. 顶层容器: Messages
class Messages:
    def __init__(self, *initial_messages: Message):
        from typing import Tuple
        self._messages: List[Message] = []
        self._providers_index: Dict[str, List[Tuple[ContextProvider, Message]]] = {}
        if initial_messages:
            for msg in initial_messages:
                self.append(msg)

    def _notify_provider_added(self, provider: ContextProvider, message: Message):
        if provider.name not in self._providers_index:
            self._providers_index[provider.name] = []
        self._providers_index[provider.name].append((provider, message))

    def _notify_provider_removed(self, provider: ContextProvider):
        if provider.name in self._providers_index:
            # Create a new list excluding the provider to be removed.
            # Comparing by object identity (`is`) is crucial here.
            providers_list = self._providers_index[provider.name]
            new_list = [(p, m) for p, m in providers_list if p is not provider]

            if not new_list:
                # If the list becomes empty, remove the key from the dictionary.
                del self._providers_index[provider.name]
            else:
                # Otherwise, update the dictionary with the new list.
                self._providers_index[provider.name] = new_list

    def provider(self, name: str) -> Optional[Union[ContextProvider, ProviderGroup]]:
        indexed_list = self._providers_index.get(name)
        if not indexed_list:
            return None

        providers = [p for p, m in indexed_list]
        if len(providers) == 1:
            return providers[0]
        else:
            return ProviderGroup(providers)

    def pop(self, key: Optional[Union[str, int]] = None) -> Union[Optional[ContextProvider], Optional[Message]]:
        # If no key is provided, pop the last message.
        if key is None:
            key = len(self._messages) - 1

        if isinstance(key, str):
            indexed_list = self._providers_index.get(key)
            if not indexed_list:
                return None
            # Pop the first one found, which is consistent with how pop usually works
            _provider, parent_message = indexed_list[0]
            # The actual removal from _providers_index happens in _notify_provider_removed
            # which is called by message.pop()
            return parent_message.pop(key)
        elif isinstance(key, int):
            try:
                if key < 0: # Handle negative indices like -1
                    key += len(self._messages)
                if not (0 <= key < len(self._messages)):
                    return None
                popped_message = self._messages.pop(key)
                popped_message._parent_messages = None
                for provider in popped_message.provider():
                    self._notify_provider_removed(provider)
                return popped_message
            except IndexError:
                return None

        return None

    async def refresh(self):
        tasks = []
        for provider_list in self._providers_index.values():
            for provider, _ in provider_list:
                tasks.append(provider.refresh())
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
            for provider in message.provider():
                last_message.append(provider)
        else:
            message._parent_messages = self
            self._messages.append(message)
            for p in message.provider():
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

    def __getitem__(self, index: Union[int, slice]) -> Union[Message, 'Messages']:
        if isinstance(index, slice):
            return Messages(*self._messages[index])
        return self._messages[index]

    def __setitem__(self, index: slice, value: 'Messages'):
        if not isinstance(index, slice) or not isinstance(value, Messages):
            raise TypeError("Unsupported operand type(s) for slice assignment")

        # Basic slice assignment logic.
        # A more robust implementation would handle step and negative indices.
        start, stop, step = index.indices(len(self._messages))

        if step != 1:
            raise ValueError("Slice assignment with step is not supported.")

        # Remove old providers from the index
        for i in range(start, stop):
            for provider in self._messages[i].provider():
                self._notify_provider_removed(provider)

        # Replace the slice in the list
        self._messages[start:stop] = value._messages

        # Add new providers to the index and set parent
        for msg in value:
            msg._parent_messages = self
            for provider in msg.provider():
                self._notify_provider_added(provider, msg)

    def __len__(self) -> int: return len(self._messages)
    def __iter__(self): return iter(self._messages)
