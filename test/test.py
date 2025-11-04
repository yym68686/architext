import unittest
from unittest.mock import AsyncMock

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from architext import *
from typing import Optional, Union, Callable
# ==============================================================================
# 单元测试部分
# ==============================================================================
class TestContextManagement(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """在每个测试前设置环境"""
        self.system_prompt_provider = Texts("你是一个AI助手。", name="system_prompt")
        self.tools_provider = Tools(tools_json=[{"name": "read_file"}])
        self.files_provider = Files()

    async def test_a_initial_construction_and_render(self):
        """测试优雅的初始化和首次渲染"""
        messages = Messages(
            SystemMessage(self.system_prompt_provider, self.tools_provider),
            UserMessage(self.files_provider, Texts("这是我的初始问题。"))
        )

        self.assertEqual(len(messages), 2)
        rendered = await messages.render_latest()

        self.assertEqual(len(rendered), 2)
        self.assertIn("<tools>", rendered[0]['content'])
        self.assertNotIn("<files>", rendered[1]['content'])

    async def test_b_provider_passthrough_and_refresh(self):
        """测试通过 mock 验证缓存和刷新逻辑"""
        # 使用一个简单的 Texts provider 来测试通用缓存逻辑，避免 Files 的副作用
        text_provider = Texts("initial text")
        text_provider.render = AsyncMock(wraps=text_provider.render)
        messages = Messages(UserMessage(text_provider))

        # 1. 首次刷新
        await messages.refresh()
        self.assertEqual(text_provider.render.call_count, 1)

        # 2. 再次刷新，内容未变，不应再次调用 render
        await messages.refresh()
        self.assertEqual(text_provider.render.call_count, 1)

        # 3. 更新内容，这会标记 provider 为 stale
        text_provider.update("updated text")

        # 4. 再次刷新，现在应该会重新调用 render
        await messages.refresh()
        rendered = messages.render()
        self.assertEqual(text_provider.render.call_count, 2)
        self.assertIn("updated text", rendered[0]['content'])

    async def test_c_global_pop_and_indexed_insert(self):
        """测试全局pop和通过索引insert的功能"""
        messages = Messages(
            SystemMessage(self.system_prompt_provider, self.tools_provider),
            UserMessage(self.files_provider)
        )

        # 验证初始状态
        initial_rendered = await messages.render_latest()
        self.assertTrue(any("<tools>" in msg['content'] for msg in initial_rendered if msg['role'] == 'system'))

        # 全局弹出 'tools' Provider
        popped_tools_provider = messages.pop("tools")
        self.assertIs(popped_tools_provider, self.tools_provider)

        # 验证 pop 后的状态
        rendered_after_pop = messages.render()
        self.assertFalse(any("<tools>" in msg['content'] for msg in rendered_after_pop if msg['role'] == 'system'))

        # 通过索引将弹出的provider插入到UserMessage的开头
        messages[1].insert(0, popped_tools_provider)

        # 验证 insert 后的状态
        rendered_after_insert = messages.render()
        user_message_content = next(msg['content'] for msg in rendered_after_insert if msg['role'] == 'user')
        self.assertTrue(user_message_content.startswith("<tools>"))

    async def test_d_multimodal_rendering(self):
        """测试多模态（文本+图片）渲染"""
        # Create a dummy image file for the test
        dummy_image_path = "test_dummy_image.png"
        with open(dummy_image_path, "w") as f:
            f.write("dummy content")

        messages = Messages(
            UserMessage(
                Texts("Describe the image."),
                Images(url=dummy_image_path) # Test with optional name
            )
        )

        rendered = await messages.render_latest()
        self.assertEqual(len(rendered), 1)

        content = rendered[0]['content']
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        # Check text part
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[0]['text'], 'Describe the image.')

        # Check image part
        self.assertEqual(content[1]['type'], 'image_url')
        self.assertIn('data:image/png;base64,', content[1]['image_url']['url'])

        # Clean up the dummy file
        import os
        os.remove(dummy_image_path)

    async def test_e_multimodal_type_switching(self):
        """测试多模态消息在pop图片后是否能正确回退到字符串渲染"""
        dummy_image_path = "test_dummy_image_2.png"
        with open(dummy_image_path, "w") as f:
            f.write("dummy content")

        messages = Messages(
            UserMessage(
                Texts("Look at this:"),
                Images(url=dummy_image_path, name="image"), # Explicit name for popping
                Texts("Any thoughts?")
            )
        )

        # 1. Initial multimodal render
        rendered_multi = await messages.render_latest()
        content_multi = rendered_multi[0]['content']
        self.assertIsInstance(content_multi, list)
        self.assertEqual(len(content_multi), 3) # prefix, image, suffix

        # 2. Pop the image
        popped_image = messages.pop("image")
        self.assertIsNotNone(popped_image)

        # 3. Render again, should fall back to string content and seamlessly join texts
        rendered_str = messages.render() # No refresh needed
        content_str = rendered_str[0]['content']
        self.assertIsInstance(content_str, str)
        # With the new render logic, adjacent texts are joined with ""
        self.assertEqual(content_str, "Look at this:Any thoughts?")

        # Clean up
        import os
        os.remove(dummy_image_path)

    def test_f_message_merging(self):
        """测试初始化和追加时自动合并消息的功能"""
        # 1. Test merging during initialization
        messages = Messages(
            UserMessage(Texts("Hello,")),
            UserMessage(Texts("world!")),
            SystemMessage(Texts("System prompt.")),
            UserMessage(Texts("How are you?"))
        )
        # Should be merged into: User, System, User
        self.assertEqual(len(messages), 3)
        self.assertEqual(len(messages[0]._items), 2) # First UserMessage has 2 items
        self.assertIn("text_", messages[0]._items[1].name)
        self.assertEqual(messages[1].role, "system")
        self.assertEqual(messages[2].role, "user")

        # 2. Test merging during append
        messages.append(UserMessage(Texts("I am fine.")))
        self.assertEqual(len(messages), 3) # Still 3 messages
        self.assertEqual(len(messages[2]._items), 2) # Last UserMessage now has 2 items
        self.assertIn("text_", messages[2]._items[1].name)

        # 3. Test appending a different role
        messages.append(SystemMessage(Texts("Another prompt.")))
        self.assertEqual(len(messages), 4) # Should not merge
        self.assertEqual(messages[3].role, "system")

    async def test_g_state_inconsistency_on_direct_message_modification(self):
        """
        测试当直接在 Message 对象上执行 pop 操作时，
        顶层 Messages 对象的 _providers_index 是否会产生不一致。
        """
        messages = Messages(
            SystemMessage(self.system_prompt_provider, self.tools_provider),
            UserMessage(self.files_provider)
        )

        # 0. 先刷新一次，确保所有 provider 的 cache 都已填充
        await messages.refresh()

        # 1. 初始状态：'tools' 提供者应该在索引中
        self.assertIsNotNone(messages.provider("tools"), "初始状态下 'tools' 提供者应该能被找到")
        self.assertIs(messages.provider("tools"), self.tools_provider)

        # 2. 直接在子消息对象上执行 pop 操作
        system_message = messages[0]
        popped_provider = system_message.pop("tools")

        # 验证是否真的从 Message 对象中弹出了
        self.assertIs(popped_provider, self.tools_provider, "应该从 SystemMessage 中成功弹出 provider")
        self.assertNotIn(self.tools_provider, system_message.provider(), "provider 不应再存在于 SystemMessage 的 provider 列表中")

        # 3. 核心问题：检查顶层 Messages 的索引
        # 在理想情况下，直接修改子消息应该同步更新顶层索引。
        # 因此，我们断言 provider 现在应该是找不到的。这个测试现在应该会失败。
        provider_after_pop = messages.provider("tools")
        self.assertIsNone(provider_after_pop, "BUG: 直接从子消息中 pop 后，顶层索引未同步，仍然可以找到 provider")

        # 4. 进一步验证：渲染结果和索引内容不一致
        # 渲染结果应该不再包含 tools 内容，因为 Message 对象本身是正确的
        rendered_messages = messages.render()
        self.assertGreater(len(rendered_messages), 0, "渲染后的消息列表不应为空")
        rendered_content = rendered_messages[0]['content']
        self.assertNotIn("<tools>", rendered_content, "渲染结果中不应再包含 'tools' 的内容，证明数据源已更新")

    async def test_h_pop_message_by_index(self):
        """测试通过整数索引弹出Message的功能"""
        user_provider = Texts("User message 1")
        messages = Messages(
            SystemMessage(Texts("System message")),
            UserMessage(user_provider),
            AssistantMessage(Texts("Assistant response"))
        )

        # 初始状态断言
        self.assertEqual(len(messages), 3)
        self.assertIsNotNone(messages.provider(user_provider.name))

        # 弹出索引为 1 的 UserMessage
        popped_message = messages.pop(1)

        # 验证弹出的消息是否正确
        self.assertIsInstance(popped_message, UserMessage)
        self.assertEqual(len(popped_message.provider()), 1)
        self.assertEqual(popped_message.provider()[0].name, user_provider.name)

        # 验证 Messages 对象的当前状态
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "assistant")

        # 验证 provider 索引是否已更新
        self.assertIsNone(messages.provider(user_provider.name))

        # 测试弹出不存在的索引
        popped_none = messages.pop(99)
        self.assertIsNone(popped_none)
        self.assertEqual(len(messages), 2)

    async def test_i_generic_update_and_refresh(self):
        """测试新添加的 update 方法是否能正确更新内容并标记为 stale"""
        # 1. Setup providers
        text_provider = Texts("Hello")
        tools_provider = Tools([{"name": "tool_A"}])

        dummy_image_path = "test_dummy_image_3.png"
        with open(dummy_image_path, "w") as f: f.write("dummy content")
        image_provider = Images(url=dummy_image_path, name="logo")

        messages = Messages(UserMessage(text_provider, tools_provider, image_provider))

        # Mock the render methods to monitor calls
        text_provider.render = AsyncMock(wraps=text_provider.render)
        tools_provider.render = AsyncMock(wraps=tools_provider.render)
        image_provider.render = AsyncMock(wraps=image_provider.render)

        # 2. Initial render
        rendered_initial = await messages.render_latest()
        self.assertIn("Hello", rendered_initial[0]['content'][0]['text'])
        self.assertIn("tool_A", rendered_initial[0]['content'][1]['text'])
        self.assertEqual(text_provider.render.call_count, 1)
        self.assertEqual(tools_provider.render.call_count, 1)
        self.assertEqual(image_provider.render.call_count, 1)

        # 3. Update providers
        text_provider.update("Goodbye")
        tools_provider.update([{"name": "tool_B"}])

        new_dummy_image_path = "test_dummy_image_4.png"
        with open(new_dummy_image_path, "w") as f: f.write("new dummy content")
        image_provider.update(url=new_dummy_image_path)

        # Calling refresh again should not re-fetch yet because we haven't called messages.refresh()
        await text_provider.refresh()
        self.assertEqual(text_provider.render.call_count, 2)

        # 4. Re-render after update
        rendered_updated = await messages.render_latest()
        self.assertIn("Goodbye", rendered_updated[0]['content'][0]['text'])
        self.assertIn("tool_B", rendered_updated[0]['content'][1]['text'])

        # Verify that render was called again for all updated providers
        self.assertEqual(text_provider.render.call_count, 2)
        self.assertEqual(tools_provider.render.call_count, 2)
        self.assertEqual(image_provider.render.call_count, 2)

        # Clean up
        os.remove(dummy_image_path)
        os.remove(new_dummy_image_path)

    async def test_j_pop_last_message_without_arguments(self):
        """测试不带参数调用 pop() 时，弹出最后一个 Message"""
        m1 = SystemMessage(Texts("System"))
        m2 = UserMessage(Texts("User"))
        m3 = AssistantMessage(Texts("Assistant"))
        messages = Messages(m1, m2, m3)

        self.assertEqual(len(messages), 3)

        # Pop the last message
        popped_message = messages.pop()

        self.assertIs(popped_message, m3)
        self.assertEqual(len(messages), 2)
        self.assertIs(messages[-1], m2)

        # Pop again
        popped_message_2 = messages.pop()
        self.assertIs(popped_message_2, m2)
        self.assertEqual(len(messages), 1)

        # Pop the last one
        popped_message_3 = messages.pop()
        self.assertIs(popped_message_3, m1)
        self.assertEqual(len(messages), 0)

        # Pop from empty
        popped_none = messages.pop()
        self.assertIsNone(popped_none)

    async def test_k_image_provider_with_base64_url(self):
        """测试 Images provider 是否能正确处理 base64 data URL"""
        # A simple 1x1 transparent PNG as a base64 string
        base64_image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

        messages = Messages(
            UserMessage(
                Texts("This is a base64 image."),
                Images(url=base64_image_url, name="base64_img")
            )
        )

        rendered = await messages.render_latest()
        self.assertEqual(len(rendered), 1)

        content = rendered[0]['content']
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        # Check text part
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[0]['text'], 'This is a base64 image.')

        # Check image part
        image_content = content[1]
        self.assertEqual(image_content['type'], 'image_url')
        self.assertEqual(image_content['image_url']['url'], base64_image_url)

        # Also test the update method
        provider = messages.provider("base64_img")
        self.assertIsNotNone(provider)

        new_base64_url = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        provider.update(url=new_base64_url)

        rendered_updated = await messages.render_latest()
        self.assertEqual(rendered_updated[0]['content'][1]['image_url']['url'], new_base64_url)

    def test_l_role_message_factory(self):
        """测试 RoleMessage 工厂类是否能创建正确的子类实例"""
        system_msg = RoleMessage('system', Texts("System content"))
        user_msg = RoleMessage('user', Texts("User content"))
        assistant_msg = RoleMessage('assistant', Texts("Assistant content"))

        self.assertIsInstance(system_msg, SystemMessage)
        self.assertEqual(system_msg.role, 'system')
        self.assertIsInstance(user_msg, UserMessage)
        self.assertEqual(user_msg.role, 'user')
        self.assertIsInstance(assistant_msg, AssistantMessage)
        self.assertEqual(assistant_msg.role, 'assistant')

        # 测试无效的 role
        with self.assertRaises(ValueError):
            RoleMessage('invalid_role', Texts("Content"))

    async def test_m_optional_name_for_texts(self):
        """测试 Texts provider 的 name 参数是否可选，并能自动生成唯一名称"""
        # 1. 不提供 name
        text_provider_1 = Texts("This is a test.")
        self.assertTrue(text_provider_1.name.startswith("text_"))

        # 2. 提供 name
        text_provider_2 = Texts("This is another test.", name="my_name")
        self.assertEqual(text_provider_2.name, "my_name")

        # 3. 验证相同内容的文本生成相同的 name
        text_provider_3 = Texts("This is a test.")
        self.assertEqual(text_provider_1.name, text_provider_3.name)

        # 4. 验证不同内容的文本生成不同的 name
        text_provider_4 = Texts("This is a different test.")
        self.assertNotEqual(text_provider_1.name, text_provider_4.name)

        # 5. 在 Messages 中使用
        messages = Messages(UserMessage(text_provider_1))
        provider_from_messages = messages.provider(text_provider_1.name)
        self.assertIs(provider_from_messages, text_provider_1)

    async def test_n_string_to_texts_conversion(self):
        """测试在Message初始化时，字符串是否能被自动转换为Texts provider"""
        # 1. 初始化一个包含字符串的UserMessage
        user_message = UserMessage(self.files_provider, "This is a raw string.")

        # 验证 _items 列表中的第二个元素是否是 Texts 类的实例
        self.assertEqual(len(user_message.provider()), 2)
        self.assertIsInstance(user_message.provider()[0], Files)
        self.assertIsInstance(user_message.provider()[1], Texts)

        # 验证转换后的 Texts provider 内容是否正确
        # 我们需要异步地获取内容
        text_provider = user_message.provider()[1]
        await text_provider.refresh() # 手动刷新以获取内容
        content_block = text_provider.get_content_block()
        self.assertIsNotNone(content_block)
        self.assertEqual(content_block.content, "This is a raw string.")

        # 2. 在 Messages 容器中测试
        messages = Messages(
            SystemMessage("System prompt here."),
            user_message
        )
        await messages.refresh()
        rendered = messages.render()

        self.assertEqual(len(rendered), 2)
        self.assertEqual(rendered[0]['content'], "System prompt here.")
        # 在user message中，files provider没有内容，所以只有string provider的内容
        self.assertEqual(rendered[1]['content'], "This is a raw string.")

        # 3. 测试RoleMessage工厂类
        factory_user_msg = RoleMessage('user', "Factory-created string.")
        self.assertIsInstance(factory_user_msg, UserMessage)
        self.assertIsInstance(factory_user_msg.provider()[0], Texts)

        # 4. 测试无效类型
        with self.assertRaises(TypeError):
            UserMessage(123) # 传入不支持的整数类型

    async def test_o_list_to_providers_conversion(self):
        """测试在Message初始化时，列表内容是否能被自动转换为相应的provider"""
        # 1. 混合内容的列表
        mixed_content_list = [
            {'type': 'text', 'text': 'Describe the following image.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,VGhpcyBpcyBhIGR1bW15IGltYWdlIGZpbGUu'}}
        ]
        user_message_mixed = UserMessage(mixed_content_list)

        self.assertEqual(len(user_message_mixed.provider()), 2)
        self.assertIsInstance(user_message_mixed.provider()[0], Texts)
        self.assertIsInstance(user_message_mixed.provider()[1], Images)

        # 验证内容
        providers = user_message_mixed.provider()
        await asyncio.gather(*[p.refresh() for p in providers]) # 刷新所有providers
        self.assertEqual(providers[0].get_content_block().content, 'Describe the following image.')
        self.assertEqual(providers[1].get_content_block().content, 'data:image/png;base64,VGhpcyBpcyBhIGR1bW15IGltYWdlIGZpbGUu')

        # 2. 纯文本内容的列表
        text_only_list = [
            {'type': 'text', 'text': 'First line.'},
            {'type': 'text', 'text': 'Second line.'}
        ]
        user_message_text_only = UserMessage(text_only_list)

        self.assertEqual(len(user_message_text_only.provider()), 2)
        self.assertIsInstance(user_message_text_only.provider()[0], Texts)
        self.assertIsInstance(user_message_text_only.provider()[1], Texts)

        # 3. 在 Messages 容器中测试
        messages = Messages(UserMessage(mixed_content_list))
        rendered = await messages.render_latest()

        self.assertEqual(len(rendered), 1)
        self.assertIsInstance(rendered[0]['content'], list)
        self.assertEqual(len(rendered[0]['content']), 2)
        self.assertEqual(rendered[0]['content'][0]['type'], 'text')
        self.assertEqual(rendered[0]['content'][1]['type'], 'image_url')

        # 4. 测试无效的列表项
        invalid_list = [{'type': 'invalid_type'}]
        with self.assertRaises(ValueError):
            UserMessage(invalid_list)

    async def test_p_empty_message_boolean_context(self):
        """测试一个空的 Message 对象在布尔上下文中是否为 False"""
        # 1. 创建一个不含任何 provider 的空 UserMessage
        empty_message = UserMessage()
        self.assertFalse(empty_message, "一个空的 UserMessage 在布尔上下文中应该为 False")

        # 2. 创建一个包含 provider 的 UserMessage
        non_empty_message = UserMessage("Hello")
        self.assertTrue(non_empty_message, "一个非空的 UserMessage 在布尔上下文中应该为 True")

        # 3. 测试一个 provider 被移除后变为空消息的情况
        message_to_be_emptied = UserMessage(Texts("content", name="removable"))
        self.assertTrue(message_to_be_emptied, "消息在移除前应为 True")
        message_to_be_emptied.pop("removable")
        self.assertFalse(message_to_be_emptied, "消息在最后一个 provider 被移除后应为 False")

    async def test_q_string_addition_to_message(self):
        """测试字符串与Message对象相加的功能"""
        # 1. 创建一个 UserMessage
        original_message = UserMessage("hello")

        # 2. 将字符串与 UserMessage 相加
        new_message = "hi" + original_message

        # 3. 验证新消息的类型和内容
        self.assertIsInstance(new_message, UserMessage, "结果应该是一个 UserMessage 实例")
        self.assertEqual(len(new_message.provider()), 2, "新消息应该包含两个 provider")

        providers = new_message.provider()
        self.assertIsInstance(providers[0], Texts, "第一个 provider 应该是 Texts 类型")
        self.assertIsInstance(providers[1], Texts, "第二个 provider 应该是 Texts 类型")

        # 刷新以获取内容
        await asyncio.gather(*[p.refresh() for p in providers])

        self.assertEqual(providers[0].get_content_block().content, "hi", "第一个 provider 的内容应该是 'hi'")
        self.assertEqual(providers[1].get_content_block().content, "hello", "第二个 provider 的内容应该是 'hello'")

        # 4. 验证原始消息没有被修改
        self.assertEqual(len(original_message.provider()), 1, "原始消息不应该被修改")

        # 5. 测试 UserMessage + "string"
        new_message_add = original_message + "world"
        self.assertIsInstance(new_message_add, UserMessage)
        self.assertEqual(len(new_message_add.provider()), 2)

        providers_add = new_message_add.provider()
        await asyncio.gather(*[p.refresh() for p in providers_add])
        self.assertEqual(providers_add[0].get_content_block().content, "hello")
        self.assertEqual(providers_add[1].get_content_block().content, "world")

    async def test_r_message_addition_and_flattening(self):
        """测试 Message 对象相加和嵌套初始化时的扁平化功能"""
        # 1. 测试 "str" + UserMessage
        combined_message = "hi" + UserMessage("hello")
        self.assertIsInstance(combined_message, UserMessage)
        self.assertEqual(len(combined_message.provider()), 2)

        providers = combined_message.provider()
        await asyncio.gather(*[p.refresh() for p in providers])
        self.assertEqual(providers[0].get_content_block().content, "hi")
        self.assertEqual(providers[1].get_content_block().content, "hello")

        # 2. 测试 UserMessage(UserMessage(...)) 扁平化
        # 按照用户的要求，UserMessage(UserMessage(...)) 应该被扁平化
        nested_message = UserMessage(UserMessage("item1", "item2"))
        self.assertEqual(len(nested_message.provider()), 2)

        providers_nested = nested_message.provider()
        self.assertIsInstance(providers_nested[0], Texts)
        self.assertIsInstance(providers_nested[1], Texts)

        await asyncio.gather(*[p.refresh() for p in providers_nested])
        self.assertEqual(providers_nested[0].get_content_block().content, "item1")
        self.assertEqual(providers_nested[1].get_content_block().content, "item2")

        # 3. 结合 1 和 2，测试用户的完整场景
        final_message = UserMessage("hi" + UserMessage("hello"))
        self.assertIsInstance(final_message, UserMessage)
        self.assertEqual(len(final_message.provider()), 2)

        providers_final = final_message.provider()
        await asyncio.gather(*[p.refresh() for p in providers_final])
        self.assertEqual(providers_final[0].get_content_block().content, "hi")
        self.assertEqual(providers_final[1].get_content_block().content, "hello")

    async def test_s_len_and_pop_with_get_method(self):
        """测试 len() 功能和 pop() 返回的对象支持 .get('role')"""
        messages = Messages(
            SystemMessage("System prompt"),
            UserMessage("User question"),
            AssistantMessage("Assistant answer")
        )

        # 1. 测试 len()
        self.assertEqual(len(messages), 3, "len(messages) 应该返回消息的数量")

        # 2. 弹出中间的消息
        popped_message = messages.pop(1)
        self.assertIsNotNone(popped_message, "pop(1) 应该返回一个消息对象")
        self.assertIsInstance(popped_message, UserMessage)

        # 3. 验证弹出的消息
        # 这行会失败，因为 Message 对象没有 get 方法
        self.assertEqual(popped_message.get("role"), "user", "弹出的消息应该可以通过 .get('role') 获取角色")

        # 4. 验证 pop 后的状态
        self.assertEqual(len(messages), 2, "pop() 后消息数量应该减少")
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "assistant")

        # 5. 测试 .get() 对不存在的键返回默认值
        self.assertIsNone(popped_message.get("non_existent_key"), ".get() 对不存在的键应该返回 None")
        self.assertEqual(popped_message.get("non_existent_key", "default"), "default", ".get() 应支持默认值")

    async def test_t_pop_and_get_tool_calls(self):
        """测试弹出 ToolCalls 消息后，可以通过 .get('tool_calls') 访问其内容"""
        from dataclasses import dataclass, field
        @dataclass
        class MockFunction:
            name: str
            arguments: str

        @dataclass
        class MockToolCall:
            id: str
            type: str = "function"
            function: MockFunction = field(default_factory=lambda: MockFunction("", ""))

        tool_call_list = [MockToolCall(id="call_123", function=MockFunction(name="test", arguments="{}"))]

        messages = Messages(
            UserMessage("A regular message"),
            ToolCalls(tool_calls=tool_call_list)
        )

        # 1. 弹出 ToolCalls 消息
        popped_tool_call_message = messages.pop(1)
        self.assertIsInstance(popped_tool_call_message, ToolCalls)

        # 2. 验证 .get("tool_calls")
        retrieved_tool_calls = popped_tool_call_message.get("tool_calls")
        self.assertIsNotNone(retrieved_tool_calls)
        self.assertEqual(len(retrieved_tool_calls), 1)
        self.assertIs(retrieved_tool_calls, tool_call_list)

        # 3. 弹出普通消息
        popped_user_message = messages.pop(0)
        self.assertIsInstance(popped_user_message, UserMessage)

        # 4. 验证 .get("tool_calls") 在普通消息上返回 None
        self.assertIsNone(popped_user_message.get("tool_calls"), "在没有 tool_calls 属性的消息上 .get() 应该返回 None")

    async def test_u_message_dictionary_style_access(self):
        """测试 Message 对象是否支持字典风格的访问 (e.g., message['content'])"""
        messages = Messages(
            UserMessage("Hello, world!"),
            AssistantMessage(
                "A picture:",
                Images(url="data:image/png;base64,FAKE", name="fake_image")
            )
        )
        await messages.refresh()

        # 1. 测试简单的文本消息
        user_msg = messages[0]
        # 这两行会因为没有 __getitem__ 而失败
        self.assertEqual(user_msg['role'], 'user')
        self.assertEqual(user_msg['content'], "Hello, world!")

        # 2. 测试多模态消息
        assistant_msg = messages[1]
        self.assertEqual(assistant_msg['role'], 'assistant')
        content = assistant_msg['content']
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[1]['type'], 'image_url')

        # 3. 测试访问不存在的键
        with self.assertRaises(KeyError):
            _ = user_msg['non_existent_key']

    async def test_v_files_initialization_with_list(self):
        """测试 Files provider 是否可以使用文件路径列表进行初始化"""
        # 1. 创建两个虚拟文件
        test_file_1 = "test_file_1.txt"
        test_file_2 = "test_file_2.txt"
        with open(test_file_1, "w") as f:
            f.write("Content of file 1.")
        with open(test_file_2, "w") as f:
            f.write("Content of file 2.")

        # 2. 使用路径列表初始化 Files provider
        # 这行代码当前会失败，因为 __init__ 不接受参数
        try:
            files_provider = Files([test_file_1, test_file_2])

            # 3. 将其放入 Messages 并渲染
            messages = Messages(UserMessage(files_provider))
            rendered = await messages.render_latest()

            # 4. 验证渲染结果
            self.assertEqual(len(rendered), 1)
            content = rendered[0]['content']
            self.assertIn("<file_path>test_file_1.txt</file_path>", content)
            self.assertIn("<file_content>Content of file 1.</file_content>", content)
            self.assertIn("<file_path>test_file_2.txt</file_path>", content)
            self.assertIn("<file_content>Content of file 2.</file_content>", content)

        finally:
            # 5. 清理创建的虚拟文件
            os.remove(test_file_1)
            os.remove(test_file_2)

    async def test_w_files_initialization_with_args(self):
        """测试 Files provider 是否可以使用多个文件路径参数进行初始化"""
        # 1. 创建两个虚拟文件
        test_file_3 = "test_file_3.txt"
        test_file_4 = "test_file_4.txt"
        with open(test_file_3, "w") as f:
            f.write("Content of file 3.")
        with open(test_file_4, "w") as f:
            f.write("Content of file 4.")

        # 2. 使用多个路径参数初始化 Files provider
        # 这行代码当前会失败
        try:
            files_provider = Files(test_file_3, test_file_4)

            # 3. 将其放入 Messages 并渲染
            messages = Messages(UserMessage(files_provider))
            rendered = await messages.render_latest()

            # 4. 验证渲染结果
            self.assertEqual(len(rendered), 1)
            content = rendered[0]['content']
            self.assertIn("<file_path>test_file_3.txt</file_path>", content)
            self.assertIn("<file_content>Content of file 3.</file_content>", content)
            self.assertIn("<file_path>test_file_4.txt</file_path>", content)
            self.assertIn("<file_content>Content of file 4.</file_content>", content)

        finally:
            # 5. 清理创建的虚拟文件
            os.remove(test_file_3)
            os.remove(test_file_4)

    async def test_x_files_provider_refresh_logic(self):
        """测试 Files provider 的 refresh 是否能正确同步文件系统"""
        test_file = "test_file_refresh.txt"
        initial_content = "Initial content for refresh."
        with open(test_file, "w", encoding='utf-8') as f:
            f.write(initial_content)

        try:
            files_provider = Files(test_file)
            messages = Messages(UserMessage(files_provider))
            files_provider.render = AsyncMock(wraps=files_provider.render)

            # 1. Initial render
            await messages.render_latest()
            self.assertEqual(files_provider.render.call_count, 1)

            # 2. Modify file externally
            updated_content = "Updated content from external."
            with open(test_file, "w", encoding='utf-8') as f:
                f.write(updated_content)

            # 3. render_latest() should detect change via refresh()
            rendered_updated = await messages.render_latest()
            self.assertEqual(files_provider.render.call_count, 2)
            self.assertIn(updated_content, rendered_updated[0]['content'])

            # 4. Delete the file externally
            os.remove(test_file)

            # 5. render_latest() should now show a file not found error
            rendered_error = await messages.render_latest()
            self.assertEqual(files_provider.render.call_count, 3)
            self.assertIn("[Error: File not found", rendered_error[0]['content'])

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_y_files_provider_update_logic(self):
        """测试 Files provider 的 update 方法的两种模式"""
        test_file = "test_file_update.txt"
        initial_content = "Initial content for update."
        with open(test_file, "w", encoding='utf-8') as f:
            f.write(initial_content)

        try:
            files_provider = Files() # Start empty
            messages = Messages(UserMessage(files_provider))
            files_provider.render = AsyncMock(wraps=files_provider.render)

            # 1. Update with content from memory
            files_provider.update(test_file, "Memory content.")
            # Calling render_latest() will trigger refresh, which reads from disk and OVERWRITES memory content.
            # This is the CORRECT behavior.
            rendered_mem_then_refresh = await messages.render_latest()
            self.assertEqual(files_provider.render.call_count, 1)
            # Assert that the content is what's on disk, not what was in memory.
            self.assertIn(initial_content, rendered_mem_then_refresh[0]['content'])
            self.assertNotIn("Memory content.", rendered_mem_then_refresh[0]['content'])

            # 2. Update from disk (no content arg)
            files_provider.update(test_file)
            rendered_disk = await messages.render_latest()
            self.assertEqual(files_provider.render.call_count, 2)
            self.assertIn(initial_content, rendered_disk[0]['content'])

            # 3. Update from a non-existent file path
            files_provider.update("non_existent.txt")
            rendered_error = await messages.render_latest()
            self.assertEqual(files_provider.render.call_count, 3)
            self.assertIn("[Error: File not found", rendered_error[0]['content'])

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_z_dynamic_texts_provider(self):
        """测试 Texts provider 是否支持可调用对象以实现动态内容"""
        import time
        from datetime import datetime

        # 1. 使用 lambda 函数创建一个动态的 Texts provider
        # 每次调用 render 时，它都应该返回当前时间
        dynamic_text_provider = Texts(lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        messages = Messages(UserMessage(dynamic_text_provider))

        # 2. 第一次渲染
        rendered1 = await messages.render_latest()
        time1_str = rendered1[0]['content']
        self.assertIsNotNone(time1_str)

        # 3. 等待一秒钟
        time.sleep(1)

        # 4. 第二次渲染，并期望内容已更新
        rendered2 = await messages.render_latest()
        time2_str = rendered2[0]['content']
        self.assertIsNotNone(time2_str)

        # 5. 验证两次渲染的时间戳不同
        self.assertNotEqual(time1_str, time2_str, "动态 Texts provider 的内容在两次渲染之间应该更新")

    async def test_z2_dynamic_texts_with_prefix(self):
        """测试动态 Texts provider 包含静态前缀时也能正确更新"""
        import time
        from datetime import datetime
        import platform

        # 1. 创建一个包含静态前缀和动态内容的 provider
        # 正确的用法是将整个表达式放入 lambda
        dynamic_provider = Texts(lambda: f"平台信息：{platform.platform()}, 时间：{datetime.now().isoformat()}")
        messages = Messages(UserMessage(dynamic_provider))

        # 2. 第一次渲染
        rendered1 = await messages.render_latest()
        content1 = rendered1[0]['content']
        self.assertIn("平台信息：", content1)

        # 3. 等待一秒
        time.sleep(1)

        # 4. 第二次渲染
        rendered2 = await messages.render_latest()
        content2 = rendered2[0]['content']
        self.assertIn("平台信息：", content2)

        # 5. 验证两次内容不同（因为时间戳变了）
        self.assertNotEqual(content1, content2, "包含静态前缀的动态 provider 内容应该更新")

    async def test_z3_deferred_text_update_via_provider(self):
        """测试 Texts(name=...) 初始化, 然后通过 provider 更新内容"""
        # This test is expected to fail with a TypeError on the next line
        # because the current Texts.__init__ requires 'text'.
        deferred_text_provider = Texts(name="deferred_content")

        messages = Messages(UserMessage(deferred_text_provider))

        # Initial render: with no text, it should probably render to an empty string.
        # If there's no content, the message itself might not be rendered.
        # Let's assume an empty provider results in the message not rendering.
        await deferred_text_provider.refresh()
        # With the new logic, it should return a block with an empty string
        content_block = deferred_text_provider.get_content_block()
        self.assertIsNotNone(content_block)
        self.assertEqual(content_block.content, "")


        rendered_initial = await messages.render_latest()
        self.assertEqual(len(rendered_initial), 0)

        # 3. Get provider and update content
        provider = messages.provider("deferred_content")
        self.assertIsNotNone(provider)
        provider.update("This is the new content.")

        # 4. Re-render and validate
        rendered_updated = await messages.render_latest()
        self.assertEqual(len(rendered_updated), 1)
        self.assertEqual(rendered_updated[0]['content'], "This is the new content.")

    async def test_z4_direct_fstring_usage(self):
        """直接使用 f-string 语法，并预期其能够被处理"""

        # 这个测试将直接使用用户期望的 f-string 语法。
        # 由于 Python 的限制，这行代码会立即对 f-string 求值，
        # 导致 providers 的字符串表示形式（而不是 provider 对象本身）被插入。
        # 因此，这个测试最初会失败。
        f_string_message = f"""<user_info>
The user's OS version is {Texts(name="os_version")}.
Tools: {Tools()}
Files: {Files()}
Current time: {Texts(name="current_time")}
</user_info>"""

        # 借助新的 f-string 处理机制，UserMessage 现在可以直接消费 f-string 的结果。
        messages = Messages(UserMessage(f_string_message))

        # 初始渲染时，provider 的内容应该为空
        rendered_initial = await messages.render_latest()

        # With the new simplest rendering logic, the output should match the f-string exactly,
        # with empty strings for the providers and no leading whitespace.
        expected_initial = (
            "<user_info>\n"
            "The user's OS version is .\n"
            "Tools: \n"
            "Files: \n"
            "Current time: \n"
            "</user_info>"
        )
        self.assertEqual(rendered_initial[0]['content'].strip(), expected_initial.strip())

        # 现在，尝试通过 provider 更新内容。这应该会成功。
        messages.provider("os_version").update("TestOS")
        messages.provider("tools").update([{"name": "test_tool"}])
        messages.provider("current_time").update("2025-12-25")

        test_file = "fstring_test.txt"
        with open(test_file, "w") as f: f.write("content from f-string test")

        try:
            messages.provider("files").update(test_file)

            rendered_final = await messages.render_latest()
            final_content = rendered_final[0]['content']

            # 断言内容已经被成功更新
            tools_str = "<tools>[{'name': 'test_tool'}]</tools>"
            files_str = f"<latest_file_content><file><file_path>fstring_test.txt</file_path><file_content>content from f-string test</file_content></file>\n</latest_file_content>"

            expected_final = f"""<user_info>
The user's OS version is TestOS.
Tools: {tools_str}
Files: {files_str}
Current time: 2025-12-25
</user_info>"""
            self.assertEqual(final_content.strip(), expected_final.strip())
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_z5_fstring_with_dynamic_lambda(self):
        """测试 f-string 消息是否支持动态 lambda 函数"""
        from datetime import datetime
        import time

        # 这个测试将验证 f-string 是否能正确处理包含 lambda 的动态 provider
        f_string_message = f"""<user_info>
The user's OS version is {Texts(name="os_version")}.
Current time: {Texts(lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
</user_info>"""

        messages = Messages(UserMessage(f_string_message))
        messages.provider("os_version").update("TestOS")

        # 第一次渲染
        rendered1 = await messages.render_latest()
        content1 = rendered1[0]['content']
        self.assertIn("TestOS", content1)

        time1_str_part = content1.split("Current time:")[1].strip().split("\n")[0]


        # 等待一秒
        time.sleep(1)

        # 第二次渲染
        rendered2 = await messages.render_latest()
        content2 = rendered2[0]['content']
        time2_str_part = content2.split("Current time:")[1].strip().split("\n")[0]

        # 验证两次渲染的时间戳不同
        self.assertNotEqual(time1_str_part, time2_str_part, "f-string 中的动态 lambda 内容在两次渲染之间应该更新")

    def test_z6_direct_content_access(self):
        """测试通过 .content 属性直接访问 Texts 内容"""
        # 1. 测试静态内容
        static_text = Texts("Hello, Architext!")
        self.assertEqual(static_text.content, "Hello, Architext!")

        # 2. 测试动态内容
        from datetime import datetime
        current_time_str = datetime.now().isoformat()
        dynamic_text = Texts(lambda: current_time_str)
        self.assertEqual(dynamic_text.content, current_time_str)

        # 3. 测试更新后的内容访问
        static_text.update("Updated content.")
        self.assertEqual(static_text.content, "Updated content.")

        # 4. 测试 None 和空字符串
        none_text = Texts(name="none_text") # Provide a name when text is None
        self.assertEqual(none_text.content, "")

        empty_text = Texts("")
        self.assertEqual(empty_text.content, "")

    async def test_z7_provider_visibility(self):
        """测试 provider 的可见性标志是否能正常工作"""
        # 1. 初始化 provider，visible 默认为 True
        text_provider = Texts("Hello, World!", name="greeting")
        self.assertTrue(text_provider.visible)

        messages = Messages(SystemMessage(text_provider))

        # 2. 初始渲染，内容应该可见
        rendered_visible = await messages.render_latest()
        self.assertEqual(len(rendered_visible), 1)
        self.assertEqual(rendered_visible[0]['content'], "Hello, World!")

        # 3. 设置为不可见
        provider = messages.provider("greeting")
        provider.visible = False
        self.assertFalse(provider.visible)

        # 4. 再次渲染，内容应该消失
        # 因为 visibility 变化会 mark_stale，所以需要 render_latest
        rendered_invisible = await messages.render_latest()
        self.assertEqual(len(rendered_invisible), 0, "设置为不可见后，消息应该不被渲染")

        # 5. 再次设置为可见
        provider.visible = True
        self.assertTrue(provider.visible)

        # 6. 渲染，内容应该再次出现
        rendered_visible_again = await messages.render_latest()
        self.assertEqual(len(rendered_visible_again), 1)
        self.assertEqual(rendered_visible_again[0]['content'], "Hello, World!")

    async def test_z8_bulk_provider_visibility_control(self):
        """测试通过名称批量控制和豁免provider的可见性"""
        # 1. 创建多个同名 provider
        messages = Messages(
            UserMessage(
                Texts("First explanation.", name="explanation"),
                Texts("Second explanation.", name="explanation"),
                Texts("Some other text."),
                Texts("Third explanation.", name="explanation")
            )
        )

        # 2. 初始渲染，所有 "explanation" 都应该可见
        rendered_initial = await messages.render_latest()
        self.assertIn("First explanation.", rendered_initial[0]['content'])
        self.assertIn("Second explanation.", rendered_initial[0]['content'])
        self.assertIn("Third explanation.", rendered_initial[0]['content'])

        # 3. 获取所有名为 "explanation" 的 provider
        explanation_providers = messages.provider("explanation")
        self.assertIsInstance(explanation_providers, ProviderGroup)
        self.assertEqual(len(explanation_providers), 3)

        # 4. 将所有 "explanation" provider 设置为不可见
        # 这是需要实现的新语法
        explanation_providers.visible = False
        for p in explanation_providers:
             self.assertFalse(p.visible)

        # 5. 渲染，所有 "explanation" 的内容都应该消失
        rendered_hidden = await messages.render_latest()
        self.assertNotIn("First explanation.", rendered_hidden[0]['content'])
        self.assertNotIn("Second explanation.", rendered_hidden[0]['content'])
        self.assertNotIn("Third explanation.", rendered_hidden[0]['content'])
        self.assertIn("Some other text.", rendered_hidden[0]['content'])

        # 6. 将最后一个 "explanation" provider 设置回可见
        # 这是需要实现的另一个新语法
        explanation_providers[-1].visible = True
        self.assertTrue(explanation_providers[-1].visible)
        self.assertFalse(explanation_providers[0].visible)

        # 7. 最终渲染，只应看到最后一个 "explanation"
        rendered_final = await messages.render_latest()
        self.assertNotIn("First explanation.", rendered_final[0]['content'])
        self.assertNotIn("Second explanation.", rendered_final[0]['content'])
        self.assertIn("Third explanation.", rendered_final[0]['content'])
        self.assertIn("Some other text.", rendered_final[0]['content'])

    async def test_z9_rolemessage_content_access(self):
        """测试是否支持 RoleMessage.content 来访问渲染好的内容"""
        # 1. 创建一个简单的 UserMessage
        user_message = UserMessage("你好, Architext!")
        # 对于简单的 Texts, refresh 不是必须的, 但这是个好习惯
        # Message 类本身没有 refresh, 调用其 providers 的 refresh
        for p in user_message.provider():
            await p.refresh()

        # 2. 直接访问 .content 属性
        # 在实现该功能前，这行代码会因 AttributeError 而失败
        self.assertEqual(user_message.content, "你好, Architext!")

        # 3. 创建一个多模态消息
        multimodal_message = AssistantMessage(
            "这是一张图片:",
            Images(url="data:image/png;base64,FAKE_IMG_DATA")
        )
        for p in multimodal_message.provider():
            await p.refresh()

        # 4. 访问多模态消息的 .content 属性，期望返回一个列表
        content_list = multimodal_message.content
        self.assertIsInstance(content_list, list)
        self.assertEqual(len(content_list), 2)
        self.assertEqual(content_list[0]['type'], 'text')
        self.assertEqual(content_list[1]['type'], 'image_url')

        # 5. 测试通过 RoleMessage 工厂创建的消息
        role_message = RoleMessage('user', "通过工厂创建的内容")
        for p in role_message.provider():
            await p.refresh()
        self.assertEqual(role_message.content, "通过工厂创建的内容")

    async def test_za_message_indexing_and_length(self):
        """测试 Message 对象是否支持通过索引访问 provider 以及获取长度"""
        # 1. 创建一个 UserMessage
        mess = UserMessage(
            Texts("some instruction"),
            Texts("hi", name="done")
        )

        # 2. 测试获取长度
        # 这在实现 __len__ 之前会失败
        self.assertEqual(len(mess), 2)

        # 3. 测试通过索引访问
        # 这在修改 __getitem__ 之前会失败
        self.assertEqual(mess[-1].name, "done")
        self.assertEqual(mess[0].name, Texts("some instruction").name)
        self.assertEqual(mess[0], Texts("some instruction"))

        # 4. 测试索引越界
        with self.assertRaises(IndexError):
            _ = mess[2]

    async def test_zb_fstring_provider_invisible_on_init(self):
        """测试在f-string中初始化的provider可以被设置为不可见"""

        # 1. 在 f-string 中初始化一个 provider 并设置 visible=False
        # 在修改前，这会因为 __init__ 不接受 'visible' 参数而失败
        message_with_invisible_provider = f"""
Tools: {Tools(tools_json=[{"name": "should_not_appear"}], visible=False)}
Files: {Files(visible=True, name="files")}
"""

        messages = Messages(UserMessage(message_with_invisible_provider))

        # 2. 准备 Files provider 的内容
        test_file = "test_invisible_fstring.txt"
        with open(test_file, "w") as f:
            f.write("visible content")

        try:
            files_provider = messages.provider("files")
            self.assertIsNotNone(files_provider)
            files_provider.update(test_file)

            # 3. 渲染并验证
            rendered = await messages.render_latest()
            self.assertEqual(len(rendered), 1)
            content = rendered[0]['content']

            # 4. 验证不可见的 provider 的内容没有出现
            self.assertNotIn("<tools>", content)
            self.assertNotIn("should_not_appear", content)

            # 5. 验证可见的 provider 的内容正常出现
            self.assertIn("<latest_file_content>", content)
            self.assertIn("visible content", content)

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_zc_message_provider_by_name(self):
        """测试是否可以通过名称从 Message 对象中获取 provider"""
        # 1. 创建一个包含命名 provider 的 Message
        message = UserMessage(
            Texts("Some instruction", name="instruction"),
            Tools([{"name": "a_tool"}], name="tools"),
            Texts("Another instruction", name="instruction")
        )

        # 2. 测试获取单个 provider
        tools_provider = message.provider("tools")
        self.assertIsInstance(tools_provider, Tools)
        self.assertEqual(tools_provider.name, "tools")

        # 3. 测试获取多个同名 provider
        instruction_providers = message.provider("instruction")
        self.assertIsInstance(instruction_providers, ProviderGroup)
        self.assertEqual(len(instruction_providers), 2)
        self.assertTrue(all(isinstance(p, Texts) for p in instruction_providers))

        # 4. 测试获取不存在的 provider
        non_existent_provider = message.provider("non_existent")
        self.assertIsNone(non_existent_provider)

    async def test_zd_slicing_support(self):
        """测试 Messages 对象是否支持切片操作"""
        m1 = SystemMessage("1")
        m2 = UserMessage("2")
        m3 = AssistantMessage("3")
        m4 = UserMessage("4")
        messages = Messages(m1, m2, m3, m4)

        # 1. Test basic slicing
        sliced_messages = messages[1:3]
        self.assertIsInstance(sliced_messages, Messages)
        self.assertEqual(len(sliced_messages), 2)
        self.assertIs(sliced_messages[0], m2)
        self.assertIs(sliced_messages[1], m3)

        # 2. Test slicing with open end
        sliced_messages_open = messages[2:]
        self.assertIsInstance(sliced_messages_open, Messages)
        self.assertEqual(len(sliced_messages_open), 2)
        self.assertIs(sliced_messages_open[0], m3)
        self.assertIs(sliced_messages_open[1], m4)

        # 3. Test slicing with open start
        sliced_messages_start = messages[:2]
        self.assertIsInstance(sliced_messages_start, Messages)
        self.assertEqual(len(sliced_messages_start), 2)
        self.assertIs(sliced_messages_start[0], m1)
        self.assertIs(sliced_messages_start[1], m2)

        # 4. Test slicing a single element
        sliced_single = messages[2:3]
        self.assertIsInstance(sliced_single, Messages)
        self.assertEqual(len(sliced_single), 1)
        self.assertIs(sliced_single[0], m3)

    async def test_ze_slice_assignment(self):
        """测试 Messages 对象的切片赋值功能"""
        # 1. Setup initial Messages objects
        m1 = SystemMessage("1")
        m2 = UserMessage("2")
        m3 = AssistantMessage("3")
        m4 = UserMessage("4")
        messages1 = Messages(m1, m2, m3, m4)

        m5 = SystemMessage("5")
        m6 = UserMessage("6")
        messages2 = Messages(m5, m6)

        # 2. Perform slice assignment
        # This should replace elements from index 1 onwards in messages1
        # with all elements from messages2
        messages1[1:] = messages2

        # 3. Verify the result
        self.assertEqual(len(messages1), 3) # Should be m1, m5, m6
        self.assertIs(messages1[0], m1)
        self.assertIs(messages1[1], m5)
        self.assertIs(messages1[2], m6)

        # 4. Test assigning from a slice, with different roles to prevent merging
        messages3 = Messages(UserMessage("A"), AssistantMessage("B"), UserMessage("C"))
        messages4 = Messages(SystemMessage("X"), AssistantMessage("Y"))

        self.assertEqual(len(messages3), 3) # Verify length before assignment

        messages3[1:2] = messages4[1:] # Replace AssistantMessage("B") with AssistantMessage("Y")

        # We need to refresh to access .content property correctly
        await messages3.refresh()

        self.assertEqual(len(messages3), 3)
        self.assertEqual(messages3[0].content, "A")
        self.assertEqual(messages3[1].content, "Y")
        self.assertEqual(messages3[2].content, "C")
        self.assertIsInstance(messages3[1], AssistantMessage)

    async def test_zf_fstring_lambda_serialization(self):
        """测试包含 lambda 的 f-string 消息是否可以被序列化和反序列化"""
        import platform
        import os

        # 1. 创建一个使用 f-string 和 lambda 的动态消息
        f_string_message = f"""系统信息: {Texts(lambda: platform.platform())}"""
        messages_to_save = Messages(SystemMessage(f_string_message))

        # 2. 定义一个临时文件路径
        test_file_path = "test_lambda_serialization.pkl"

        # 3. 序列化和反序列化
        try:
            # 保存
            messages_to_save.save(test_file_path)

            # 确认文件已创建
            self.assertTrue(os.path.exists(test_file_path))

            # 加载
            messages_loaded = Messages.load(test_file_path)

            # 验证加载的对象
            self.assertIsNotNone(messages_loaded)
            self.assertIsInstance(messages_loaded, Messages)
            self.assertEqual(len(messages_loaded), 1)

            # 4. 渲染加载后的消息以验证 lambda 是否仍然有效
            rendered = await messages_loaded.render_latest()

            self.assertEqual(len(rendered), 1)
            self.assertIn("系统信息:", rendered[0]['content'])
            # 验证 platform.platform() 的结果是否在渲染内容中
            self.assertIn(platform.platform(), rendered[0]['content'])

        except Exception as e:
            # 如果出现任何异常，测试失败
            self.fail(f"序列化或反序列化带有 lambda 的 f-string 消息时出错: {e}")
        finally:
            # 5. 清理临时文件
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

    async def test_zg_provider_plus_message_addition(self):
        """测试所有 ContextProvider 子类与 Message 子类相加的功能"""
        # 1. 准备 providers
        text_provider = Texts("Some text.")
        tools_provider = Tools([{"name": "a_tool"}])

        test_file = "test_provider_addition.txt"
        with open(test_file, "w") as f: f.write("File content.")
        files_provider = Files(test_file)

        providers_to_test = [text_provider, tools_provider, files_provider]

        # 2. 准备 message aclsdd
        messages_to_test = [
            UserMessage(Texts("Initial user message.")),
            SystemMessage(Texts("Initial system message.")),
            AssistantMessage(Texts("Initial assistant message."))
        ]

        try:
            for provider in providers_to_test:
                for message in messages_to_test:
                    with self.subTest(provider=type(provider).__name__, message=type(message).__name__):
                        # 执行加法操作
                        result_message = provider + message

                        # 验证结果类型是否与原始 message 相同
                        self.assertIsInstance(result_message, type(message), f"结果应为 {type(message).__name__} 类型")

                        # 验证 provider 数量
                        self.assertEqual(len(result_message), 2, "结果消息应包含两个 provider")

                        # 验证 provider 的类型和顺序
                        result_providers = result_message.provider()
                        self.assertIsInstance(result_providers[0], type(provider), f"第一个 provider 应为 {type(provider).__name__} 类型")
                        self.assertIsInstance(result_providers[1], Texts, "第二个 provider 应为 Texts 类型")

                        # 验证原始消息没有被修改
                        self.assertEqual(len(message), 1)

        finally:
            # 3. 清理文件
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_zh_provider_plus_message_assignment(self):
        """测试 ContextProvider + Message 的结果可以赋值回 Messages 列表"""
        # 1. 准备 provider 和 messages
        text_provider = Texts("Prefix text.")
        messages = Messages(
            UserMessage("Initial user message.")
        )

        # 2. 执行操作
        # 这行代码在修改 __setitem__ 之前应该会失败
        messages[0] = text_provider + messages[0]

        # 3. 验证结果
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]), 2) # Now UserMessage has two providers

        # 4. 验证内容
        rendered = await messages.render_latest()
        self.assertEqual(rendered[0]['content'], "Prefix text.Initial user message.")

        # 5. 验证 provider 索引
        self.assertIsNotNone(messages.provider(text_provider.name))

    async def test_zi_provider_in_message_and_messages(self):
        """测试 `in` 操作符是否能检查 provider 是否存在于 Message 或 Messages 中"""
        # 1. 准备 providers 和 messages
        text_hi = Texts("hi")
        text_hello = Texts("hello")
        text_world = Texts("world")

        message = UserMessage(text_hello, text_hi)
        messages_collection = Messages(SystemMessage("System"), message)

        # 2. 测试 `in` Message
        self.assertTrue(Texts("hi") in UserMessage(Texts("hello"), Texts("hi")))
        self.assertTrue(text_hi in message)
        self.assertTrue(text_hello in message)
        self.assertFalse(text_world in message)

        # 3. 测试 `in` Messages
        self.assertTrue(text_hi in messages_collection)
        self.assertTrue(text_hello in messages_collection)
        self.assertFalse(text_world in messages_collection)

        # 4. 测试一个 Message 对象是否在 Messages 中
        self.assertTrue(message in messages_collection)
        self.assertFalse(UserMessage("not in collection") in messages_collection)

    async def test_zz_none_input_ignored(self):
        """测试在Message初始化时，None值是否被自动忽略"""
        # 1. 在初始化列表中包含 None
        message = UserMessage("Hello", None, "World")
        self.assertEqual(len(message.provider()), 2)
        self.assertIsInstance(message.provider()[0], Texts)
        self.assertIsInstance(message.provider()[1], Texts)
        rendered = await message.render_latest()
        self.assertEqual(rendered['content'], "HelloWorld")

        # 2. 测试只有 None
        message_none = SystemMessage(None)
        self.assertEqual(len(message_none.provider()), 0)
        self.assertFalse(message_none)

        # 3. 测试混合 provider 和 None
        message_mixed = SystemMessage(Texts("hi"), None)
        self.assertEqual(len(message_mixed.provider()), 1)
        self.assertIsInstance(message_mixed.provider()[0], Texts)

    async def test_zaa_has_method_for_provider_type_check(self):
        """测试 Message.has(type) 方法是否能正确检查 provider 类型"""
        # 1. 创建一个混合类型的消息
        message_with_text = UserMessage(Texts("hi"), Images("url"))

        # 2. 测试存在的情况
        # This line is expected to fail with an AttributeError before implementation
        self.assertTrue(message_with_text.has(Texts))
        self.assertTrue(message_with_text.has(Images))

        # 3. 测试不存在的情况
        self.assertFalse(message_with_text.has(Tools))

        # 4. 测试空消息
        empty_message = UserMessage()
        self.assertFalse(empty_message.has(Texts))

        # 5. 测试传入无效类型
        with self.assertRaises(TypeError):
            message_with_text.has(str)

        with self.assertRaises(TypeError):
            # Also test with a class that is not a subclass of ContextProvider
            class NotAProvider: pass
            message_with_text.has(NotAProvider)

    async def test_zab_lstrip_and_rstrip(self):
        """测试 lstrip, rstrip, 和 strip 方法是否能正确移除两侧的特定类型的 provider"""
        # 1. 定义一个用于测试的子类
        class SpecialTexts(Texts):
            pass
        url = "data:image/png;base64,FAKE_IMG"

        # 2. 创建一个复杂的测试消息
        message = UserMessage(
            Texts("leading1"),
            Texts("leading2"),
            Images(url, name="image1"),
            Texts("middle"),
            SpecialTexts("special_middle"),
            Images(url, name="image2"),
            Texts("trailing1"),
            SpecialTexts("special_trailing"), # rstrip(Texts) should stop here
            Texts("trailing2")
        )

        # 3. 测试 rstrip(Texts)
        r_stripped_message = UserMessage(*message.provider()) # 创建副本
        r_stripped_message.rstrip(Texts)
        # 应移除 "trailing2"，但在 "special_trailing" 处停止
        self.assertEqual(len(r_stripped_message), 8)
        self.assertIs(type(r_stripped_message[-1]), SpecialTexts)

        # 4. 测试 lstrip(Texts)
        l_stripped_message = UserMessage(*message.provider()) # 创建副本
        l_stripped_message.lstrip(Texts)
        # 应移除 "leading1" 和 "leading2"，但在 "image1" 处停止
        self.assertEqual(len(l_stripped_message), 7)
        self.assertIs(type(l_stripped_message[0]), Images)

        # 5. 测试 strip(Texts)
        stripped_message = UserMessage(*message.provider()) # 创建副本
        stripped_message.strip(Texts)
        # 应同时移除 "leading1", "leading2", 和 "trailing2"
        self.assertEqual(len(stripped_message), 6)
        self.assertIs(type(stripped_message[0]), Images)
        self.assertIs(type(stripped_message[-1]), SpecialTexts)

        # 6. 测试在一个只包含一种类型的消息上进行剥离
        only_texts = UserMessage(Texts("a"), Texts("b"))
        only_texts.strip(Texts)
        self.assertEqual(len(only_texts), 0)

        # 7. 测试剥离一个不包含目标类型的消息
        only_images = UserMessage(Images("url1"), Images("url2"))
        only_images.strip(Texts)
        self.assertEqual(len(only_images), 2) # 不应改变

        # 8. 测试在一个空消息上进行剥离
        empty_message = UserMessage()
        empty_message.strip(Texts)
        self.assertEqual(len(empty_message), 0)

        # 9. 测试剥离子类
        message_ending_with_special = UserMessage(Texts("a"), SpecialTexts("b"))
        message_ending_with_special.rstrip(SpecialTexts)
        self.assertEqual(len(message_ending_with_special), 1)
        self.assertIsInstance(message_ending_with_special[0], Texts)

    async def test_zac_texts_join_parameter(self):
        """测试 Texts provider 是否支持通过参数控制拼接方式"""
        # 1. 测试默认行为：直接拼接
        message_default = UserMessage(
            Texts("First line."),
            Texts("Second line.")
        )
        rendered_default = await message_default.render_latest()
        self.assertEqual(rendered_default['content'], "First line.Second line.")

        # 2. 测试新功能：使用 \n\n 拼接
        # 假设新参数为 `newline=True`
        message_newline = UserMessage(
            Texts("First paragraph."),
            Texts("Second paragraph.", newline=True)
        )
        rendered_newline = await message_newline.render_latest()
        self.assertEqual(rendered_newline['content'], "First paragraph.\n\nSecond paragraph.")

        # 3. 测试多个 provider 的情况
        message_multiple = UserMessage(
            Texts("First."),
            Texts("Second.", newline=True),
            Texts("Third.", newline=True)
        )
        rendered_multiple = await message_multiple.render_latest()
        self.assertEqual(rendered_multiple['content'], "First.\n\nSecond.\n\nThird.")

        # 4. 测试只有一个 provider 的情况
        message_single = UserMessage(
            Texts("Only one.", newline=True)
        )
        rendered_single = await message_single.render_latest()
        self.assertEqual(rendered_single['content'], "Only one.")

    async def test_zad_simple_render_without_refresh(self):
        """测试 Messages(UserMessage('hi')).render() 是否能直接同步渲染"""
        # This test checks if a simple message can be rendered synchronously
        # without an explicit `await refresh()` or `await render_latest()`.
        # Calling the synchronous render method directly on a new instance
        rendered = Messages(UserMessage("hi", Images(url="data:image/png;base64,FAKE"))).render()

        # The current implementation will likely fail here, returning []
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0]['role'], 'user')

        # Now we expect a list for multimodal content
        content = rendered[0]['content']
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[0]['text'], 'hi')
        self.assertEqual(content[1]['type'], 'image_url')
        self.assertEqual(content[1]['image_url']['url'], "data:image/png;base64,FAKE")

    async def test_zae_messages_representation(self):
        """测试 Messages 对象的 __repr__ 方法是否提供可读的输出"""
        messages = Messages(
            UserMessage("Hello"),
            AssistantMessage("Hi there!")
        )

        actual_repr = repr(messages)

        # 一个可读的字符串形式应该像 Messages([...]) 这样，并包含其内部 message 的 repr
        self.assertTrue(actual_repr.startswith("Messages(["), f"期望输出以 'Messages([' 开头，但得到 '{actual_repr}'")
        self.assertTrue(actual_repr.endswith("])"), f"期望输出以 '])' 结尾，但得到 '{actual_repr}'")
        self.assertIn("Message(role='user', items=", actual_repr)
        self.assertIn("Message(role='assistant', items=", actual_repr)

    async def test_zaf_message_absorption(self):
        """测试Message对象是否能吸收嵌套的Message对象作为其内容"""
        # 1. ToolResults吸收UserMessage
        tool_results_1 = ToolResults(tool_call_id="call_1", content=UserMessage("hi"))
        rendered_1 = await tool_results_1.render_latest()
        self.assertEqual(rendered_1['content'], "hi")
        self.assertEqual(rendered_1['tool_call_id'], "call_1")

        # 2. UserMessage吸收AssistantMessage
        user_message_1 = UserMessage("prefix", AssistantMessage("absorbed content"))
        rendered_user_1 = await user_message_1.render_latest()
        self.assertEqual(rendered_user_1['content'], "prefixabsorbed content")
        self.assertEqual(len(user_message_1.provider()), 2) # Should be flattened

        # 3. 复杂嵌套
        final_message = ToolResults(tool_call_id="call_final", content=UserMessage("A", AssistantMessage("B", UserMessage("C"))))
        rendered_final = await final_message.render_latest()
        self.assertEqual(rendered_final['content'], "ABC")

        # 4. 组合情况: ToolResults(UserMessage(Texts("a"), Texts("b"))) -> content="ab"
        tool_results_2 = ToolResults(tool_call_id="call_2", content=UserMessage(Texts("a"), Texts("b")))
        rendered_2 = await tool_results_2.render_latest()
        self.assertEqual(rendered_2['content'], "ab")

        # 5. 包含多模态内容的情况 (ToolResults应该只提取文本)
        tool_results_3 = ToolResults(tool_call_id="call_3", content=UserMessage("text part", Images(url="some_url")))
        rendered_3 = await tool_results_3.render_latest()
        self.assertEqual(rendered_3['content'], "text part") # Images should be ignored

        # 6. 直接传入字符串的情况应保持不变
        tool_results_4 = ToolResults(tool_call_id="call_4", content="just a string")
        rendered_4 = await tool_results_4.render_latest()
        self.assertEqual(rendered_4['content'], "just a string")

        # 7. 传入一个空的 Message
        tool_results_5 = ToolResults(tool_call_id="call_5", content=UserMessage())
        rendered_5 = await tool_results_5.render_latest()
        self.assertEqual(rendered_5['content'], "")

    async def test_zzb_final_message_render_logic(self):
        """
        最终版测试:
        - render() 首次调用保证获取完整结果。
        - 后续 render() 调用返回缓存结果，不会自动刷新。
        - refresh() 显式刷新内容。
        - render_latest() 总是获取最新内容。
        """
        from datetime import datetime
        import time

        # 1. 创建带有动态 provider 的 Message
        timestamp_provider = Texts(lambda: str(datetime.now().timestamp()))
        message = UserMessage("Time: ", timestamp_provider)

        # 2. 第一次调用 render() - 应该刷新并返回完整内容
        rendered_1 = await message.render()
        content1 = rendered_1['content']
        timestamp1_str = content1.replace("Time: ", "")
        self.assertTrue(timestamp1_str, "render() 第一次调用应返回完整动态内容")

        # 3. 第二次调用 render() - 不应自动刷新，返回缓存的内容
        time.sleep(1)
        rendered_2 = await message.render()
        content2 = rendered_2['content']
        self.assertEqual(content1, content2, "第二次调用 render() 应返回缓存内容，不应刷新")

        # 4. 调用 refresh() - 显式刷新
        time.sleep(1)
        await message.refresh()

        # 5. refresh() 后调用 render() - 应该返回刚刚刷新的新内容
        rendered_3 = await message.render()
        content3 = rendered_3['content']
        timestamp3_str = content3.replace("Time: ", "")
        self.assertNotEqual(content2, content3, "refresh() 后 render() 应返回新内容")

        # 6. 调用 render_latest() - 总是获取最新内容
        time.sleep(1)
        rendered_latest = await message.render_latest()
        content_latest = rendered_latest['content']
        timestamp_latest_str = content_latest.replace("Time: ", "")
        self.assertNotEqual(content3, content_latest, "render_latest() 应总是获取最新内容")

        # 7. 测试 ToolResults
        tool_results_msg = ToolResults(tool_call_id="call_123", content="Result from tool")
        # ToolResults's content is static, so render() should always return the same full content
        rendered_tool_1 = await tool_results_msg.render()
        self.assertEqual(rendered_tool_1, {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Result from tool"
        })
        # Subsequent calls should also work
        rendered_tool_2 = await tool_results_msg.render()
        self.assertEqual(rendered_tool_1, rendered_tool_2)

    async def test_zag_iadd_on_provider(self):
        """测试对 provider 使用 += 操作符来追加文本"""
        class Goal(Texts):
            def __init__(self, text: Optional[Union[str, Callable[[], str]]] = None, name: str = "goal"):
                super().__init__(text=text, name=name)

            async def render(self) -> Optional[str]:
                content = await super().render()
                if content is None:
                    return None
                return f"<goal>{content}</goal>"

        messages = Messages(UserMessage(Goal("hi")))

        # This is the new syntax we want to test
        goal_provider = messages.provider("goal")
        goal_provider += "test"

        rendered = await messages.render_latest()

        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0]['content'], "<goal>hitest</goal>")

    async def test_zz_user_message_auto_merging(self):
        """测试连续的UserMessage是否能自动合并"""
        # 场景1: 初始化时合并
        messages_init = Messages(UserMessage("hi"), UserMessage("hi2"))
        self.assertEqual(len(messages_init), 1, "初始化时，两个连续的UserMessage应该合并为一个")
        self.assertEqual(len(messages_init[0]), 2, "合并后的UserMessage应该包含两个Texts provider")

        rendered_init = await messages_init.render_latest()
        self.assertEqual(rendered_init[0]['content'], "hihi2", "合并后渲染的内容不正确")

        # 场景2: 追加时合并
        messages_append = Messages(UserMessage("hi"))
        messages_append.append(UserMessage("hi2"))
        self.assertEqual(len(messages_append), 1, "追加时，两个连续的UserMessage应该合并为一个")
        self.assertEqual(len(messages_append[0]), 2, "追加合并后的UserMessage应该包含两个Texts provider")

        rendered_append = await messages_append.render_latest()
        self.assertEqual(rendered_append[0]['content'], "hihi2", "追加合并后渲染的内容不正确")

        # 场景3: 追加RoleMessage时合并
        messages_append.append(RoleMessage("user", "hi3"))
        self.assertEqual(len(messages_append), 1, "追加RoleMessage时，连续的UserMessage应该合并为一个")
        self.assertEqual(len(messages_append[0]), 3, "追加RoleMessage合并后的UserMessage应该包含三个Texts provider")

        rendered_append_role = await messages_append.render_latest()
        self.assertEqual(rendered_append_role[0]['content'], "hihi2hi3", "追加RoleMessage合并后渲染的内容不正确")

        # 场景4: 追加包含ContextProvider和字符串组合的RoleMessage时合并
        class Goal(Texts):
            def __init__(self, text: Optional[Union[str, Callable[[], str]]] = None, name: str = "goal", visible: bool = True, newline: bool = False):
                super().__init__(text=text, name=name, visible=visible, newline=newline)

            async def render(self) -> Optional[str]:
                content = await super().render()
                if content is None:
                    return None
                return f"<goal>{content}</goal>"

        messages_append.append(RoleMessage("user", Goal("goal") + "hi4"))
        self.assertEqual(len(messages_append), 1, "追加(ContextProvider + str)的RoleMessage时，未能正确合并")
        self.assertEqual(len(messages_append[0]), 4, "追加(ContextProvider + str)的RoleMessage合并后的provider数量不正确")

        rendered_append_combo = await messages_append.render_latest()
        self.assertEqual(rendered_append_combo[0]['content'], "hihi2hi3<goal>goalhi4</goal>", "追加(ContextProvider + str)合并后渲染的内容不正确")

        # 场景5: 被空消息隔开的同角色消息在渲染时合并
        messages_separated = Messages(UserMessage("hi"), AssistantMessage(""), UserMessage("hi2"))
        rendered_separated = await messages_separated.render_latest()
        self.assertEqual(len(rendered_separated), 1, "被空消息隔开的同角色消息在渲染时应该合并")
        self.assertEqual(rendered_separated[0]['content'], "hihi2", "被空消息隔开的同角色消息合并后内容不正确")

    async def test_zzc_files_update_with_head(self):
        """测试 Files.update 是否支持 head 参数以及新的 refresh 逻辑"""
        test_file = "test_file_head.txt"
        file_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        with open(test_file, "w", encoding='utf-8') as f:
            f.write(file_content)

        try:
            messages = Messages(UserMessage(Files(name="files")))

            # 1. Test update with head=3 using the provider from Messages
            files_provider = messages.provider("files")
            self.assertIsInstance(files_provider, Files)
            files_provider.update(path=test_file, head=3)

            rendered_head = await messages.render_latest()
            expected_head_content = "Line 1\nLine 2\nLine 3"
            self.assertIn(f"<file_content>{expected_head_content}</file_content>", rendered_head[0]['content'])
            self.assertNotIn("Line 4", rendered_head[0]['content'])

            # 2. Test update without head (default behavior)
            files_provider.update(path=test_file)
            rendered_full = await messages.render_latest()
            self.assertIn(f"<file_content>{file_content}</file_content>", rendered_full[0]['content'])

            # 3. Test that refresh overwrites manual content when file exists
            files_provider.update(path=test_file, content="manual content")
            rendered_overwritten = await messages.render_latest()
            self.assertIn(f"<file_content>{file_content}</file_content>", rendered_overwritten[0]['content'])
            self.assertNotIn("manual content", rendered_overwritten[0]['content'])

            # 4. Test that manual content is kept if file does not exist
            non_existent_file = "non_existent_for_sure.txt"
            files_provider.update(path=non_existent_file, content="manual content to keep")
            rendered_kept = await messages.render_latest()
            self.assertIn("<file_content>manual content to keep</file_content>", rendered_kept[0]['content'])

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_zzd_files_update_with_string_head(self):
        """测试 Files.update 是否能自动将字符串 head 参数转换为整数"""
        test_file = "test_file_string_head.txt"
        file_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        with open(test_file, "w", encoding='utf-8') as f:
            f.write(file_content)

        try:
            messages = Messages(UserMessage(Files(name="files")))

            # Test update with head="3"
            files_provider = messages.provider("files")
            self.assertIsInstance(files_provider, Files)
            # This is the part that will likely fail before implementation
            files_provider.update(path=test_file, head="3")

            rendered_head = await messages.render_latest()
            expected_head_content = "Line 1\nLine 2\nLine 3"
            self.assertIn(f"<file_content>{expected_head_content}</file_content>", rendered_head[0]['content'])
            self.assertNotIn("Line 4", rendered_head[0]['content'])

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_zzc_files_update_with_head_and_content(self):
        """测试update同时设置path, content, head时的行为"""
        existing_file = "test_file_existing.txt"
        non_existent_file = "test_file_non_existent.txt"
        existing_content = "Line 1 (from file)\nLine 2 (from file)\nLine 3 (from file)\nLine 4 (from file)"
        manual_content = "Line 1 (manual)\nLine 2 (manual)\nLine 3 (manual)\nLine 4 (manual)"

        # --- 场景1: 文件存在 ---
        with open(existing_file, "w", encoding='utf-8') as f:
            f.write(existing_content)
        try:
            messages = Messages(UserMessage(Files(name="files")))
            files_provider = messages.provider("files")
            files_provider.update(path=existing_file, content=manual_content, head=2)
            rendered_existing = await messages.render_latest()
            expected_existing_head = "Line 1 (from file)\nLine 2 (from file)"
            self.assertIn(f"<file_content>{expected_existing_head}</file_content>", rendered_existing[0]['content'])
            self.assertNotIn("manual", rendered_existing[0]['content'])
        finally:
            if os.path.exists(existing_file):
                os.remove(existing_file)

        # --- 场景2: 文件不存在 ---
        try:
            messages = Messages(UserMessage(Files(name="files")))
            files_provider = messages.provider("files")
            files_provider.update(path=non_existent_file, content=manual_content, head=2)
            rendered_non_existent = await messages.render_latest()
            expected_manual_head = "Line 1 (manual)\nLine 2 (manual)"
            self.assertIn(f"<file_content>{expected_manual_head}</file_content>", rendered_non_existent[0]['content'])
            self.assertNotIn("from file", rendered_non_existent[0]['content'])
        finally:
            if os.path.exists(non_existent_file):
                os.remove(non_existent_file)

    async def test_auto_convert_image_dict_to_image_provider(self):
        """测试 UserMessage 是否能自动转换 image_url 字典为 Images provider"""
        base64_image_data = "/9j/4AAQSkZJRgABAQEAYABgAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAEAAAAAAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKKAP/2Q=="
        image_dict = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image_data}'
            }
        }

        # This is the functionality we are testing.
        # The UserMessage should be able to take this dictionary directly.
        message = UserMessage(image_dict)

        # 1. Verify that the message contains exactly one provider.
        self.assertEqual(len(message.provider()), 1)

        # 2. Verify that the provider is an instance of the Images class.
        image_provider = message.provider()[0]
        self.assertIsInstance(image_provider, Images)

        # 3. Verify that the content of the Images provider is correct.
        rendered = await message.render_latest()
        self.assertEqual(rendered['content'][0]['image_url']['url'], image_dict['image_url']['url'])

    async def test_zzd_texts_cache_control(self):
        """测试 Texts provider deepcopy in relation to dynamic content"""
        import copy
        counter = 0
        def dynamic_content():
            nonlocal counter
            counter += 1
            return str(counter)

        # When a message with a dynamic provider is deep-copied,
        # the dynamic nature (the callable) should be preserved.
        counter = 0
        # provider_in_fstring = f"{Texts(lambda: dynamic_content())}"
        provider_in_fstring = f"{Texts(dynamic_content)}"
        messages_copied = copy.deepcopy(Messages(UserMessage(provider_in_fstring)))

        # Each call to render_latest() should re-evaluate the dynamic content.
        await messages_copied.render_latest()
        await messages_copied.render_latest()
        await messages_copied.render_latest()
        self.assertEqual(counter, 3, "Dynamic function should be re-evaluated on each render_latest call")


# ==============================================================================
# 6. 演示
# ==============================================================================
async def run_demo():
    # --- 1. 初始化提供者 ---
    system_prompt_provider = Texts("你是一个AI助手。", name="system_prompt")
    tools_provider = Tools(tools_json=[{"name": "read_file"}])
    files_provider = Files()

    # --- 2. 演示新功能：优雅地构建 Messages ---
    print("\n>>> 场景 A: 使用新的、优雅的构造函数直接初始化 Messages")
    messages = Messages(
        SystemMessage(system_prompt_provider, tools_provider),
        UserMessage(files_provider, Texts("这是我的初始问题。")),
        UserMessage(Texts("这是我的初始问题2。"))
    )

    print("\n--- 渲染后的初始 Messages (首次渲染，全部刷新) ---")
    for msg_dict in await messages.render_latest(): print(msg_dict)
    print("-" * 40)

    # --- 3. 演示穿透更新 ---
    print("\n>>> 场景 B: 穿透更新 File Provider，渲染时自动刷新")
    files_provider_instance = messages.provider("files")
    if isinstance(files_provider_instance, Files):
        files_provider_instance.update("file1.py", "这是新的文件内容！")

    print("\n--- 再次渲染 Messages (只有文件提供者会刷新) ---")
    for msg_dict in await messages.render_latest(): print(msg_dict)
    print("-" * 40)

    # --- 4. 演示全局 Pop 和通过索引 Insert ---
    print("\n>>> 场景 C: 全局 Pop 工具提供者，并 Insert 到 UserMessage 中")
    popped_tools_provider = messages.pop("tools")
    if popped_tools_provider:
        messages[1].insert(0, popped_tools_provider)
        print(f"\n已成功将 '{popped_tools_provider.name}' 提供者移动到用户消息。")

    print("\n--- Pop 和 Insert 后渲染的 Messages (验证移动效果) ---")
    for msg_dict in messages.render(): print(msg_dict)
    print("-" * 40)

    # --- 5. 演示多模态渲染 ---
    print("\n>>> 场景 D: 演示多模态 (文本+图片) 渲染")
    with open("dummy_image.png", "w") as f:
        f.write("This is a dummy image file.")

    multimodal_message = Messages(
        UserMessage(
            Texts("What do you see in this image?"),
            Images(url="dummy_image.png")
        )
    )
    print("\n--- 渲染后的多模态 Message ---")
    for msg_dict in await multimodal_message.render_latest():
        if isinstance(msg_dict['content'], list):
            for item in msg_dict['content']:
                if item['type'] == 'image_url':
                    item['image_url']['url'] = item['image_url']['url'][:80] + "..."
        print(msg_dict)
    print("-" * 40)

    # --- 6. 演示 Tool-Use 流程 ---
    print("\n>>> 场景 E: 模拟完整的 Tool-Use 流程")
    # 模拟一个 OpenAI SDK 返回的 tool_call 对象 (使用 dataclass 或 mock object)
    from dataclasses import dataclass, field
    @dataclass
    class MockFunction:
        name: str
        arguments: str

    @dataclass
    class MockToolCall:
        id: str
        type: str = "function"
        function: MockFunction = field(default_factory=MockFunction)


    tool_call_request = [
        MockToolCall(
            id="call_rddWXkDikIxllRgbPrR6XjtMVSBPv",
            function=MockFunction(name="add", arguments='{"b": 10, "a": 5}')
        )
    ]

    tool_use_messages = Messages(
        SystemMessage(Texts("You are a helpful assistant. You must use the provided tools to answer questions.")),
        UserMessage(Texts("What is the sum of 5 and 10?")),
        ToolCalls(tool_call_request),
        ToolResults(tool_call_id="call_rddWXkDikIxllRgbPrR6XjtMVSBPv", content="15"),
        AssistantMessage(Texts("The sum of 5 and 10 is 15."))
    )

    print("\n--- 渲染后的 Tool-Use Messages ---")
    import json
    print(json.dumps(await tool_use_messages.render_latest(), indent=2))
    print("-" * 40)

if __name__ == '__main__':
    # 为了在普通脚本环境中运行，添加这两行
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestContextManagement)
    runner = unittest.TextTestRunner()
    runner.run(suite)
    asyncio.run(run_demo())
