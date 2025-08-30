import unittest
from unittest.mock import AsyncMock

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from architext import *

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
        # 我们真正关心的是 render 是否被不必要地调用
        # 所以我们 mock 底层的它，而不是 refresh 方法
        original_render = self.files_provider.render
        self.files_provider.render = AsyncMock(side_effect=original_render)

        messages = Messages(UserMessage(self.files_provider))

        # 1. 首次刷新
        self.files_provider.update("path1", "content1")
        await messages.refresh()
        # render 应该被调用了 1 次
        self.assertEqual(self.files_provider.render.call_count, 1)

        # 2. 再次刷新，内容未变，不应再次调用 render
        await messages.refresh()
        # 调用次数应该仍然是 1，证明缓存生效
        self.assertEqual(self.files_provider.render.call_count, 1)

        # 3. 更新文件内容，这会标记 provider 为 stale
        self.files_provider.update("path2", "content2")

        # 4. 再次刷新，现在应该会重新调用 render
        await messages.refresh()
        rendered = messages.render()
        # 调用次数应该变为 2
        self.assertEqual(self.files_provider.render.call_count, 2)
        # 并且渲染结果包含了新内容
        self.assertIn("content2", rendered[0]['content'])

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

        # 3. Render again, should fall back to string content
        rendered_str = messages.render() # No refresh needed
        content_str = rendered_str[0]['content']
        self.assertIsInstance(content_str, str)
        self.assertEqual(content_str, "Look at this:\n\nAny thoughts?")

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
        self.assertNotIn(self.tools_provider, system_message.providers(), "provider 不应再存在于 SystemMessage 的 providers 列表中")

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
        self.assertEqual(len(popped_message.providers()), 1)
        self.assertEqual(popped_message.providers()[0].name, user_provider.name)

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
        self.assertEqual(len(user_message.providers()), 2)
        self.assertIsInstance(user_message.providers()[0], Files)
        self.assertIsInstance(user_message.providers()[1], Texts)

        # 验证转换后的 Texts provider 内容是否正确
        # 我们需要异步地获取内容
        text_provider = user_message.providers()[1]
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
        self.assertIsInstance(factory_user_msg.providers()[0], Texts)

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

        self.assertEqual(len(user_message_mixed.providers()), 2)
        self.assertIsInstance(user_message_mixed.providers()[0], Texts)
        self.assertIsInstance(user_message_mixed.providers()[1], Images)

        # 验证内容
        providers = user_message_mixed.providers()
        await asyncio.gather(*[p.refresh() for p in providers]) # 刷新所有providers
        self.assertEqual(providers[0].get_content_block().content, 'Describe the following image.')
        self.assertEqual(providers[1].get_content_block().content, 'data:image/png;base64,VGhpcyBpcyBhIGR1bW15IGltYWdlIGZpbGUu')

        # 2. 纯文本内容的列表
        text_only_list = [
            {'type': 'text', 'text': 'First line.'},
            {'type': 'text', 'text': 'Second line.'}
        ]
        user_message_text_only = UserMessage(text_only_list)

        self.assertEqual(len(user_message_text_only.providers()), 2)
        self.assertIsInstance(user_message_text_only.providers()[0], Texts)
        self.assertIsInstance(user_message_text_only.providers()[1], Texts)

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
        self.assertEqual(len(new_message.providers()), 2, "新消息应该包含两个 provider")

        providers = new_message.providers()
        self.assertIsInstance(providers[0], Texts, "第一个 provider 应该是 Texts 类型")
        self.assertIsInstance(providers[1], Texts, "第二个 provider 应该是 Texts 类型")

        # 刷新以获取内容
        await asyncio.gather(*[p.refresh() for p in providers])

        self.assertEqual(providers[0].get_content_block().content, "hi", "第一个 provider 的内容应该是 'hi'")
        self.assertEqual(providers[1].get_content_block().content, "hello", "第二个 provider 的内容应该是 'hello'")

        # 4. 验证原始消息没有被修改
        self.assertEqual(len(original_message.providers()), 1, "原始消息不应该被修改")

        # 5. 测试 UserMessage + "string"
        new_message_add = original_message + "world"
        self.assertIsInstance(new_message_add, UserMessage)
        self.assertEqual(len(new_message_add.providers()), 2)

        providers_add = new_message_add.providers()
        await asyncio.gather(*[p.refresh() for p in providers_add])
        self.assertEqual(providers_add[0].get_content_block().content, "hello")
        self.assertEqual(providers_add[1].get_content_block().content, "world")

    async def test_r_message_addition_and_flattening(self):
        """测试 Message 对象相加和嵌套初始化时的扁平化功能"""
        # 1. 测试 "str" + UserMessage
        combined_message = "hi" + UserMessage("hello")
        self.assertIsInstance(combined_message, UserMessage)
        self.assertEqual(len(combined_message.providers()), 2)

        providers = combined_message.providers()
        await asyncio.gather(*[p.refresh() for p in providers])
        self.assertEqual(providers[0].get_content_block().content, "hi")
        self.assertEqual(providers[1].get_content_block().content, "hello")

        # 2. 测试 UserMessage(UserMessage(...)) 扁平化
        # 按照用户的要求，UserMessage(UserMessage(...)) 应该被扁平化
        nested_message = UserMessage(UserMessage("item1", "item2"))
        self.assertEqual(len(nested_message.providers()), 2)

        providers_nested = nested_message.providers()
        self.assertIsInstance(providers_nested[0], Texts)
        self.assertIsInstance(providers_nested[1], Texts)

        await asyncio.gather(*[p.refresh() for p in providers_nested])
        self.assertEqual(providers_nested[0].get_content_block().content, "item1")
        self.assertEqual(providers_nested[1].get_content_block().content, "item2")

        # 3. 结合 1 和 2，测试用户的完整场景
        final_message = UserMessage("hi" + UserMessage("hello"))
        self.assertIsInstance(final_message, UserMessage)
        self.assertEqual(len(final_message.providers()), 2)

        providers_final = final_message.providers()
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
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestContextManagement))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    asyncio.run(run_demo())
