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
        self.system_prompt_provider = Texts("system_prompt", "你是一个AI助手。")
        self.tools_provider = Tools(tools_json=[{"name": "read_file"}])
        self.files_provider = Files()

    async def test_a_initial_construction_and_render(self):
        """测试优雅的初始化和首次渲染"""
        messages = Messages(
            SystemMessage(self.system_prompt_provider, self.tools_provider),
            UserMessage(self.files_provider, Texts("user_input", "这是我的初始问题。"))
        )

        self.assertEqual(len(messages), 2)
        rendered = await messages.render_latest()

        self.assertEqual(len(rendered), 2)
        self.assertIn("<tools>", rendered[0]['content'])
        self.assertNotIn("<files>", rendered[1]['content'])

    async def test_b_provider_passthrough_and_refresh(self):
        """测试通过 mock 验证缓存和刷新逻辑"""
        # 我们真正关心的是 _fetch_content 是否被不必要地调用
        # 所以我们 mock 底层的它，而不是 refresh 方法
        original_fetch_content = self.files_provider._fetch_content
        self.files_provider._fetch_content = AsyncMock(side_effect=original_fetch_content)

        messages = Messages(UserMessage(self.files_provider))

        # 1. 首次刷新
        self.files_provider.update("path1", "content1")
        await messages.refresh()
        # _fetch_content 应该被调用了 1 次
        self.assertEqual(self.files_provider._fetch_content.call_count, 1)

        # 2. 再次刷新，内容未变，不应再次调用 _fetch_content
        await messages.refresh()
        # 调用次数应该仍然是 1，证明缓存生效
        self.assertEqual(self.files_provider._fetch_content.call_count, 1)

        # 3. 更新文件内容，这会标记 provider 为 stale
        self.files_provider.update("path2", "content2")

        # 4. 再次刷新，现在应该会重新调用 _fetch_content
        await messages.refresh()
        rendered = messages.render()
        # 调用次数应该变为 2
        self.assertEqual(self.files_provider._fetch_content.call_count, 2)
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
                Texts("prompt", "Describe the image."),
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
                Texts("prefix", "Look at this:"),
                Images(url=dummy_image_path, name="image"), # Explicit name for popping
                Texts("suffix", "Any thoughts?")
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
            UserMessage(Texts("part1", "Hello,")),
            UserMessage(Texts("part2", "world!")),
            SystemMessage(Texts("system", "System prompt.")),
            UserMessage(Texts("part3", "How are you?"))
        )
        # Should be merged into: User, System, User
        self.assertEqual(len(messages), 3)
        self.assertEqual(len(messages[0]._items), 2) # First UserMessage has 2 items
        self.assertEqual(messages[0]._items[1].name, "part2")
        self.assertEqual(messages[1].role, "system")
        self.assertEqual(messages[2].role, "user")

        # 2. Test merging during append
        messages.append(UserMessage(Texts("part4", "I am fine.")))
        self.assertEqual(len(messages), 3) # Still 3 messages
        self.assertEqual(len(messages[2]._items), 2) # Last UserMessage now has 2 items
        self.assertEqual(messages[2]._items[1].name, "part4")

        # 3. Test appending a different role
        messages.append(SystemMessage(Texts("system2", "Another prompt.")))
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
        messages = Messages(
            SystemMessage(Texts("system_prompt", "System message")),
            UserMessage(Texts("user_prompt_1", "User message 1")),
            AssistantMessage(Texts("assistant_response", "Assistant response"))
        )

        # 初始状态断言
        self.assertEqual(len(messages), 3)
        self.assertIsNotNone(messages.provider("user_prompt_1"))

        # 弹出索引为 1 的 UserMessage
        popped_message = messages.pop(1)

        # 验证弹出的消息是否正确
        self.assertIsInstance(popped_message, UserMessage)
        self.assertEqual(len(popped_message.providers()), 1)
        self.assertEqual(popped_message.providers()[0].name, "user_prompt_1")

        # 验证 Messages 对象的当前状态
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "assistant")

        # 验证 provider 索引是否已更新
        self.assertIsNone(messages.provider("user_prompt_1"))

        # 测试弹出不存在的索引
        popped_none = messages.pop(99)
        self.assertIsNone(popped_none)
        self.assertEqual(len(messages), 2)

    async def test_i_generic_update_and_refresh(self):
        """测试新添加的 update 方法是否能正确更新内容并标记为 stale"""
        # 1. Setup providers
        text_provider = Texts("greet", "Hello")
        tools_provider = Tools([{"name": "tool_A"}])

        dummy_image_path = "test_dummy_image_3.png"
        with open(dummy_image_path, "w") as f: f.write("dummy content")
        image_provider = Images(url=dummy_image_path, name="logo")

        messages = Messages(UserMessage(text_provider, tools_provider, image_provider))

        # Mock the _fetch_content methods to monitor calls
        text_provider._fetch_content = AsyncMock(wraps=text_provider._fetch_content)
        tools_provider._fetch_content = AsyncMock(wraps=tools_provider._fetch_content)
        image_provider._fetch_content = AsyncMock(wraps=image_provider._fetch_content)

        # 2. Initial render
        rendered_initial = await messages.render_latest()
        self.assertIn("Hello", rendered_initial[0]['content'][0]['text'])
        self.assertIn("tool_A", rendered_initial[0]['content'][1]['text'])
        self.assertEqual(text_provider._fetch_content.call_count, 1)
        self.assertEqual(tools_provider._fetch_content.call_count, 1)
        self.assertEqual(image_provider._fetch_content.call_count, 1)

        # 3. Update providers
        text_provider.update("Goodbye")
        tools_provider.update([{"name": "tool_B"}])

        new_dummy_image_path = "test_dummy_image_4.png"
        with open(new_dummy_image_path, "w") as f: f.write("new dummy content")
        image_provider.update(url=new_dummy_image_path)

        # Calling refresh again should not re-fetch yet because we haven't called messages.refresh()
        await text_provider.refresh()
        self.assertEqual(text_provider._fetch_content.call_count, 2)

        # 4. Re-render after update
        rendered_updated = await messages.render_latest()
        self.assertIn("Goodbye", rendered_updated[0]['content'][0]['text'])
        self.assertIn("tool_B", rendered_updated[0]['content'][1]['text'])

        # Verify that _fetch_content was called again for all updated providers
        self.assertEqual(text_provider._fetch_content.call_count, 2)
        self.assertEqual(tools_provider._fetch_content.call_count, 2)
        self.assertEqual(image_provider._fetch_content.call_count, 2)

        # Clean up
        os.remove(dummy_image_path)
        os.remove(new_dummy_image_path)

    async def test_j_pop_last_message_without_arguments(self):
        """测试不带参数调用 pop() 时，弹出最后一个 Message"""
        m1 = SystemMessage(Texts("system", "System"))
        m2 = UserMessage(Texts("user", "User"))
        m3 = AssistantMessage(Texts("assistant", "Assistant"))
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
                Texts("prompt", "This is a base64 image."),
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
        UserMessage(files_provider, Texts("user_input", "这是我的初始问题。")),
        UserMessage(Texts("user_input2", "这是我的初始问题2。"))
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
            Texts("prompt", "What do you see in this image?"),
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
        SystemMessage(Texts("system_prompt", "You are a helpful assistant. You must use the provided tools to answer questions.")),
        UserMessage(Texts("user_question", "What is the sum of 5 and 10?")),
        ToolCalls(tool_call_request),
        ToolResults(tool_call_id="call_rddWXkDikIxllRgbPrR6XjtMVSBPv", content="15"),
        AssistantMessage(Texts("final_answer", "The sum of 5 and 10 is 15."))
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
