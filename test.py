import unittest
from unittest.mock import AsyncMock
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
        await messages.refresh()
        rendered = messages.render()

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
        await messages.refresh()
        initial_rendered = messages.render()
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

if __name__ == '__main__':
    # 为了在普通脚本环境中运行，添加这两行
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestContextManagement))
    runner = unittest.TextTestRunner()
    runner.run(suite)
