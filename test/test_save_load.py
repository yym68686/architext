import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architext.core import (
    Messages,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCalls,
    ToolResults,
    Texts,
    Tools,
    Files,
)

async def main():
    """
    Tests the save and load functionality of the Messages class using pickle.
    """
    print("--- Test Save/Load (pickle) ---")

    # 1. Create an initial Messages object
    messages = Messages(
        SystemMessage(Texts("system_prompt", "You are a helpful assistant.")),
        UserMessage(Texts("user_input", "What is the weather in Shanghai?")),
        AssistantMessage(Texts("thought", "I should call a tool for this.")),
        ToolCalls(tool_calls=[{
            'id': 'call_1234',
            'type': 'function',
            'function': {'name': 'get_weather', 'arguments': '{"location": "Shanghai"}'}
        }]),
        ToolResults(tool_call_id="call_1234", content='{"temperature": "25°C"}')
    )

    # Add a message with Files provider
    files_provider = Files()
    files_provider.update("test.txt", "This is a test file.")
    messages.append(UserMessage(files_provider))

    # Render the original messages
    original_render = await messages.render_latest()
    print("Original Messages Render:")
    print(original_render)

    # 2. Save the messages to a file
    file_path = "test_messages.pkl"
    messages.save(file_path)
    print(f"\nMessages saved to {file_path}")

    assert os.path.exists(file_path), "Save file was not created."

    # 3. Load the messages from the file
    loaded_messages = Messages.load(file_path)
    print("\nMessages loaded from file.")

    assert loaded_messages is not None, "Loaded messages should not be None."

    # Render the loaded messages
    loaded_render = await loaded_messages.render_latest()
    print("\nLoaded Messages Render:")
    print(loaded_render)

    # 4. Compare the original and loaded content
    assert original_render == loaded_render, "Rendered content of original and loaded messages do not match."
    print("\n✅ Assertion passed: Original and loaded message renders are identical.")

    # 5. Check if the loaded object retains its class structure and methods
    print(f"\nType of loaded object: {type(loaded_messages)}")
    assert isinstance(loaded_messages, Messages), "Loaded object is not a Messages instance."

    # Test pop functionality on the loaded object
    popped_item = loaded_messages.pop(0)
    assert isinstance(popped_item, SystemMessage), "Popped item is not a SystemMessage."
    print(f"Popped first message: {popped_item}")

    popped_render = await loaded_messages.render_latest()
    print("\nRender after popping first message from loaded object:")
    print(popped_render)
    assert len(popped_render) == len(original_render) - 1, "Popping a message did not reduce the message count."
    print("✅ Assertion passed: Pop functionality works on the loaded object.")

    # 6. Clean up the test file
    os.remove(file_path)
    print(f"\nCleaned up {file_path}.")

    print("\n--- Test Completed Successfully ---")

if __name__ == "__main__":
    asyncio.run(main())
