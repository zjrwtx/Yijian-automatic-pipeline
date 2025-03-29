
import asyncio
from pathlib import Path

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import MCPToolkit
from camel.types import ModelPlatformType, ModelType


async def main():
    config_path = Path(__file__).parent / "mcp_servers_config.json"
    mcp_toolkit = MCPToolkit(config_path=str(config_path))

    # Connect to all MCP servers.
    await mcp_toolkit.connect()

    sys_msg = "You are a helpful assistant"
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )
    camel_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=[*mcp_toolkit.get_tools()],
    )
    user_msg = "找胃癌相关的paper，总结研究趋势"
    response = await camel_agent.astep(user_msg)
    print(response.msgs[0].content)
    print(response.info['tool_calls'])

    # Disconnect from all MCP servers and clean up resources.
    await mcp_toolkit.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
