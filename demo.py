# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import pathlib
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_community.chat_models import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain.tools.base import ToolException
from typing import List, Dict, Any, Optional
import json

import mcp_connector

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

def get_model():
    # Force reload environment variables
    load_dotenv(override=True)
    
    # Get model name with debug output
    model_name = os.getenv('MODEL_CHOICE')
    if not model_name:
        model_name = 'qwen2.5:14b-instruct'
        logger.warning(f"MODEL_CHOICE not found in environment, using default: {model_name}")
    else:
        logger.info(f"Found MODEL_CHOICE in environment: {model_name}")
    
    # Get base URL with debug output
    base_url = os.getenv('BASE_URL')
    if not base_url:
        base_url = 'http://localhost:11434'
        logger.warning(f"BASE_URL not found in environment, using default: {base_url}")
    else:
        logger.info(f"Found BASE_URL in environment: {base_url}")
        # Remove /v1 from the base URL if present
        if base_url.endswith('/v1'):
            old_url = base_url
            base_url = base_url.rstrip('/v1')
            logger.info(f"Removed /v1 from base URL for ChatOllama compatibility: {old_url} -> {base_url}")
    
    logger.info(f"Initializing Ollama with model: {model_name} at {base_url}")
    
    try:
        # Create the ChatOllama instance
        llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7
        )
        
        # Test the connection
        logger.info("Testing Ollama connection...")
        test_response = llm.invoke("Test message - are you working?")
        logger.info(f"Ollama test response received: {test_response.content}")
        
        return llm
    except Exception as e:
        logger.error(f"Error initializing Ollama: {str(e)}")
        logger.error(f"Full error details: {e.__class__.__name__}")
        if "404" in str(e):
            logger.info(f"\nTrying to pull the model first...")
            try:
                import requests
                response = requests.post(f"{base_url}/api/pull", json={"name": model_name})
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model {model_name}")
                    # Try creating the LLM again
                    llm = ChatOllama(
                        model=model_name,
                        base_url=base_url,
                        streaming=True,
                        callbacks=[StreamingStdOutCallbackHandler()],
                        temperature=0.7
                    )
                    return llm
            except Exception as pull_error:
                logger.error(f"Error pulling model: {str(pull_error)}")
        raise

class MCPToolWrapper:
    def __init__(self, mcp_client, tool_spec: Dict[str, Any]):
        self.mcp_client = mcp_client
        self.tool_spec = tool_spec
        self.name = tool_spec['name']
        self.description = tool_spec['description']
        self.parameters = tool_spec['parameters']

    async def execute(self, **kwargs) -> Any:
        """Execute the MCP tool with the given parameters."""
        try:
            result = await self.mcp_client.execute_tool(
                self.name,
                kwargs
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            raise ToolException(f"Error executing tool {self.name}: {str(e)}")

def convert_mcp_tool_to_langchain(mcp_client, mcp_tool: Dict[str, Any]) -> Tool:
    """Convert an MCP tool to a LangChain tool."""
    wrapper = MCPToolWrapper(mcp_client, mcp_tool)
    
    # Create a description that includes parameter information
    param_desc = []
    for name, param in mcp_tool['parameters'].get('properties', {}).items():
        required = name in mcp_tool['parameters'].get('required', [])
        desc = param.get('description', '')
        type_info = param.get('type', 'string')
        param_desc.append(f"- {name} ({type_info}{'*' if required else ''}): {desc}")
    
    full_description = f"{mcp_tool['description']}\n\nParameters:\n" + "\n".join(param_desc)
    
    return Tool(
        name=mcp_tool['name'],
        description=full_description,
        func=wrapper.execute,
        coroutine=wrapper.execute,
        args_schema=None  # Let LangChain handle the schema
    )

async def get_langchain_agent():
    print("Step 1: Initializing MCP client...")
    # Initialize MCP client and get tools
    client = mcp_connector.MCPClient()
    client.load_servers(str(CONFIG_FILE))
    
    print("Step 2: Starting MCP client and getting tools...")
    mcp_tools = await client.start()
    print(f"Retrieved {len(mcp_tools)} MCP tools")
    
    # Convert MCP tools to LangChain tools
    print("Step 3: Converting MCP tools to LangChain tools...")
    tools = [convert_mcp_tool_to_langchain(client, tool) for tool in mcp_tools]
    print(f"Converted {len(tools)} tools")
    
    # Initialize the LangChain agent
    print("Step 4: Initializing Ollama LLM...")
    llm = get_model()
    
    print("Step 5: Creating LangChain agent...")
    
    # Create the prompt template with a simpler format
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that uses tools to accomplish tasks. You have access to the following tools:

{tools}

To use a tool, format your response like this:

Thought: what you're thinking
Action: one of [{tool_names}]
Action Input: {{"param1": "value1", "param2": "value2"}}

After using tools, end with:
Thought: I know the answer
Final Answer: your response"""),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True
    )
    print("Agent creation complete!")
    
    return client, agent_executor, tools

async def main():
    print("=== LangChain MCP CLI Chat ===")
    print("Type 'exit' to quit the chat")
    
    print("Initializing agent...")
    # Initialize the agent
    mcp_client = None
    try:
        mcp_client, agent, tools = await get_langchain_agent()
        print("Agent initialized successfully!")
        
        console = Console()
        chat_history = []  # Initialize empty chat history
        
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    # Run the agent with empty scratchpad
                    response = await agent.ainvoke({
                        "input": user_input,
                        "tools": tools,  # Now tools is in scope
                        "tool_names": [tool.name for tool in tools],  # List of tool names
                        "agent_scratchpad": []  # Start with empty list for scratchpad
                    })
                    
                    # Update the display
                    if isinstance(response, dict):
                        output = response.get("output", str(response))
                        # If we have intermediate steps, format them nicely
                        if "intermediate_steps" in response:
                            steps = []
                            for action, observation in response["intermediate_steps"]:
                                steps.append(f"Thought: {action.log}\nObservation: {observation}")
                            if steps:
                                output = "\n".join(steps) + f"\n\nFinal Answer: {output}"
                    else:
                        output = str(response)
                    live.update(Markdown(output))
                    
                    # Add the interaction to chat history
                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=output))
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
                import traceback
                print("Full error details:")
                traceback.print_exc()
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
    finally:
        # Ensure proper cleanup of MCP client resources when exiting
        if mcp_client:
            try:
                # Cancel any pending tasks
                for task in asyncio.all_tasks():
                    if task != asyncio.current_task():
                        task.cancel()
                
                # Wait briefly for tasks to cancel
                await asyncio.sleep(0.1)
                
                # Cleanup MCP client
                await mcp_client.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        # Force cleanup of any remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel() 
