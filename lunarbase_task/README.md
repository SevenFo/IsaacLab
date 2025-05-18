# Isaac Lab LunarBase 环境与远程服务

本项目包含一个基于 Isaac Lab 框架封装的强化学习环境 (`LunarBaseEnv`)，以及两种用于远程访问该环境的服务端实现：

1.  一个基于 RPyC 的传统 RPC 服务器，适用于强化学习训练循环。
2.  一个基于官方 `mcp-server-python` SDK 的 **概念性** Model Context Protocol (MCP) 服务器，演示如何将环境功能作为 "工具" 暴露给基于 LLM 的 AI 应用。

## 文件结构

-   `lunar_env.py`: 定义了 `MultiObjectSceneCfg` (场景配置) 和 `LunarBaseEnv` (强化学习环境类)。
-   `rpyc_server.py`: RPyC 服务器实现，用于远程执行环境的 `reset`, `step` 等操作。
-   `mcp_anthropic_server.py`: **概念性** MCP 服务器。使用 Anthropic 的 `mcp-server-python` SDK 来定义和暴露与 `LunarBaseEnv` 交互的 "工具"。它旨在展示如何为 LLM 提供与仿真环境交互的接口，**并非用于传统 RL 的高频 `step/reset` 循环**。
-   `run_local_test.py` (可选，用于本地测试 `LunarBaseEnv` - *请参考上一轮回复中的代码*)
-   `rpyc_client_example.py` (可选，RPyC 客户端示例 - *请参考上一轮回复中的代码*)
-   `mcp_client_conceptual_example.py` (可选，演示如何与 `mcp_anthropic_server.py` 交互的简单 Python HTTP 客户端 - *代码见下文*)

## 环境配置 (`lunar_env.py`)

`LunarBaseEnv` 类实现了标准的 `gymnasium.Env` 接口。

### 主要特性:

-   **并行环境**: 支持创建多个并行的仿真环境 (`num_envs`)。
-   **可配置场景**: 通过 `MultiObjectSceneCfg` 加载和配置场景元素。
    -   **请务必修改 `MultiObjectSceneCfg` 中 `lunar_base` 和 `robot` 的 `usd_path` 为你本地的实际 USD 文件路径！**
    -   **请务必根据你的机械臂和夹爪模型调整 `robot.actuators` 和 `init_state.joint_pos` 中的夹爪关节名称和初始值！**
-   **标准化接口**: `reset()`, `step(actions)`, `observation_space`, `action_space`, `close()`。
-   **动作和观测**: 详细定义见代码。
-   **奖励函数与终止条件**: 需根据具体任务调整。

### 本地测试 `LunarBaseEnv`

使用 `run_local_test.py` (参考之前提供的代码) 在本地直接测试 `LunarBaseEnv`。

**用法:**
```bash
./isaaclab.sh -p path/to/run_local_test.py --num_envs 2
```

## RPyC 服务器 (`rpyc_server.py`)

适用于传统的强化学习训练，允许远程 agent 与仿真环境进行低延迟交互。

### 运行 RPyC 服务器
```bash
./isaaclab.sh -p path/to/rpyc_server.py --host 0.0.0.0 --port 18861 --num_envs 1 --headless
```
### RPyC 客户端示例 (`rpyc_client_example.py`)
(参考之前提供的代码)
```bash
python path/to/rpyc_client_example.py --host <server_host> --port <server_port>
```

## MCP 服务器 (`mcp_anthropic_server.py`)

此服务器使用 Anthropic 官方的 `mcp-server-python` SDK，将 `LunarBaseEnv` 的特定功能封装为符合 Model Context Protocol (MCP) 的 "工具 (Tools)"。这些工具旨在被基于大型语言模型 (LLM) 的 AI 应用（MCP 客户端）调用，以获取环境信息或触发高级别的交互。

**此服务器不适用于传统 RL agent 的高频 `step/reset` 循环。**

### 安装 MCP 相关依赖
```bash
pip install mcp-server-python uvicorn pydantic
```

### 服务器特性:
-   **官方 SDK**: 基于 `mcp-server-python`。
-   **工具定义**: 使用 Pydantic 模型定义工具的输入/输出 schema。
    -   `GetEnvironmentOverviewTool`: 提供环境的概览信息。
    -   `GetObjectStateTool`: 获取指定物体的当前状态。
-   **ASGI 应用**: 通过 `uvicorn` 运行 `MCPApplication`。
-   **`LunarBaseEnv` 集成 (概念性)**:
    -   脚本**尝试**在启动时初始化一个全局的 `LunarBaseEnv` 实例。**这要求此脚本通过 `./isaaclab.sh -p ...` 方式运行，以确保 Isaac Sim 环境正确初始化。**
    -   在生产环境中，`LunarBaseEnv` (Isaac Sim) 和 MCP 服务器 (ASGI) 的生命周期管理和进程间通信需要更复杂和健壮的设计。例如，MCP 工具可能需要通过 RPyC 或其他 IPC 机制与一个独立运行的 `LunarBaseEnv` 进程通信。

### 运行 MCP 服务器
```bash
# 确保 lunar_env.py 中的 USD 路径正确!
./isaaclab.sh -p path/to/mcp_anthropic_server.py --host 0.0.0.0 --port 8080 --env_headless
```
-   服务器启动后，MCP 规范 (OpenAPI JSON) 通常在 `/mcp/spec` 或 `/openapi.json` 端点可用。
-   工具调用端点通常在 `/mcp/tools/{tool_name}`。

### MCP 客户端示例 (`mcp_client_conceptual_example.py`)

这是一个简单的 Python HTTP 客户端，用于演示如何与 `mcp_anthropic_server.py` 交互。它使用 `requests` 库。

```python
# mcp_client_conceptual_example.py
import requests # pip install requests
import json
import argparse

def call_mcp_tool(base_url: str, tool_name: str, tool_input: dict):
    """Sends a POST request to call an MCP tool."""
    tool_url = f"{base_url}/mcp/tools/{tool_name}" # Default tools prefix
    headers = {"Content-Type": "application/json"}
    # The mcp-server-python SDK expects the parameters directly as the JSON body for POST
    # It does not expect a wrapper like {"tool_input": ...} at the HTTP level for POST.
    # The `params` in the Tool's `call` method comes from parsing this JSON body.
    
    print(f"\nCalling MCP Tool: {tool_name}")
    print(f"URL: POST {tool_url}")
    print(f"Input Body: {json.dumps(tool_input)}")

    try:
        response = requests.post(tool_url, json=tool_input, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        print("Response Status Code:", response.status_code)
        # The mcp-server-python SDK directly returns the Tool's Output model as JSON.
        # It does not wrap it in a {"payload": {"tool_output": ...}} structure at HTTP level.
        response_data = response.json()
        print("Response Body:")
        print(json.dumps(response_data, indent=2))
        return response_data
    except requests.exceptions.RequestException as e:
        print(f"Error calling tool {tool_name}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print("Error Response Body:", e.response.json())
            except json.JSONDecodeError:
                print("Error Response Body (not JSON):", e.response.text)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conceptual MCP client example.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080) # Default MCP server port
    args = parser.parse_args()

    server_base_url = f"http://{args.host}:{args.port}"

    print(f"--- Conceptual MCP Client targeting {server_base_url} ---")

    # 1. Get Environment Overview
    # The GetEnvironmentOverviewTool uses GetEnvironmentOverviewParams which is empty
    overview_output = call_mcp_tool(server_base_url, "get_environment_overview", {})
    if overview_output and not overview_output.get("is_simulation_ready", False):
        print("\nSimulation environment on server is not ready. Further tool calls might fail.")
    
    # 2. Get Object State for env_idx 0
    # The GetObjectStateTool uses GetObjectStateParams
    object_state_input = {"env_idx": 0, "object_name": "object"}
    call_mcp_tool(server_base_url, "get_object_state", object_state_input)

    # To test with an invalid env_idx:
    # object_state_input_invalid = {"env_idx": 99}
    # call_mcp_tool(server_base_url, "get_object_state", object_state_input_invalid)

    print("\n--- MCP Client Example Finished ---")
```

**运行 MCP 客户端示例 (在 MCP 服务器运行后):**
```bash
pip install requests
python path/to/mcp_client_conceptual_example.py --host <server_host> --port <server_port>
```

## 二次开发指南

### `LunarBaseEnv` 扩展

-   **自定义奖励**: 修改 `_compute_rewards` 方法以实现你的任务特定奖励。
-   **修改观测/动作空间**: 调整 `__init__` 中 `observation_space` 和 `action_space` 的定义，并确保 `_get_observations` 和 `step` 方法与之匹配。
-   **新的终止条件**: 在 `_check_terminated_truncated` 中添加或修改逻辑。
-   **场景元素**: 在 `MultiObjectSceneCfg` 中添加或修改机器人、物体等。

### RPyC 服务器扩展

-   **暴露新方法**: 在 `LunarEnvRPyCService` 类中，以 `exposed_` 为前缀添加新的方法，这些方法可以调用 `self._env` (即 `LunarBaseEnv` 实例) 的相应功能。
-   **暴露新属性**: 添加新的 `@property` 并以 `exposed_` 命名，以提供更多环境信息。
-   **错误处理**: 增强错误处理和日志记录。
-   **服务配置**: 可以通过修改 `LunarEnvRPyCService` 的 `__init__` 或传递更复杂的配置对象来进一步定制服务行为。

### MCP 服务器 (`mcp_anthropic_server.py`) 扩展
-   **添加新工具**:
    1.  在 `mcp_anthropic_server.py` 中定义新的 Pydantic 模型类用于工具的 `Parameters` (输入) 和 `Output` (输出) schema。
    2.  创建新的工具类，继承自 `mcp_server.Tool`，并实现其 `async def call(self, params: YourParamsModel, context: ToolCallContext) -> YourOutputModel:` 方法。在此方法中与 `LUNAR_ENV_INSTANCE` 交互。
    3.  在脚本末尾，使用 `mcp_app.register_tool(YourNewTool())` 将新工具注册到 `MCPApplication`。
-   **添加 MCP 资源 (`Resource`)**: 如果您有希望作为静态数据源暴露的信息（例如，环境的详细配置文档），可以定义并注册 `mcp_server.Resource` 类的子类。
-   **添加提示模板 (`PromptTemplate`)**: 可以定义和注册 `mcp_server.PromptTemplate`，为 LLM 提供与您的工具交互的预设提示。
-   **环境交互逻辑**: 工具的 `call` 方法中与 `LUNAR_ENV_INSTANCE` 交互的部分是核心。您需要确保这些交互是线程安全的（如果 Uvicorn 使用多 worker 或 Isaac Sim 本身有并发限制），并且能够高效、准确地获取所需数据。
-   **错误处理**: 在工具的 `call` 方法中实现更健壮的错误处理，并可以返回符合您定义的 Output schema 的错误信息，或者抛出 `mcp_server.errors` 中的特定 MCP 错误。
-   **安全性**: 官方 `mcp-server-python` SDK 支持通过 `security_schemes` 和 `authenticators` 来配置认证和授权。对于实际部署，这是非常重要的一步。

## 注意事项
-   **Isaac Sim 环境**: 所有与 Isaac Sim 交互的代码（尤其是环境实例化和仿真步骤）通常需要在 Isaac Sim 的主 Python 解释器上下文中运行，即通过 `./isaaclab.sh` 启动。客户端代码 (如 RPyC 或 MCP 客户端示例) 可以是普通的 Python 程序。
-   **资源管理**: 确保在服务器关闭或环境不再使用时，通过调用 `LunarBaseEnv.close()` 来正确关闭 Isaac Sim 应用，以释放 GPU 和其他资源。服务器脚本中的 `finally`块尝试处理此问题。
-   **USD 路径**: 再次强调，请务必更新代码中所有 `.usd` 文件的路径。
-   **MCP 服务器与 Isaac Sim 的集成**: `mcp_anthropic_server.py` 中的 `LUNAR_ENV_INSTANCE` 管理是一个简化。在生产环境中，确保 Isaac Sim 应用 (通常需要主线程和特定上下文) 和 ASGI Web 服务器 (如 Uvicorn，有自己的事件循环和线程/进程模型) 能够稳定、高效地共存并共享数据是一个关键的架构挑战。
    -   **同进程方案**: 将 Isaac Sim 初始化和 Uvicorn 运行都放在由 `./isaaclab.sh` 启动的同一个 Python 进程中（如示例所示）。这需要仔细处理线程和异步操作，以避免阻塞。
    -   **跨进程方案**: 将 `LunarBaseEnv` 作为一个独立的服务运行（例如，使用我们提供的 RPyC 服务器，或者一个专门的内部服务），然后 MCP 工具通过 IPC (如 RPyC 客户端调用、HTTP 请求到内部服务) 与该环境服务交互。这种方案更复杂但更解耦。
