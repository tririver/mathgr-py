import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mathgr.mcp_server import (
    evaluate_mathgr,
    get_mathgr_topic,
    list_mathgr_capabilities,
)


def test_list_mathgr_capabilities_groups_public_api_for_agents():
    capabilities = list_mathgr_capabilities()

    assert "tensor" in capabilities
    assert "gr" in capabilities
    assert "Simp" in capabilities["tensor"]
    assert "R" in capabilities["gr"]


def test_get_mathgr_topic_returns_quick_reference_text():
    topic = get_mathgr_topic("decomp")

    assert topic["topic"] == "decomp"
    assert "Decomp0i" in topic["content"]
    assert "UTot" in topic["content"]


def test_evaluate_mathgr_runs_trusted_snippet_with_mathgr_imports():
    result = evaluate_mathgr(
        """
f = tensor("fMcp")
expr = Dta(UP("a"), DN("b")) * f(UP("b"))
result = Simp(expr)
"""
    )

    assert result["ok"] is True
    assert result["result"] == "fMcp(UP('a'))"
    assert result["stdout"] == ""


def test_evaluate_mathgr_captures_stdout_and_json_result():
    result = evaluate_mathgr(
        """
import json
print("hello")
result = {"simplified": str(Simp(Dta(UP("a"), DN("a"))))}
"""
    )

    assert result["ok"] is True
    assert result["stdout"] == "hello\n"
    assert json.loads(result["result"]) == {"simplified": "DefaultDim"}


def test_evaluate_mathgr_reports_snippet_errors():
    result = evaluate_mathgr("raise ValueError('bad snippet')")

    assert result["ok"] is False
    assert "ValueError: bad snippet" in result["stderr"]


def test_evaluate_mathgr_blocks_unapproved_imports():
    result = evaluate_mathgr("import mathgr.mcp_server\nresult = mathgr.mcp_server.import_module('os').getcwd()")

    assert result["ok"] is False
    assert "not available" in result["stderr"]


def test_evaluate_mathgr_times_out_nonterminating_snippet():
    result = evaluate_mathgr("while True:\n    pass", timeout_seconds=1)

    assert result["ok"] is False
    assert "timed out" in result["stderr"]


def test_evaluate_mathgr_restores_mathgr_global_state_between_calls():
    mutate = evaluate_mathgr(
        """
u, d = declare_idx("McpStateU", "McpStateD", dim=4, index_set=LatinIdx)
metric = tensor("gMcpState")
UseMetric(metric, (u, d))
result = mathgr.Metric
"""
    )
    probe = evaluate_mathgr("result = mathgr.Metric")

    assert mutate["ok"] is True
    assert mutate["result"] == "gMcpState"
    assert probe["ok"] is True
    assert probe["result"] == "g"


def test_evaluate_mathgr_exposes_live_metric_after_use_metric_within_call():
    result = evaluate_mathgr(
        """
u, d = declare_idx("McpLiveU", "McpLiveD", dim=4, index_set=LatinIdx)
metric = tensor("gMcpLive")
UseMetric(metric, (u, d))
result = Metric
"""
    )

    assert result["ok"] is True
    assert result["result"] == "gMcpLive"


def test_create_mcp_registers_mathgr_tools():
    async def list_tool_names():
        params = StdioServerParameters(command="uv", args=["run", "mathgr-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return {tool.name for tool in tools.tools}

    registered = asyncio.run(list_tool_names())
    assert {"mathgr_capabilities", "mathgr_topic", "mathgr_eval"} <= registered
