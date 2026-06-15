import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mathgr.mcp_server import (
    clear_mathgr_context,
    compute_mathgr,
    create_mathgr_context,
    evaluate_mathgr,
    get_mathgr_context,
    get_mathgr_manual,
    get_mathgr_topic,
    inspect_mathgr,
    list_mathgr_capabilities,
    parse_mathgr,
    script_mathgr,
    tex_mathgr,
    update_mathgr_context,
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


def test_parse_mathgr_auto_declares_python_like_indices_tensors_and_symbols():
    result = parse_mathgr("Dta(U('a'), D('b')) * f(U('b')) + x")

    assert result["ok"] is True
    assert result["auto_declarations"]["indices"] == [{"up": "U", "down": "D", "dim": "Dim"}]
    assert result["auto_declarations"]["tensors"] == ["f"]
    assert "x" in result["auto_declarations"]["symbols"]
    assert "Dim = sp.Symbol('Dim')" in result["python"]
    assert "U, D = declare_idx('U', 'D', dim=Dim" in result["python"]


def test_compute_mathgr_uses_auto_declarations_and_explicit_simp_call():
    contraction = compute_mathgr("Simp(Dta(U('a'), D('b')) * f(U('b')))")
    trace = compute_mathgr("Dta(U('a'), D('a'))")

    assert contraction["ok"] is True
    assert contraction["result"] == "f(U('a'))"
    assert contraction["free"] == ["a"]
    assert contraction["dummy"] == []
    assert "f^{a}" in contraction["tex"]
    assert trace["ok"] is True
    assert trace["result"] == "Dim"


def test_compute_mathgr_allows_dimension_overrides_and_context_expressions():
    context = create_mathgr_context("mcp_test_context")
    update = update_mathgr_context(
        context["context"],
        declarations={"index_dims": {"U/D": 2}},
        expressions={"trace": "Dta(U('a'), D('a'))"},
    )
    result = compute_mathgr("trace", context=context["context"])
    fetched = get_mathgr_context(context["context"], name="trace")
    cleared = clear_mathgr_context(context["context"])

    assert update["ok"] is True
    assert result["ok"] is True
    assert result["result"] == "2"
    assert fetched["expressions"] == {"trace": "Dta(U('a'), D('a'))"}
    assert cleared["ok"] is True


def test_compute_identity_and_inspect_mathgr_return_index_diagnostics():
    comparison = compute_mathgr("Simp(Dta(U('a'), D('b')) * f(U('b')) - f(U('a')))")
    inspected = inspect_mathgr("f(U('a'), D('b')) * g(U('b'))")

    assert comparison["ok"] is True
    assert comparison["result"] == "0"
    assert inspected["ok"] is True
    assert inspected["free"] == ["a"]
    assert inspected["dummy"] == ["b"]
    assert inspected["tensor_heads"] == ["f", "g"]


def test_compute_mathgr_runs_ordinary_mathgr_functions_and_tex_script_helpers():
    derivative = compute_mathgr("Pd(x*y, D('i'))")
    rewritten = compute_mathgr("ReplaceAll(x + x, [(x, y)])")
    decomposed = compute_mathgr("Decomp0i(f(DTot('a')) * f(UTot('a')))")
    coefficient = compute_mathgr("OO(2)((1 + Eps*x)**3)")
    ibp = compute_mathgr("Ibp(y * Pd(x, D('i')))")
    transformed = compute_mathgr("Simp(Dta(U('a'), D('b')) * f(U('b')))")
    tex = tex_mathgr("f(U('a'), D('b'))")
    script = script_mathgr("Dta(U('a'), D('a'))", operation="simplify")
    manual = get_mathgr_manual(section="MCP Server")

    assert derivative["ok"] is True
    assert "PdT(x" in derivative["result"]
    assert "PdT(y" in derivative["result"]
    assert rewritten["ok"] is True
    assert rewritten["result"] == "2*y"
    assert decomposed["ok"] is True
    assert "f(DE(0))*f(UE(0))" in decomposed["result"]
    assert "f(DN('a'))*f(UP('a'))" in decomposed["result"]
    assert coefficient["ok"] is True
    assert coefficient["result"] == "3*x**2"
    assert ibp["ok"] is True
    assert "PdHold" in ibp["result"]
    assert transformed["ok"] is True
    assert transformed["result"] == "f(U('a'))"
    assert tex["ok"] is True
    assert tex["tex"] == "f^{a}_{b}"
    assert script["ok"] is True
    assert "result = Simp(" in script["python"]
    assert manual["ok"] is True
    assert "mathgr_compute" in manual["content"]


def test_compute_mathgr_supports_custom_expansion_symbol_directly():
    coefficient = compute_mathgr("TSeries((1 + eps*x)**3, (eps, 0, 2)).coeff(eps, 2)")

    assert coefficient["ok"] is True
    assert coefficient["result"] == "3*x**2"


def test_create_mcp_registers_mathgr_tools():
    async def list_tool_names():
        params = StdioServerParameters(command="uv", args=["run", "mathgr-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return {tool.name for tool in tools.tools}

    registered = asyncio.run(list_tool_names())
    assert {
        "mathgr_capabilities",
        "mathgr_topic",
        "mathgr_eval",
        "mathgr_run_python",
        "mathgr_manual",
        "mathgr_parse",
        "mathgr_compute",
        "mathgr_inspect",
        "mathgr_tex",
        "mathgr_context_create",
        "mathgr_context_update",
        "mathgr_context_get",
        "mathgr_context_clear",
        "mathgr_script",
    } <= registered
