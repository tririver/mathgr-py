import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mathgr.mcp_server import (
    clear_mathgr_context,
    compute_mathgr,
    evaluate_mathgr,
    get_mathgr_context,
    get_mathgr_manual,
    inspect_mathgr,
    list_mathgr_capabilities,
    load_mathgr_context,
    parse_mathgr,
    save_mathgr_context,
    script_mathgr,
    tex_mathgr,
)
from mathgr.mcp_structured import create_mathgr_context, simplify_mathgr, update_mathgr_context


def test_list_mathgr_capabilities_groups_public_api_for_agents():
    capabilities = list_mathgr_capabilities()

    assert "tensor" in capabilities
    assert "gr" in capabilities
    assert "Simp" in capabilities["tensor"]
    assert "R" in capabilities["gr"]
    assert capabilities["mcp_primary"][0] == "mathgr_compute"
    assert "mathgr_eval" in capabilities["mcp_escape_hatch"]


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


def test_compute_mathgr_accepts_utf8_scalar_names_and_index_labels():
    result = compute_mathgr("Pd(δφ, D1('α'))", output=["str", "tex", "diagnostics"])

    assert result["ok"] is True
    assert result["result"] == "_PdT(δφ, _PdVars(D1('α')))"
    assert result["free"] == ["α"]
    assert result["tex"] == "\\partial_{α} δφ"


def test_compute_mathgr_accepts_restricted_lambda_hooks():
    result = compute_mathgr("Simp(x, hooks=[lambda e: e.xreplace({x: y})])")

    assert result["ok"] is True
    assert result["result"] == "y"


def test_compute_mathgr_explicit_symbols_override_preloaded_names():
    result = compute_mathgr(
        "a**3 / k**2 + H",
        context="mcp_symbol_override",
        symbols=["a", "H", "k"],
        output=["str"],
    )
    momentum = compute_mathgr("k(1)(DN('i'))", context="mcp_momentum_no_override", output=["str"])
    clear_mathgr_context("mcp_symbol_override")
    clear_mathgr_context("mcp_momentum_no_override")

    assert result["ok"] is True
    assert result["result"] == "H + a**3/k**2"
    assert momentum["ok"] is True
    assert momentum["result"] == "k(1, DN('i'))"


def test_compute_mathgr_auto_creates_default_context_and_persists_block_assignments():
    clear_mathgr_context("default")

    result = compute_mathgr(
        """
trace = Dta(U('a'), D('a'))
simplified = Simp(Dta(U('a'), D('b')) * f(U('b')))
result = trace
"""
    )
    reused = compute_mathgr("simplified")
    rendered = tex_mathgr("simplified")
    fetched = get_mathgr_context()
    filtered = get_mathgr_context(name="trace")
    cleared = clear_mathgr_context("default")

    assert result["ok"] is True
    assert result["context"] == "default"
    assert result["result"] == "Dim"
    assert reused["ok"] is True
    assert reused["result"] == "f(U('a'))"
    assert rendered["ok"] is True
    assert rendered["tex"] == "f^{a}"
    assert fetched["expressions"]["trace"] == "Dta(U('a'), D('a'))"
    assert fetched["expressions"]["simplified"] == "Simp(Dta(U('a'), D('b')) * f(U('b')))"
    assert "result" not in fetched["expressions"]
    assert filtered["expressions"] == {"trace": "Dta(U('a'), D('a'))"}
    assert cleared["ok"] is True


def test_compute_mathgr_rehydrates_requested_context_expression_without_stale_result_dependency():
    clear_mathgr_context("mcp_stale_result")
    create_mathgr_context("mcp_stale_result")
    update_mathgr_context(
        "mcp_stale_result",
        expressions={"result": "L2_raw", "L2_raw": "x + 1"},
    )

    result = compute_mathgr("L2_raw", context="mcp_stale_result", output=["str", "diagnostics"])
    clear_mathgr_context("mcp_stale_result")

    assert result["ok"] is True
    assert result["result"] == "x + 1"


def test_compute_mathgr_auto_creates_named_context_and_persists_params():
    clear_mathgr_context("mcp_named_auto")

    result = compute_mathgr(
        """
trace = Dta(U('a'), D('a'))
result = trace
""",
        context="mcp_named_auto",
        index_dims={"U/D": 3},
    )
    reused = compute_mathgr("trace", context="mcp_named_auto")
    fetched = get_mathgr_context("mcp_named_auto")
    clear_mathgr_context("mcp_named_auto")

    assert result["ok"] is True
    assert result["result"] == "3"
    assert reused["ok"] is True
    assert reused["result"] == "3"
    assert fetched["declarations"]["index_dims"] == {"U/D": 3}
    assert fetched["expressions"]["trace"] == "Dta(U('a'), D('a'))"


def test_compute_mathgr_block_uses_last_expression_without_result_assignment():
    clear_mathgr_context("mcp_last_expr")

    result = compute_mathgr(
        """
trace = Dta(U('a'), D('a'))
Simp(trace)
""",
        context="mcp_last_expr",
        index_dims={"U/D": 4},
    )
    fetched = get_mathgr_context("mcp_last_expr")
    clear_mathgr_context("mcp_last_expr")

    assert result["ok"] is True
    assert result["result"] == "4"
    assert fetched["expressions"]["trace"] == "Dta(U('a'), D('a'))"


def test_compute_mathgr_block_persists_metric_symmetry_and_module_aliases():
    clear_mathgr_context("mcp_declaration_block")

    result = compute_mathgr(
        """
gMcpBlock = tensor("gMcpBlock")
UseMetric(gMcpBlock, (U, D))
F = tensor("FMcpAnti")
DeclareSym(F, (D, D), Antisymmetric((1, 2)))
alias_value = frwadm.Simp(Dta(U('a'), D('b')) * f(U('b')))
result = Simp(F(D('c'), D('c'))) + alias_value
""",
        context="mcp_declaration_block",
    )
    metric = compute_mathgr("Metric", context="mcp_declaration_block")
    antisym = compute_mathgr("Simp(F(D('c'), D('c')))", context="mcp_declaration_block")
    fetched = get_mathgr_context("mcp_declaration_block")
    clear_mathgr_context("mcp_declaration_block")

    assert result["ok"] is True
    assert result["result"] == "f(U('a'))"
    assert metric["ok"] is True
    assert metric["result"] == "gMcpBlock"
    assert antisym["ok"] is True
    assert antisym["result"] == "0"
    assert fetched["declarations"]["metric"] == {"head": "gMcpBlock", "indices": "U/D"}
    assert fetched["declarations"]["symmetries"] == [
        {"head": "F", "signature": ["D", "D"], "symmetry": "Antisymmetric", "slots": [1, 2]}
    ]


def test_context_save_and_load_round_trips_json(tmp_path):
    clear_mathgr_context("mcp_save_ctx")
    path = tmp_path / "ctx.json"

    compute_mathgr(
        """
trace = Dta(U('a'), D('a'))
result = trace
""",
        context="mcp_save_ctx",
        index_dims={"U/D": 5},
    )
    saved = save_mathgr_context("mcp_save_ctx", path=str(path))
    clear_mathgr_context("mcp_save_ctx")
    loaded = load_mathgr_context(path=str(path), context="mcp_save_ctx")
    result = compute_mathgr("trace", context="mcp_save_ctx")
    clear_mathgr_context("mcp_save_ctx")

    assert saved["ok"] is True
    assert loaded["ok"] is True
    assert result["ok"] is True
    assert result["result"] == "5"
    saved_json = json.loads(path.read_text(encoding="utf-8"))
    assert saved_json["schema_version"] == 1
    assert saved_json["context"] == "mcp_save_ctx"


def test_script_mathgr_context_exports_stored_expressions():
    clear_mathgr_context("mcp_script_context")
    compute_mathgr(
        """
trace = Dta(U('a'), D('a'))
simplified = Simp(trace)
result = simplified
""",
        context="mcp_script_context",
    )

    script = script_mathgr(context="mcp_script_context")
    clear_mathgr_context("mcp_script_context")

    assert script["ok"] is True
    assert "trace = Dta(U('a'), D('a'))" in script["python"]
    assert "simplified = Simp(trace)" in script["python"]
    assert "result =" not in script["python"]


def test_compute_mathgr_rejects_unsafe_block_syntax():
    imported = compute_mathgr("import os\nresult = 1", context="mcp_unsafe")
    looped = compute_mathgr("for value in [1]:\n    result = value", context="mcp_unsafe")
    opened = compute_mathgr("result = open('x')", context="mcp_unsafe")
    clear_mathgr_context("mcp_unsafe")

    assert imported["ok"] is False
    assert "Import" in imported["stderr"] or "Unsupported syntax" in imported["stderr"]
    assert looped["ok"] is False
    assert "For" in looped["stderr"] or "Unsupported syntax" in looped["stderr"]
    assert opened["ok"] is False
    assert "open" in opened["stderr"]


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
    assert "first-choice tool" in manual["content"]


def test_get_mathgr_manual_query_returns_partial_matches_instead_of_empty_content():
    manual = get_mathgr_manual(query="lambda hooks")

    assert manual["ok"] is True
    assert manual["content"]
    assert "hooks" in manual["content"].lower()


def test_compute_mathgr_supports_custom_expansion_symbol_directly():
    coefficient = compute_mathgr("TSeries((1 + eps*x)**3, (eps, 0, 2)).coeff(eps, 2)")

    assert coefficient["ok"] is True
    assert coefficient["result"] == "3*x**2"


def test_compute_mathgr_respects_timeout_for_expensive_expression():
    result = compute_mathgr("Simp((1 + x)**3000)", timeout_seconds=0.1, output=["str"])

    assert result["ok"] is False
    assert "timed out" in result["stderr"]


def test_simplify_mathgr_store_as_uses_default_context():
    clear_mathgr_context("default")

    result = simplify_mathgr("Dta(UP('a'), DN('a'))", store_as="trace")
    fetched = get_mathgr_context()
    clear_mathgr_context("default")

    assert result["ok"] is True
    assert fetched["expressions"]["trace"] == "DefaultDim"


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
        "mathgr_eval",
        "mathgr_run_python",
        "mathgr_manual",
        "mathgr_parse",
        "mathgr_compute",
        "mathgr_inspect",
        "mathgr_tex",
        "mathgr_context_get",
        "mathgr_context_clear",
        "mathgr_context_save",
        "mathgr_context_load",
        "mathgr_script",
    } <= registered
    assert "mathgr_context_create" not in registered
    assert "mathgr_context_update" not in registered
    assert "mathgr_topic" not in registered


def test_mcp_guidance_prefers_expr_only_default_context_calls():
    async def read_guidance():
        params = StdioServerParameters(command="uv", args=["run", "mathgr-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
                descriptions = {tool.name: tool.description or "" for tool in tools.tools}
                return initialized.instructions or "", descriptions

    instructions, descriptions = asyncio.run(read_guidance())

    assert "pass only the expr string" in instructions
    assert "omit context" in instructions
    assert "JSON is only the transport format" in instructions
    assert "pass only expr" in descriptions["mathgr_compute"]
    assert "Do not call by default" in descriptions["mathgr_context_clear"]
