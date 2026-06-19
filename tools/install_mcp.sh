#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/install_mcp.sh [OPTIONS]

Install MathGR MCP for Codex and/or Claude Code.

Default mode is portable: store a uvx GitHub source in user MCP config so the
server works from any directory and does not depend on this checkout path.

Options:
  --portable          Use uvx --from git+https://github.com/tririver/mathgr-py (default)
  --dev-local         Use uv --directory "$REPO" run mathgr-mcp; repo path must persist
  --client CLIENT     codex, claude, or both (default: both)
  --from SOURCE       Portable uvx source (default: git+https://github.com/tririver/mathgr-py)
  --python VERSION    Python version for uvx (default: 3.14)
  --repo PATH         Repo path for --dev-local (default: parent of this script)
  --uv PATH           uv path for --dev-local (default: /usr/local/bin/uv or PATH)
  --uvx PATH          uvx path for --portable (default: /usr/local/bin/uvx or PATH)
  --dry-run           Print commands without changing MCP config
  -h, --help          Show this help

Examples:
  tools/install_mcp.sh
  tools/install_mcp.sh --client codex
  tools/install_mcp.sh --dev-local --repo /opt/mathgr-py
  tools/install_mcp.sh --from git+https://github.com/tririver/mathgr-py@v0.1.0
EOF
}

default_tool() {
  local preferred="$1"
  local fallback="$2"

  if [[ -x "$preferred" ]]; then
    printf '%s\n' "$preferred"
    return 0
  fi

  if command -v "$fallback" >/dev/null 2>&1; then
    command -v "$fallback"
    return 0
  fi

  printf '%s\n' "$preferred"
}

need_arg() {
  local option="$1"
  local value="${2:-}"

  if [[ -z "$value" ]]; then
    printf '%s requires a value.\n' "$option" >&2
    exit 2
  fi
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/.." && pwd)"

mode="portable"
client="both"
source="${MATHGR_MCP_SOURCE:-git+https://github.com/tririver/mathgr-py}"
python_version="${MATHGR_MCP_PYTHON:-3.14}"
uv_path="${MATHGR_UV:-$(default_tool /usr/local/bin/uv uv)}"
uvx_path="${MATHGR_UVX:-$(default_tool /usr/local/bin/uvx uvx)}"
dry_run=0
installed_any=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --portable)
      mode="portable"
      shift
      ;;
    --dev-local)
      mode="dev-local"
      shift
      ;;
    --client)
      need_arg "$1" "${2:-}"
      client="${2:-}"
      shift 2
      ;;
    --from)
      need_arg "$1" "${2:-}"
      source="${2:-}"
      shift 2
      ;;
    --python)
      need_arg "$1" "${2:-}"
      python_version="${2:-}"
      shift 2
      ;;
    --repo)
      need_arg "$1" "${2:-}"
      repo="${2:-}"
      shift 2
      ;;
    --uv)
      need_arg "$1" "${2:-}"
      uv_path="${2:-}"
      shift 2
      ;;
    --uvx)
      need_arg "$1" "${2:-}"
      uvx_path="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$client" in
  codex|claude|both) ;;
  *)
    printf 'Invalid --client: %s\n' "$client" >&2
    exit 2
    ;;
esac

if [[ "$mode" == "portable" ]]; then
  if [[ -z "$source" || -z "$python_version" || -z "$uvx_path" ]]; then
    printf 'Portable install needs --from, --python, and --uvx values.\n' >&2
    exit 2
  fi
  if [[ "$dry_run" -eq 0 && ! -x "$uvx_path" ]]; then
    printf 'uvx not executable: %s\n' "$uvx_path" >&2
    exit 1
  fi
  mcp_cmd=(env -u VIRTUAL_ENV -u PYTHONPATH "$uvx_path" --python "$python_version" --from "$source" mathgr-mcp)
else
  if [[ -z "$repo" || -z "$uv_path" ]]; then
    printf 'Dev-local install needs --repo and --uv values.\n' >&2
    exit 2
  fi
  if [[ "$dry_run" -eq 0 && ! -d "$repo" ]]; then
    printf 'Repo path does not exist: %s\n' "$repo" >&2
    exit 1
  fi
  if [[ "$dry_run" -eq 0 && ! -x "$uv_path" ]]; then
    printf 'uv not executable: %s\n' "$uv_path" >&2
    exit 1
  fi
  mcp_cmd=(env -u VIRTUAL_ENV -u PYTHONPATH "$uv_path" --directory "$repo" run mathgr-mcp)
fi

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

install_codex() {
  if ! command -v codex >/dev/null 2>&1; then
    if [[ "$client" == "codex" ]]; then
      printf 'codex command not found.\n' >&2
      exit 1
    fi
    return 0
  fi

  run_cmd codex mcp remove mathgr || true
  run_cmd codex mcp add mathgr -- "${mcp_cmd[@]}"
  installed_any=1
  run_cmd codex mcp get mathgr
}

install_claude() {
  if ! command -v claude >/dev/null 2>&1; then
    if [[ "$client" == "claude" ]]; then
      printf 'claude command not found.\n' >&2
      exit 1
    fi
    return 0
  fi

  run_cmd claude mcp remove mathgr || true
  run_cmd claude mcp add --scope user mathgr -- "${mcp_cmd[@]}"
  installed_any=1
  run_cmd claude mcp get mathgr
}

case "$client" in
  codex)
    install_codex
    ;;
  claude)
    install_claude
    ;;
  both)
    install_codex
    install_claude
    ;;
esac

if [[ "$installed_any" -eq 0 ]]; then
  printf 'No supported agent CLI found. Install codex or claude, or pass --client for the one you have.\n' >&2
  exit 1
fi

printf 'MathGR MCP install command:'
printf ' %q' "${mcp_cmd[@]}"
printf '\n'
