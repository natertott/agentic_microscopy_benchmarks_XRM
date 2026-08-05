"""
Shared utilities for multi-agent LLM evaluation analysis.

This module is intentionally boring: one canonical loader, one canonical export path,
and reusable helpers for sweep plots, RAG analysis, surrogate modeling, and active
learning recommendations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _as_list(x: Any) -> list:
    """Return x as a list, accepting native lists, JSON-encoded lists, scalars, or NaN."""
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    if isinstance(x, list):
        return x
    if isinstance(x, tuple) or isinstance(x, set):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [x]
        # Common CSV-ish fallback.
        if "," in s:
            return [part.strip() for part in s.split(",") if part.strip()]
        return [x]
    return [x]


def _unique_preserve_order(items: Iterable[Any]) -> list:
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, float) and math.isnan(item):
            continue
        key = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else item
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)


def _safe_bool_passed(status: Any, passed: Any = None) -> bool:
    if isinstance(passed, bool):
        return passed
    if passed in (0, 1):
        return bool(passed)
    return str(status).lower() == "passed"


def _hash_config(row: Mapping[str, Any], cfg_cols: Sequence[str] | None = None) -> str:
    if cfg_cols is None:
        cfg_cols = sorted(k for k in row if k.startswith("cfg_"))
    payload = {k: row.get(k) for k in cfg_cols}
    return hashlib.sha1(_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def _numeric_if_possible(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    # Keep categorical strings if numeric conversion loses almost everything.
    if converted.notna().sum() >= max(1, int(0.8 * series.notna().sum())):
        return converted
    return series


def _deep_merge_config(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Deep-merge two config dicts, with `override`'s leaves winning where present.

    Used to combine a per-result `config_snapshot` with the suite-level one it
    was derived from. Some fields (e.g. `agent.graph_topology`) are only ever
    written into the *suite-level* snapshot, while each individual result's
    own snapshot is a narrower, independently-serialized copy that may not
    include every key the suite-level one does. A shallow `{**a, **b}` merge
    at the top level (or a naive "prefer flattened result over flattened
    suite") would let that narrower per-result `agent` sub-dict silently wipe
    out a real suite-level value. Merging at the leaf level instead means a
    key is only overridden when the override actually specifies it.
    """
    base = base or {}
    override = override or {}
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_config(out[k], v)
        else:
            out[k] = v
    return out


def _flatten_dict(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


# -----------------------------------------------------------------------------
# Canonical config extraction
# -----------------------------------------------------------------------------

#: Older benchmark suites predate `agent.graph_topology` in the config
#: snapshot. Rather than leaving those runs with a null architecture (which
#: would silently drop them from any framework-comparison groupby), backfill
#: with the topology actually in use at the time: every suite predating this
#: field was a 2-agent (supervisor + worker) run. This is the ONE place that
#: assumption lives, so every notebook/script going through
#: `flatten_config_snapshot` gets it for free.
DEFAULT_GRAPH_TOPOLOGY: str = "two_agent"


def flatten_config_snapshot(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Flatten a suite `config_snapshot` into stable `cfg_*` columns.

    Keeps the names already used by the regression JSONL exports where possible,
    while also retaining additional namespaced fields for future analysis.

    `cfg_graph_topology` is backfilled with `DEFAULT_GRAPH_TOPOLOGY` when the
    source suite predates that field, so older result files still load and
    still participate in architecture-based grouping/plotting.
    """
    config = config or {}
    model = config.get("model", {}) or {}
    rag = config.get("rag", {}) or {}
    message = config.get("message_management", {}) or {}
    agent = config.get("agent", {}) or {}
    tools = config.get("tools_enabled", {}) or {}
    testing = config.get("testing", {}) or {}
    metrics = config.get("metrics", {}) or {}

    out = {
        "cfg_graph_topology": agent.get("graph_topology") or DEFAULT_GRAPH_TOPOLOGY,
        "cfg_llm_provider": model.get("provider"),
        "cfg_llm_model": model.get("model"),
        "cfg_llm_temperature": model.get("temperature"),
        "cfg_embedding_model": rag.get("embedding_model"),
        "cfg_episodic_top_k": rag.get("episodic_top_k"),
        "cfg_contextual_top_k": rag.get("contextual_top_k"),
        "cfg_chunk_size": rag.get("chunk_size"),
        "cfg_chunk_overlap": rag.get("chunk_overlap"),
        "cfg_similarity_threshold": rag.get("similarity_threshold"),
        "cfg_rag_cache_enabled": rag.get("cache_enabled"),
        "cfg_max_messages": message.get("max_messages"),
        "cfg_max_tokens": message.get("max_tokens"),
        "cfg_pruning_strategy": message.get("pruning_strategy"),
        "cfg_preserve_recent_messages": message.get("preserve_recent_messages"),
        "cfg_agent_timeout": agent.get("timeout"),
        "cfg_max_retries": agent.get("max_retries"),
        "cfg_recursion_limit": agent.get("recursion_limit"),
        "cfg_test_mock_mode": testing.get("mock_mode"),
        "cfg_test_timeout_seconds": testing.get("timeout_seconds"),
        "cfg_cost_per_1k_input": metrics.get("cost_per_1k_input"),
        "cfg_cost_per_1k_output": metrics.get("cost_per_1k_output"),
    }
    for name, enabled in tools.items():
        out[f"cfg_tool_{name}_enabled"] = enabled

    # Keep this deliberately narrow. Directories, run IDs, and test names are metadata,
    # not hyperparameters, and including them creates leakage/redundant design columns.
    return out


def config_label_table(
    runs: pd.DataFrame,
    max_values: int = 4,
    always_include: Sequence[str] = ("cfg_llm_model", "cfg_llm_temperature", "cfg_episodic_top_k"),
) -> tuple[pd.DataFrame, list[str]]:
    """Build human-readable config labels from the *varying* cfg_ columns.

    Columns in ``always_include`` are kept in the label (and placed first)
    regardless of cardinality, so high-cardinality keys like the LLM model
    are never silently dropped by the ``max_values`` filter.
    """
    cfg_cols = sorted([c for c in runs.columns if c.startswith("cfg_")])
    varying = [c for c in cfg_cols if runs[c].nunique(dropna=False) > 1]
    # Priority columns first (if they vary), then the rest in alphabetical order.
    priority = [c for c in always_include if c in varying]
    ordered = priority + [c for c in varying if c not in priority]
    representatives = runs.drop_duplicates("config_hash").copy()

    def label(row: pd.Series) -> str:
        parts = []
        for c in ordered:
            vals = runs[c].dropna().unique()
            if c in priority or len(vals) <= max_values:
                parts.append(f"{c.replace('cfg_', '')}={row.get(c)}")
        return " | ".join(parts) if parts else str(row.get("config_hash", ""))[:8]

    representatives["config_label"] = representatives.apply(label, axis=1)
    return representatives[["config_hash", "config_label"] + cfg_cols], varying


# -----------------------------------------------------------------------------
# Loaders and normalized tables
# -----------------------------------------------------------------------------

def _suite_files(root_or_files: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(root_or_files, (str, Path)):
        p = Path(root_or_files)
        if p.is_file():
            return [p]
        files = sorted(p.rglob("suite_*.json"))
        # Also allow flat result names that do not begin with suite_ but have suite shape.
        files += sorted(p.rglob("suite_results*.json"))
        # also just grab every .json file
        return sorted(set(files))
    return [Path(p) for p in root_or_files]


def _jsonl_files(root_or_files: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(root_or_files, (str, Path)):
        p = Path(root_or_files)
        if p.is_file():
            return [p]
        return sorted(p.rglob("regression_data_*.jsonl"))
    return [Path(p) for p in root_or_files]


def extract_rag_entry_ids(result: Mapping[str, Any]) -> list[str]:
    explicit = _as_list(result.get("rag_entry_ids"))
    from_events = []
    for ev in _as_list(result.get("rag_retrievals")):
        if isinstance(ev, Mapping):
            from_events.extend(_as_list(ev.get("retrieved_entry_ids")))
    return _unique_preserve_order([str(x) for x in explicit + from_events if x not in (None, "")])


def suite_file_to_tables(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one suite JSON into run-level rows and retrieval-event rows."""
    path = Path(path)
    suite = json.loads(path.read_text(encoding="utf-8"))
    suite_config_snapshot = suite.get("config_snapshot")
    suite_meta = {
        "suite_run_id": suite.get("suite_run_id"),
        "suite_name": suite.get("suite_name"),
        "suite_config_hash": suite.get("config_hash"),
        "suite_start_time": suite.get("start_time"),
        "suite_end_time": suite.get("end_time"),
        "suite_duration_ms": suite.get("duration_ms"),
        "suite_total_tests": suite.get("total_tests"),
        "suite_passed": suite.get("passed"),
        "suite_failed": suite.get("failed"),
        "suite_errors": suite.get("errors"),
        "source_file": str(path),
        "source_kind": "suite_json",
    }
    run_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for result_index, r in enumerate(suite.get("results", []) or []):
        merged_config_snapshot = _deep_merge_config(suite_config_snapshot, r.get("config_snapshot"))
        cfg = flatten_config_snapshot(merged_config_snapshot)
        failure_types = [f.get("check_name") for f in _as_list(r.get("failures")) if isinstance(f, Mapping)]
        failure_messages = [f.get("message") for f in _as_list(r.get("failures")) if isinstance(f, Mapping)]
        rag_ids = extract_rag_entry_ids(r)
        rag_events = _as_list(r.get("rag_retrievals"))
        tool_names = _unique_preserve_order(_as_list(r.get("tool_names_used")))
        agents = _unique_preserve_order(_as_list(r.get("agents_visited")))
        status = r.get("status")

        row = {
            **suite_meta,
            **cfg,
            "result_index": result_index,
            "suite_run_id": r.get("suite_run_id") or suite.get("suite_run_id"),
            # Kept for reference/debugging only -- NOT used for grouping. See
            # note below on why `config_hash` is always recomputed instead.
            "reported_config_hash": r.get("config_hash") or suite.get("config_hash"),
            "test_id": r.get("test_id"),
            "test_name": r.get("test_name"),
            "status": status,
            "passed": _safe_bool_passed(status, r.get("passed")),
            "start_time": r.get("start_time"),
            "end_time": r.get("end_time"),
            "timestamp": r.get("start_time") or suite.get("start_time"),
            "duration_ms": r.get("duration_ms"),
            "total_tokens": r.get("total_tokens"),
            "input_tokens": r.get("input_tokens"),
            "output_tokens": r.get("output_tokens"),
            "llm_calls": r.get("llm_calls"),
            "tool_calls": r.get("tool_calls"),
            "estimated_cost": r.get("estimated_cost"),
            "tasks_completed": r.get("tasks_completed"),
            "num_failures": len(failure_types),
            "failure_types": failure_types,
            "failure_messages": failure_messages,
            "error_message": r.get("error_message"),
            "error_traceback": r.get("error_traceback"),
            "tool_names_used": tool_names,
            "num_tool_names_used": len(tool_names),
            "agents_visited": agents,
            "num_agents_visited": len(agents),
            "rag_entry_ids": rag_ids,
            "num_rag_retrievals": len(rag_events),
            "num_unique_rag_entries": len(rag_ids),
            "final_output": r.get("final_output"),
            "files_created": _as_list(r.get("files_created")),
            "workspace_dir": r.get("workspace_dir"),
        }
        row["run_key"] = f"{row['suite_run_id']}::{row['test_id']}::{row['result_index']}"
        # ALWAYS derive config_hash from this row's own flattened cfg_ fields
        # rather than trusting the suite/result-reported hash. The reported
        # hash is computed by the benchmarking suite itself, at a point in
        # time that may predate newer config fields (graph_topology being
        # the concrete case): two runs that are identical except for
        # graph_topology can carry the *same* reported hash if the suite's
        # own hashing was never updated to include the new field. Since
        # every downstream table/plot that groups or deduplicates by
        # "config" (config_label_table, summarize_by_config, Pareto ranking,
        # surrogate modeling, config_test_pass_matrix, ...) keys off
        # config_hash, trusting a stale externally-computed hash would
        # silently merge runs that actually used different architectures.
        # Recomputing from cfg_* here guarantees the hash always reflects
        # every field flatten_config_snapshot extracts, including any added
        # in the future -- no more silent collisions.
        row["config_hash"] = _hash_config(row)
        run_rows.append(row)

        for retrieval_index, ev in enumerate(rag_events):
            if not isinstance(ev, Mapping):
                continue
            entry_ids = _as_list(ev.get("retrieved_entry_ids"))
            if not entry_ids:
                entry_ids = [None]
            for entry_rank, entry_id in enumerate(entry_ids):
                event_rows.append({
                    "run_key": row["run_key"],
                    "suite_run_id": row["suite_run_id"],
                    "config_hash": row["config_hash"],
                    "test_id": row["test_id"],
                    "test_name": row["test_name"],
                    "passed": row["passed"],
                    "status": row["status"],
                    "failure_types": failure_types,
                    "source_file": str(path),
                    "retrieval_index": retrieval_index,
                    "entry_rank": entry_rank,
                    "entry_id": str(entry_id) if entry_id is not None else None,
                    "agent": ev.get("agent"),
                    "query_type": ev.get("query_type"),
                    "query_text": ev.get("query_text"),
                    "results_count": ev.get("results_count"),
                    "context_used": ev.get("context_used"),
                    "context_tokens": ev.get("context_tokens"),
                    "latency_ms": ev.get("latency_ms"),
                    "step_number": ev.get("step_number"),
                    "timestamp": ev.get("timestamp"),
                    "retrieved_sources": ev.get("retrieved_sources"),
                })

    return pd.DataFrame(run_rows), pd.DataFrame(event_rows)


def load_suite_tables(root_or_files: str | Path | Sequence[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_frames, event_frames = [], []
    files = _suite_files(root_or_files)
    for path in files:
        try:
            runs, events = suite_file_to_tables(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}")
            continue
        if not runs.empty:
            run_frames.append(runs)
        if not events.empty:
            event_frames.append(events)
    runs = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    if not runs.empty:
        runs = clean_run_table(runs)
        events = events[events["run_key"].isin(set(runs["run_key"]))].copy() if not events.empty else events
    return runs, events


def load_regression_jsonl(root_or_files: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    rows = []
    for path in _jsonl_files(root_or_files):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["source_file"] = str(path)
            row["source_kind"] = "regression_jsonl"
            row.setdefault("result_index", i)
            row.setdefault("run_key", f"{row.get('suite_run_id')}::{row.get('test_id')}::{i}")
            row["rag_entry_ids"] = [str(x) for x in _as_list(row.get("rag_entry_ids"))]
            row["failure_types"] = [str(x) for x in _as_list(row.get("failure_types"))]
            row["passed"] = _safe_bool_passed(row.get("status"), row.get("passed"))
            rows.append(row)
    return clean_run_table(pd.DataFrame(rows)) if rows else pd.DataFrame()


def clean_run_table(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs.copy()
    df = runs.copy()
    for c in ["duration_ms", "total_tokens", "input_tokens", "output_tokens", "llm_calls", "tool_calls", "estimated_cost", "tasks_completed", "num_failures", "num_rag_retrievals", "num_unique_rag_entries"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "duration_ms" in df.columns:
        df["duration_s"] = df["duration_ms"] / 1000.0
    if "passed" in df.columns:
        df["passed"] = df["passed"].astype(bool)
        df["passed_int"] = df["passed"].astype(int)
    for c in ["rag_entry_ids", "failure_types", "failure_messages", "tool_names_used", "agents_visited", "files_created"]:
        if c in df.columns:
            df[c] = df[c].apply(_as_list)
    if "num_unique_rag_entries" not in df.columns and "rag_entry_ids" in df.columns:
        df["num_unique_rag_entries"] = df["rag_entry_ids"].apply(lambda x: len(set(_as_list(x))))
    if "num_failures" not in df.columns and "failure_types" in df.columns:
        df["num_failures"] = df["failure_types"].apply(lambda x: len(_as_list(x)))

    # Deduplicate identical suite result copies. Prefer rows with richer RAG data.
    subset = [c for c in ["suite_run_id", "test_id", "result_index"] if c in df.columns]
    if subset:
        sort_cols = [c for c in ["num_rag_retrievals", "num_unique_rag_entries"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        df = df.drop_duplicates(subset=subset, keep="first")
    label_map, _ = config_label_table(df) if "config_hash" in df.columns else (pd.DataFrame(), [])
    if not label_map.empty:
        df = df.merge(label_map[["config_hash", "config_label"]], on="config_hash", how="left", suffixes=("", "_new"))
        if "config_label_new" in df.columns:
            df["config_label"] = df["config_label"].fillna(df["config_label_new"])
            df = df.drop(columns=["config_label_new"])
    return df.reset_index(drop=True)


def load_evaluation_data(root_or_files: str | Path | Sequence[str | Path], prefer_suite_json: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Canonical load entrypoint.

    Returns:
        runs: one row per test result/run
        rag_events: one row per retrieved entry per retrieval event
    """
    suite_runs, rag_events = load_suite_tables(root_or_files)
    jsonl_runs = load_regression_jsonl(root_or_files)

    if prefer_suite_json and not suite_runs.empty:
        runs = suite_runs
        if not jsonl_runs.empty:
            # Fill suite rows with JSONL-only columns if present, but do not overwrite rich suite fields.
            key_cols = [c for c in ["suite_run_id", "test_id"] if c in runs.columns and c in jsonl_runs.columns]
            if key_cols:
                jsonl_small = jsonl_runs.drop_duplicates(key_cols).copy()
                add_cols = [c for c in jsonl_small.columns if c not in runs.columns and c not in key_cols]
                if add_cols:
                    runs = runs.merge(jsonl_small[key_cols + add_cols], on=key_cols, how="left")
                    runs = clean_run_table(runs)
    elif not jsonl_runs.empty:
        runs = jsonl_runs
        rag_events = pd.DataFrame()
    else:
        raise FileNotFoundError("No suite_*.json, suite_results*.json, or regression_data_*.jsonl files found.")
    return runs, rag_events


# -----------------------------------------------------------------------------
# Summaries and exports
# -----------------------------------------------------------------------------

def summarize_by_config(runs: pd.DataFrame) -> pd.DataFrame:
    df = runs.copy()
    if "passed_int" not in df and "passed" in df:
        df["passed_int"] = df["passed"].astype(int)
    agg = {}
    for col in ["passed_int", "duration_s", "duration_ms", "total_tokens", "input_tokens", "output_tokens", "llm_calls", "tool_calls", "tasks_completed", "num_failures", "num_rag_retrievals", "num_unique_rag_entries"]:
        if col in df.columns:
            if col == "passed_int":
                agg[col] = ["mean", "sum", "count"]
            elif col == "num_failures":
                agg[col] = ["mean", "median", "sum"]
            else:
                agg[col] = ["mean", "median"]
    if "estimated_cost" in df.columns:
        agg["estimated_cost"] = ["mean", "sum"]
    cfg = df.groupby("config_hash").agg(agg)
    cfg.columns = ["_".join([p for p in col if p]) for col in cfg.columns.to_flat_index()]
    cfg = cfg.reset_index().rename(columns={"passed_int_mean": "pass_rate", "passed_int_sum": "n_passed", "passed_int_count": "n_tests"})
    cfg_cols = sorted(c for c in df.columns if c.startswith("cfg_"))
    reps = df.drop_duplicates("config_hash")[["config_hash"] + cfg_cols + (["config_label"] if "config_label" in df.columns else [])]
    cfg = cfg.merge(reps, on="config_hash", how="left")
    return cfg


def failure_summary(runs: pd.DataFrame) -> pd.DataFrame:
    if "failure_types" not in runs.columns:
        return pd.DataFrame(columns=["failure_type", "n_runs", "fail_rate_among_all_runs"])
    rows = []
    n = len(runs)
    for _, row in runs.iterrows():
        for ft in _as_list(row.get("failure_types")):
            rows.append({"failure_type": ft, "run_key": row.get("run_key"), "test_id": row.get("test_id"), "config_hash": row.get("config_hash")})
    if not rows:
        return pd.DataFrame(columns=["failure_type", "n_runs", "fail_rate_among_all_runs"])
    out = pd.DataFrame(rows).groupby("failure_type").agg(n_runs=("run_key", "nunique"), n_tests=("test_id", "nunique"), n_configs=("config_hash", "nunique")).reset_index()
    out["fail_rate_among_all_runs"] = out["n_runs"] / max(n, 1)
    return out.sort_values(["n_runs", "failure_type"], ascending=[False, True])


def rag_entry_summary(runs: pd.DataFrame, rag_events: pd.DataFrame | None = None) -> pd.DataFrame:
    if "rag_entry_ids" not in runs.columns:
        return pd.DataFrame()
    rows = []
    for _, row in runs.iterrows():
        for entry_id in set(_as_list(row.get("rag_entry_ids"))):
            rows.append({
                "entry_id": entry_id,
                "run_key": row.get("run_key"),
                "test_id": row.get("test_id"),
                "config_hash": row.get("config_hash"),
                "passed": bool(row.get("passed")),
                "duration_s": row.get("duration_s"),
                "total_tokens": row.get("total_tokens"),
                "estimated_cost": row.get("estimated_cost"),
                "failure_types": row.get("failure_types"),
            })
    if not rows:
        return pd.DataFrame(columns=["entry_id", "n_runs", "pass_rate"])
    edf = pd.DataFrame(rows)
    out = edf.groupby("entry_id").agg(
        n_runs=("run_key", "nunique"),
        n_tests=("test_id", "nunique"),
        n_configs=("config_hash", "nunique"),
        pass_rate=("passed", "mean"),
        mean_duration_s=("duration_s", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        mean_estimated_cost=("estimated_cost", "mean"),
    ).reset_index()
    out["fail_rate"] = 1 - out["pass_rate"]
    if rag_events is not None and not rag_events.empty and "entry_id" in rag_events.columns:
        ev = rag_events.dropna(subset=["entry_id"]).groupby("entry_id").agg(
            n_retrieval_events=("entry_id", "size"),
            mean_context_tokens=("context_tokens", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        ).reset_index()
        out = out.merge(ev, on="entry_id", how="left")
    return out.sort_values(["n_runs", "entry_id"], ascending=[False, True])


def rag_failure_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in runs.iterrows():
        entries = set(_as_list(row.get("rag_entry_ids")))
        failures = set(_as_list(row.get("failure_types"))) or ({"__passed__"} if row.get("passed") else {"__failed_unspecified__"})
        for eid in entries:
            for ft in failures:
                rows.append({"entry_id": eid, "failure_type": ft, "count": 1})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot_table(index="entry_id", columns="failure_type", values="count", aggfunc="sum", fill_value=0)


def rag_significance_table(runs: pd.DataFrame, min_with_entry: int = 1) -> pd.DataFrame:
    """Fisher-exact unadjusted association between entry presence and pass/fail."""
    try:
        from scipy.stats import fisher_exact
    except Exception:
        warnings.warn("scipy is not installed; returning an empty significance table")
        return pd.DataFrame()
    all_entries = sorted(set(itertools.chain.from_iterable(runs.get("rag_entry_ids", pd.Series(dtype=object)).apply(_as_list))))
    rows = []
    y = runs["passed"].astype(bool)
    for eid in all_entries:
        has = runs["rag_entry_ids"].apply(lambda xs: eid in set(_as_list(xs)))
        a = int(((has) & (y)).sum())          # entry, pass
        b = int(((has) & (~y)).sum())         # entry, fail
        c = int(((~has) & (y)).sum())         # no entry, pass
        d = int(((~has) & (~y)).sum())        # no entry, fail
        if a + b < min_with_entry:
            continue
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        pass_with = a / (a + b) if a + b else np.nan
        pass_without = c / (c + d) if c + d else np.nan
        rows.append({
            "entry_id": eid,
            "n_with_entry": a + b,
            "n_without_entry": c + d,
            "pass_with_entry": pass_with,
            "pass_without_entry": pass_without,
            "delta_pass_rate": pass_with - pass_without,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "neg_log10_p": -np.log10(max(p_value, np.finfo(float).tiny)),
            "with_entry_pass": a,
            "with_entry_fail": b,
            "without_entry_pass": c,
            "without_entry_fail": d,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values(["p_value", "entry_id"])


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = prev
    return np.clip(q, 0, 1)


def export_modeling_tables(runs: pd.DataFrame, rag_events: pd.DataFrame, export_dir: str | Path) -> dict[str, Path]:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    tables = {
        "runs_normalized.csv": runs,
        "config_summary.csv": summarize_by_config(runs),
        "failure_summary.csv": failure_summary(runs),
        "rag_entry_summary.csv": rag_entry_summary(runs, rag_events),
        "rag_significance.csv": rag_significance_table(runs),
        "rag_retrieval_events_long.csv": rag_events if rag_events is not None else pd.DataFrame(),
    }
    for name, table in tables.items():
        path = export_dir / name
        table.to_csv(path, index=True if name == "rag_failure_matrix.csv" else False)
        outputs[name] = path

    matrix = rag_failure_matrix(runs)
    path = export_dir / "rag_failure_matrix.csv"
    matrix.to_csv(path)
    outputs[path.name] = path

    # Modeling matrices: pre-run and RAG-aware.
    for name, include_rag in [("design_pre_run.csv", False), ("design_rag_aware.csv", True)]:
        try:
            X, y, spec = make_design_matrix(runs, include_rag=include_rag)
            design = pd.concat([runs[[c for c in ["run_key", "suite_run_id", "config_hash", "test_id", "passed", "status"] if c in runs.columns]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
            path = export_dir / name
            design.to_csv(path, index=False)
            outputs[name] = path
            spec_path = export_dir / name.replace(".csv", "_features.json")
            spec_path.write_text(json.dumps(spec.to_json(), indent=2), encoding="utf-8")
            outputs[spec_path.name] = spec_path
        except Exception as exc:
            warnings.warn(f"Could not export {name}: {exc}")
    manifest = pd.DataFrame([{"name": k, "path": str(v)} for k, v in outputs.items()])
    manifest_path = export_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    outputs[manifest_path.name] = manifest_path
    return outputs


# -----------------------------------------------------------------------------
# Pareto analysis
# -----------------------------------------------------------------------------

def pareto_front_mask(costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        is_efficient[i] = not np.any(dominated)
    return is_efficient


def pareto_rank(costs: np.ndarray) -> np.ndarray:
    remaining = np.arange(len(costs))
    ranks = np.full(len(costs), fill_value=-1, dtype=int)
    rank = 0
    while len(remaining):
        front = pareto_front_mask(costs[remaining])
        ranks[remaining[front]] = rank
        remaining = remaining[~front]
        rank += 1
    return ranks


def add_pareto_columns(configs: pd.DataFrame, objectives: Mapping[str, str] | None = None) -> pd.DataFrame:
    """
    Add `pareto_rank` and a simple weighted `efficiency_score` to config summary.

    objectives maps column -> direction, where direction is "max" or "min".
    """
    if objectives is None:
        objectives = {
            "pass_rate": "max",
            "duration_s_mean": "min",
            "total_tokens_mean": "min",
            "estimated_cost_sum": "min",
            "llm_calls_mean": "min",
            "tool_calls_mean": "min",
        }
    cfg = configs.copy()
    cols = [c for c in objectives if c in cfg.columns]
    if not cols:
        return cfg
    costs = []
    for c in cols:
        s = pd.to_numeric(cfg[c], errors="coerce")
        if objectives[c] == "max":
            s = -s
        costs.append(s.fillna(s.max() if np.isfinite(s.max()) else 0).to_numpy())
    cost_matrix = np.column_stack(costs)
    cfg["pareto_rank"] = pareto_rank(cost_matrix)

    # Composite score: min-max normalize each objective, then average with equal weights.
    score_parts = []
    for c in cols:
        s = pd.to_numeric(cfg[c], errors="coerce")
        lo, hi = s.min(), s.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            norm = pd.Series(1.0, index=cfg.index)
        else:
            norm = (s - lo) / (hi - lo)
        if objectives[c] == "min":
            norm = 1 - norm
        score_parts.append(norm.fillna(0.0))
    cfg["efficiency_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    return cfg.sort_values(["pareto_rank", "efficiency_score"], ascending=[True, False])


# -----------------------------------------------------------------------------
# Feature encoding and surrogate modeling
# -----------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    numeric_cols: list[str]
    categorical_cols: list[str]
    list_cols: dict[str, str]
    dummy_columns: list[str]
    numeric_medians: dict[str, float]
    feature_columns: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "list_cols": self.list_cols,
            "dummy_columns": self.dummy_columns,
            "numeric_medians": self.numeric_medians,
            "feature_columns": self.feature_columns,
        }


def fit_feature_spec(
    runs: pd.DataFrame,
    include_config: bool = True,
    include_test: bool = True,
    include_rag: bool = False,
    include_tools: bool = False,
    min_list_count: int = 1,
) -> FeatureSpec:
    numeric_cols, categorical_cols = [], []
    if include_config:
        for c in sorted([c for c in runs.columns if c.startswith("cfg_") and not c.startswith("cfg_raw_directories")]):
            s = runs[c]
            if pd.api.types.is_bool_dtype(s):
                numeric_cols.append(c)
            elif pd.api.types.is_numeric_dtype(_numeric_if_possible(s)):
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)
    if include_test and "test_id" in runs.columns:
        categorical_cols.append("test_id")

    list_cols: dict[str, str] = {}
    if include_rag and "rag_entry_ids" in runs.columns:
        list_cols["rag_entry_ids"] = "rag_entry"
    if include_tools and "tool_names_used" in runs.columns:
        list_cols["tool_names_used"] = "tool"

    # Build once to freeze dummy/list feature columns.
    numeric_medians = {}
    for c in numeric_cols:
        s = pd.to_numeric(runs[c], errors="coerce")
        med = float(s.median()) if s.notna().any() else 0.0
        numeric_medians[c] = med

    dummies = pd.get_dummies(runs[categorical_cols].astype("string").fillna("__missing__"), columns=categorical_cols, prefix=categorical_cols, dtype=int) if categorical_cols else pd.DataFrame(index=runs.index)
    dummy_columns = dummies.columns.tolist()

    list_feature_columns = []
    for col, prefix in list_cols.items():
        counts = {}
        for xs in runs[col].apply(_as_list):
            for item in set(xs):
                counts[str(item)] = counts.get(str(item), 0) + 1
        keep = sorted([item for item, count in counts.items() if count >= min_list_count])
        list_feature_columns.extend([f"{prefix}_{item}" for item in keep])

    feature_columns = numeric_cols + dummy_columns + list_feature_columns
    return FeatureSpec(numeric_cols, categorical_cols, list_cols, dummy_columns, numeric_medians, feature_columns)


def transform_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    parts = []
    if spec.numeric_cols:
        num = pd.DataFrame(index=df.index)
        for c in spec.numeric_cols:
            if c in df.columns:
                num[c] = pd.to_numeric(df[c], errors="coerce").fillna(spec.numeric_medians.get(c, 0.0))
            else:
                num[c] = spec.numeric_medians.get(c, 0.0)
        parts.append(num)
    if spec.categorical_cols:
        cat = pd.DataFrame(index=df.index)
        for c in spec.categorical_cols:
            cat[c] = df[c] if c in df.columns else "__missing__"
        dum = pd.get_dummies(cat.astype("string").fillna("__missing__"), columns=spec.categorical_cols, prefix=spec.categorical_cols, dtype=int)
        dum = dum.reindex(columns=spec.dummy_columns, fill_value=0)
        parts.append(dum)
    for col, prefix in spec.list_cols.items():
        list_features = [c for c in spec.feature_columns if c.startswith(prefix + "_")]
        mat = pd.DataFrame(0, index=df.index, columns=list_features, dtype=int)
        if col in df.columns:
            for idx, xs in df[col].apply(_as_list).items():
                present = {f"{prefix}_{str(x)}" for x in xs}
                for f in present.intersection(list_features):
                    mat.at[idx, f] = 1
        parts.append(mat)
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return X.reindex(columns=spec.feature_columns, fill_value=0)


def make_design_matrix(
    runs: pd.DataFrame,
    include_rag: bool = False,
    include_tools: bool = False,
    min_list_count: int = 1,
    include_test_id: bool = True,
) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    """Build the design matrix used by the surrogate models.

    ``include_test_id=True`` preserves the original, task-adjusted model used
    for within-benchmark analysis. Set it to ``False`` when the evaluation
    target is generalization to entirely unseen tasks; otherwise the model is
    given an identity feature that cannot carry useful information for a task
    category absent from the training folds.
    """
    spec = fit_feature_spec(
        runs,
        include_config=True,
        include_test=include_test_id,
        include_rag=include_rag,
        include_tools=include_tools,
        min_list_count=min_list_count,
    )
    X = transform_features(runs, spec)
    y = runs["passed"].astype(int) if "passed" in runs.columns else pd.Series(dtype=int)
    return X, y, spec


def choose_cv_splits(y: pd.Series, groups: pd.Series | None = None, max_splits: int = 5) -> int:
    y = pd.Series(y)
    if y.nunique() < 2:
        return 0
    class_min = y.value_counts().min()
    if groups is not None:
        n_groups = pd.Series(groups).nunique()
        return int(max(0, min(max_splits, class_min, n_groups)))
    return int(max(0, min(max_splits, class_min)))


def fit_surrogate_models(
    runs: pd.DataFrame,
    include_rag: bool = False,
    group_col: str = "config_hash",
    include_test_id: bool = True,
) -> dict[str, Any]:

    """Fit compact logistic and random-forest classifiers, with grouped CV when possible."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    X, y, spec = make_design_matrix(runs,include_rag=include_rag,include_test_id=include_test_id,)
    result: dict[str, Any] = {"X": X, "y": y, "spec": spec, "models": {}, "metrics": pd.DataFrame(), "cv_predictions": pd.DataFrame(index=runs.index)}
    if len(y) == 0 or y.nunique() < 2 or X.shape[1] == 0:
        result["status"] = "not_fit_insufficient_classes_or_features"
        return result

    models = {
        "logistic_l2": LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0),
    }
    groups = runs[group_col] if group_col in runs.columns else None
    n_splits = choose_cv_splits(y, groups)
    metrics_rows = []
    cv_pred_cols = {}

    if n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups=groups) if groups is not None and pd.Series(groups).nunique() >= n_splits else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y)
        splits = list(splitter)
        for name, model in models.items():
            pred = np.full(len(y), np.nan)
            for train_idx, test_idx in splits:
                m = clone(model)
                m.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
            mask = ~np.isnan(pred)
            row = {"model": name, "n_cv": int(mask.sum()), "n_splits": n_splits}
            if y[mask].nunique() > 1:
                row["roc_auc"] = roc_auc_score(y[mask], pred[mask])
                row["log_loss"] = log_loss(y[mask], np.clip(pred[mask], 1e-6, 1 - 1e-6))
            row["brier"] = brier_score_loss(y[mask], pred[mask])
            row["accuracy_at_0_5"] = accuracy_score(y[mask], pred[mask] >= 0.5)
            metrics_rows.append(row)
            cv_pred_cols[name] = pred
    else:
        result["status"] = "fit_no_cv"

    fitted = {}
    for name, model in models.items():
        fitted[name] = clone(model).fit(X, y)
    result["models"] = fitted
    result["metrics"] = pd.DataFrame(metrics_rows)
    result["include_test_id"] = include_test_id
    result["cv_group_col"] = group_col
    for name, pred in cv_pred_cols.items():
        result["cv_predictions"][name] = pred
    if "status" not in result:
        result["status"] = "fit_with_cv"
    return result

from typing import Any

import pandas as pd


def fit_surrogates_by_category(
    runs: pd.DataFrame,
    categories: Sequence[str] | None = None,
    category_col: str = "test_category",
    include_rag: bool = False,
    minimum_runs: int = 30,
    minimum_tasks_for_unseen_cv: int = 4,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Fit category-specific surrogate models.

    For each category:
      1. task-adjusted model, grouped by configuration
      2. configuration-only model, grouped by configuration
      3. strict unseen-task model, grouped by test_id

    Returns models, coverage diagnostics, and a combined CV metrics table.
    """
    if category_col not in runs.columns:
        raise KeyError(
            f"{category_col!r} is not present. "
            "Run add_test_category() first."
        )

    work = runs.copy()

    if categories is None:
        selected_categories = sorted(
            work.loc[
                work[category_col] != "unclassified",
                category_col,
            ]
            .dropna()
            .unique()
            .tolist()
        )
    else:
        selected_categories = list(categories)

    category_results: dict[str, Any] = {}
    metric_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for category in selected_categories:
        subset = (
            work.loc[work[category_col] == category]
            .reset_index(drop=True)
            .copy()
        )

        n_runs = len(subset)
        n_tasks = subset["test_id"].nunique()
        n_configs = subset["config_hash"].nunique()
        n_pass = int(subset["passed"].sum())
        n_fail = int((~subset["passed"].astype(bool)).sum())

        coverage_row = {
            "category": category,
            "n_runs": n_runs,
            "n_tasks": n_tasks,
            "n_configs": n_configs,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_rate": subset["passed"].mean() if n_runs else float("nan"),
            "status": "fit",
            "skip_reason": "",
        }

        if n_runs < minimum_runs:
            coverage_row["status"] = "skipped"
            coverage_row["skip_reason"] = (
                f"fewer than {minimum_runs} runs"
            )
            coverage_rows.append(coverage_row)
            continue

        if subset["passed"].nunique() < 2:
            coverage_row["status"] = "skipped"
            coverage_row["skip_reason"] = (
                "category contains only one outcome class"
            )
            coverage_rows.append(coverage_row)
            continue

        # A. Known tasks, unseen configurations, task identity included.
        task_adjusted = fit_surrogate_models(
            subset,
            include_rag=include_rag,
            group_col="config_hash",
            include_test_id=True,
        )

        # B. Known category, unseen configurations, no task ID.
        config_only = fit_surrogate_models(
            subset,
            include_rag=include_rag,
            group_col="config_hash",
            include_test_id=False,
        )

        # C. Entire tasks held out within this category.
        unseen_task = None
        if n_tasks >= minimum_tasks_for_unseen_cv:
            unseen_task = fit_unseen_task_surrogate(
                subset,
                include_rag=include_rag,
                task_col="test_id",
                max_splits=min(5, n_tasks),
                random_state=random_state,
                standardize_numeric=True,
            )

        category_results[category] = {
            "runs": subset,
            "task_adjusted": task_adjusted,
            "config_only": config_only,
            "unseen_task": unseen_task,
        }

        for analysis_name, model_result in [
            ("task_adjusted_config_cv", task_adjusted),
            ("config_only_config_cv", config_only),
            ("config_only_unseen_task_cv", unseen_task),
        ]:
            if model_result is None:
                continue

            metrics = model_result.get("metrics", pd.DataFrame()).copy()
            if metrics.empty:
                continue

            metrics.insert(0, "analysis", analysis_name)
            metrics.insert(0, "category", category)
            metrics["n_category_runs"] = n_runs
            metrics["n_category_tasks"] = n_tasks
            metrics["n_category_configs"] = n_configs
            metric_frames.append(metrics)

        coverage_rows.append(coverage_row)

    combined_metrics = (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames
        else pd.DataFrame()
    )

    return {
        "by_category": category_results,
        "metrics": combined_metrics,
        "coverage": pd.DataFrame(coverage_rows),
    }


def fit_unseen_task_surrogate(
    runs: pd.DataFrame,
    include_rag: bool = False,
    task_col: str = "test_id",
    max_splits: int = 5,
    random_state: int = 0,
    standardize_numeric: bool = True,
) -> dict[str, Any]:
    """Fit a surrogate intended to generalize to benchmark tasks not seen in training.

    This is additive to :func:`fit_surrogate_models`; it does not replace the
    original task-adjusted analysis. The strict unseen-task variant keeps all
    repetitions of a task in one fold, excludes ``test_id`` from the predictor
    matrix, and fits feature preprocessing only on each training fold.
    """
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:  # pragma: no cover - very old scikit-learn only
        StratifiedGroupKFold = None

    if task_col not in runs.columns:
        raise ValueError(f"Cannot run unseen-task CV: {task_col!r} is not present in runs.")

    work = runs.reset_index(drop=True).copy()
    y = work["passed"].astype(int) if "passed" in work.columns else pd.Series(dtype=int)
    groups = work[task_col].astype("string").fillna("__missing_task__")

    # The full-data design is used only for the final refit and downstream
    # predictions. Every CV fold fits its own feature specification below.
    X_full, _, full_spec = make_design_matrix(
        work,
        include_rag=include_rag,
        include_test_id=False,
    )
    result: dict[str, Any] = {
        "X": X_full,
        "y": y,
        "spec": full_spec,
        "models": {},
        "metrics": pd.DataFrame(),
        "cv_predictions": pd.DataFrame(index=work.index),
        "fold_summary": pd.DataFrame(),
        "cv_target": "unseen_tasks",
        "cv_group_col": task_col,
        "include_test_id": False,
        "preprocessing_fitted_within_fold": True,
        "standardize_numeric": bool(standardize_numeric),
    }

    if len(y) == 0 or y.nunique() < 2 or X_full.shape[1] == 0:
        result["status"] = "not_fit_insufficient_classes_or_features"
        return result

    n_splits = choose_cv_splits(y, groups, max_splits=max_splits)
    if n_splits < 2:
        result["status"] = "fit_no_cv"
        return result

    if StratifiedGroupKFold is not None:
        splitter_name = "StratifiedGroupKFold"
        split_iter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ).split(np.zeros((len(work), 1)), y, groups=groups)
    else:
        splitter_name = "GroupKFold"
        split_iter = GroupKFold(n_splits=n_splits).split(
            np.zeros((len(work), 1)), y, groups=groups
        )
    splits = list(split_iter)
    result["cv_splitter"] = splitter_name
    result["n_splits_requested"] = n_splits

    def build_models(spec: FeatureSpec) -> dict[str, Any]:
        logistic: Any = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )
        if standardize_numeric and spec.numeric_cols:
            scale = ColumnTransformer(
                [("numeric", StandardScaler(), spec.numeric_cols)],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
            logistic = Pipeline([("scale", scale), ("clf", logistic)])
        return {
            "logistic_l2": logistic,
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
            ),
        }

    pred_by_model = {
        "logistic_l2": np.full(len(work), np.nan),
        "random_forest": np.full(len(work), np.nan),
    }
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_runs = work.iloc[train_idx]
        test_runs = work.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        fold_row = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_tasks": groups.iloc[train_idx].nunique(),
            "n_test_tasks": groups.iloc[test_idx].nunique(),
            "train_pass_rate": float(y_train.mean()),
            "test_pass_rate": float(y_test.mean()),
            "test_tasks": ";".join(sorted(groups.iloc[test_idx].astype(str).unique())),
            "valid": True,
            "skip_reason": "",
        }
        if y_train.nunique() < 2:
            fold_row["valid"] = False
            fold_row["skip_reason"] = "training fold contains one outcome class"
            fold_rows.append(fold_row)
            continue

        fold_spec = fit_feature_spec(
            train_runs,
            include_config=True,
            include_test=False,
            include_rag=include_rag,
            include_tools=False,
        )
        X_train = transform_features(train_runs, fold_spec)
        X_test = transform_features(test_runs, fold_spec)
        if X_train.shape[1] == 0:
            fold_row["valid"] = False
            fold_row["skip_reason"] = "training fold contains no usable predictors"
            fold_rows.append(fold_row)
            continue

        for name, model in build_models(fold_spec).items():
            fitted = clone(model).fit(X_train, y_train)
            pred_by_model[name][test_idx] = fitted.predict_proba(X_test)[:, 1]
        fold_rows.append(fold_row)

    metrics_rows = []
    for name, pred in pred_by_model.items():
        mask = ~np.isnan(pred)
        row = {
            "model": name,
            "n_cv": int(mask.sum()),
            "n_splits": n_splits,
            "cv_target": "unseen_tasks",
            "group_col": task_col,
            "test_id_feature_included": False,
            "preprocessing_within_fold": True,
        }
        if mask.any():
            if y[mask].nunique() > 1:
                row["roc_auc"] = roc_auc_score(y[mask], pred[mask])
                row["log_loss"] = log_loss(y[mask], np.clip(pred[mask], 1e-6, 1 - 1e-6))
            row["brier"] = brier_score_loss(y[mask], pred[mask])
            row["accuracy_at_0_5"] = accuracy_score(y[mask], pred[mask] >= 0.5)
        metrics_rows.append(row)
        result["cv_predictions"][name] = pred

    fitted_models = {}
    for name, model in build_models(full_spec).items():
        fitted_models[name] = clone(model).fit(X_full, y)
    result["models"] = fitted_models
    result["metrics"] = pd.DataFrame(metrics_rows)
    result["fold_summary"] = pd.DataFrame(fold_rows)
    result["status"] = "fit_with_cv" if any(r["valid"] for r in fold_rows) else "fit_no_valid_cv_folds"
    return result


def feature_effect_table(model: Any, feature_names: Sequence[str], model_name: str | None = None) -> pd.DataFrame:
    name = model_name or type(model).__name__
    estimator = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
    if hasattr(estimator, "coef_"):
        vals = np.ravel(estimator.coef_)
        out = pd.DataFrame({"feature": feature_names, "coefficient": vals})
        out["abs_coefficient"] = out["coefficient"].abs()
        out["odds_ratio"] = np.exp(np.clip(out["coefficient"], -20, 20))
        out["model"] = name
        return out.sort_values("abs_coefficient", ascending=False)
    if hasattr(estimator, "feature_importances_"):
        out = pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
        out["model"] = name
        return out.sort_values("importance", ascending=False)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Active learning / next-run recommendations
# -----------------------------------------------------------------------------

def make_candidate_grid(
    runs: pd.DataFrame,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    test_ids: Sequence[str] | None = None,
    include_static_config_values: bool = True,
    max_candidates: int = 100_000,
) -> pd.DataFrame:
    """Create config × test candidates from an explicit grid or observed config values."""
    cfg_cols = sorted(c for c in runs.columns if c.startswith("cfg_") and not c.startswith("cfg_raw_"))
    if param_grid is None:
        varying = [c for c in cfg_cols if runs[c].nunique(dropna=False) > 1]
        if not varying:
            varying = [c for c in cfg_cols if c in runs.columns]
        param_grid = {c: sorted(runs[c].dropna().unique().tolist(), key=str) for c in varying}
    else:
        # Accept both bare names (llm_temperature) and cfg_* names.
        param_grid = {k if k.startswith("cfg_") else f"cfg_{k}": list(v) for k, v in param_grid.items()}

    static_values = {}
    if include_static_config_values:
        reps = runs.drop_duplicates("config_hash")
        for c in cfg_cols:
            if c not in param_grid and runs[c].nunique(dropna=False) == 1:
                static_values[c] = runs[c].iloc[0]

    test_ids = list(test_ids) if test_ids is not None else sorted(runs["test_id"].dropna().unique().tolist())
    keys = list(param_grid.keys())
    combos = itertools.product(*[param_grid[k] for k in keys]) if keys else [()]
    rows = []
    for combo in combos:
        base = dict(zip(keys, combo))
        base.update(static_values)
        for tid in test_ids:
            row = {**base, "test_id": tid}
            row["config_hash"] = _hash_config(row, cfg_cols=sorted([k for k in row if k.startswith("cfg_")]))
            rows.append(row)
            if len(rows) >= max_candidates:
                warnings.warn(f"Candidate grid truncated at max_candidates={max_candidates}")
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def recommend_next_runs(
    runs: pd.DataFrame,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    test_ids: Sequence[str] | None = None,
    n: int = 20,
    model_result: dict[str, Any] | None = None,
    model_name: str = "random_forest",
    exclude_observed_pairs: bool = True,
    exploration_weight: float = 1.0,
    exploitation_weight: float = 0.25,
) -> pd.DataFrame:
    """
    Rank candidate config × test runs using uncertainty sampling.
    """
    candidates = make_candidate_grid(runs, param_grid=param_grid, test_ids=test_ids)
    observed_pairs = set(zip(runs.get("config_hash", pd.Series(dtype=str)), runs.get("test_id", pd.Series(dtype=str))))
    candidates["observed_pair"] = list(zip(candidates["config_hash"], candidates["test_id"]))
    candidates["already_observed"] = candidates["observed_pair"].isin(observed_pairs)

    counts = runs.groupby(["config_hash", "test_id"]).size().rename("n_prior_runs").reset_index() if {"config_hash", "test_id"}.issubset(runs.columns) else pd.DataFrame()
    if not counts.empty:
        candidates = candidates.merge(counts, on=["config_hash", "test_id"], how="left")
    candidates["n_prior_runs"] = candidates.get("n_prior_runs", 0).fillna(0).astype(int)

    rec = candidates.copy()
    if model_result is None:
        model_result = fit_surrogate_models(runs, include_rag=False)
    can_predict = model_result.get("status") in {"fit_with_cv", "fit_no_cv"} and model_name in model_result.get("models", {})

    if can_predict:
        spec = model_result["spec"]
        model = model_result["models"][model_name]
        Xc = transform_features(rec, spec)
        rec["pred_pass_prob"] = model.predict_proba(Xc)[:, 1]
        rec["uncertainty"] = 1 - np.abs(2 * rec["pred_pass_prob"] - 1)
        rec["active_learning_score"] = exploration_weight * rec["uncertainty"] + exploitation_weight * rec["pred_pass_prob"]
        rec["recommendation_reason"] = np.where(
            rec["uncertainty"] > 0.75,
            "high model uncertainty",
            "high predicted pass probability" if exploitation_weight > 0 else "model-ranked candidate",
        )
    else:
        rec["pred_pass_prob"] = np.nan
        rec["uncertainty"] = np.nan
        rec["active_learning_score"] = -rec["n_prior_runs"] - rec["already_observed"].astype(int)
        rec["recommendation_reason"] = "coverage fallback; model needs both pass and fail examples"

    if exclude_observed_pairs and (~rec["already_observed"]).any():
        rec = rec[~rec["already_observed"]].copy()
    sort_cols = ["active_learning_score", "already_observed", "n_prior_runs"]
    rec = rec.sort_values(sort_cols, ascending=[False, True, True])
    drop_cols = ["observed_pair"]
    return rec.drop(columns=drop_cols).head(n).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plotting helpers: light wrappers returning Matplotlib figures.
# -----------------------------------------------------------------------------

def plot_config_tradeoff(configs: pd.DataFrame, x: str = "duration_s_mean", y: str = "pass_rate", size: str | None = "total_tokens_mean", label_top: int = 5):
    import matplotlib.pyplot as plt
    cfg = configs.copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if size and size in cfg.columns:
        s = pd.to_numeric(cfg[size], errors="coerce")
        sizes = 50 + 250 * (s - s.min()) / (s.max() - s.min() if s.max() != s.min() else 1)
    else:
        sizes = 80
    color = cfg["pareto_rank"] if "pareto_rank" in cfg.columns else None
    sc = ax.scatter(cfg[x], cfg[y], s=sizes, c=color, cmap=BLUE_CMAP, alpha=0.78, edgecolors=BLUE_EDGE_COLOR, linewidths=0.6)
    if color is not None:
        fig.colorbar(sc, ax=ax, label="Pareto rank")
    for _, row in cfg.sort_values(["pareto_rank", y] if "pareto_rank" in cfg.columns else [y], ascending=[True, False] if "pareto_rank" in cfg.columns else [False]).head(label_top).iterrows():
        ax.annotate(str(row.get("config_label", row.get("config_hash", "")))[:60], (row[x], row[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Configuration tradeoff")
    return fig, ax


def plot_failure_breakdown(runs: pd.DataFrame):
    import matplotlib.pyplot as plt
    fs = failure_summary(runs)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(fs))))
    if fs.empty:
        ax.text(0.5, 0.5, "No failures recorded", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    fs = fs.sort_values("n_runs")
    cats = fs["failure_type"].astype(str).tolist()
    cmap = get_categorical_colors(cats)
    bars = ax.barh(fs["failure_type"], fs["n_runs"], color=[cmap[c] for c in cats])
    apply_bar_hatches(bars.patches, cats)
    ax.set_xlabel("Runs")
    ax.set_title("Failure type breakdown")
    return fig, ax


def plot_retrieval_frequency(runs: pd.DataFrame):
    import matplotlib.pyplot as plt
    summary = rag_entry_summary(runs)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(summary))))
    if summary.empty:
        ax.text(0.5, 0.5, "No RAG entries recorded", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    s = summary.sort_values("n_runs")
    cats = s["entry_id"].astype(str).tolist()
    cmap = get_categorical_colors(cats)
    bars = ax.barh(s["entry_id"], s["n_runs"], color=[cmap[c] for c in cats])
    apply_bar_hatches(bars.patches, cats)
    ax.set_xlabel("Runs where entry was retrieved")
    ax.set_title("RAG retrieval frequency")
    return fig, ax


def plot_entry_pass_rates(runs: pd.DataFrame, min_runs: int = 1):
    import matplotlib.pyplot as plt
    summary = rag_entry_summary(runs)
    summary = summary[summary["n_runs"] >= min_runs] if not summary.empty else summary
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(summary))))
    if summary.empty:
        ax.text(0.5, 0.5, "No entries meet threshold", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    s = summary.sort_values("pass_rate")
    cats = s["entry_id"].astype(str).tolist()
    cmap = get_categorical_colors(cats)
    bars = ax.barh(s["entry_id"], s["pass_rate"], color=[cmap[c] for c in cats])
    apply_bar_hatches(bars.patches, cats)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pass rate when retrieved")
    ax.set_title("Per-entry pass rate")
    return fig, ax


def plot_rag_failure_heatmap(runs: pd.DataFrame):
    import matplotlib.pyplot as plt
    mat = rag_failure_matrix(runs)
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(mat.columns)), max(3, 0.35 * len(mat))))
    if mat.empty:
        ax.text(0.5, 0.5, "No RAG/failure matrix available", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    im = ax.imshow(mat.values, aspect="auto", cmap=BLUE_CMAP)
    ax.set_xticks(range(len(mat.columns)), labels=mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)), labels=mat.index)
    ax.set_title("RAG entry × failure type counts")
    fig.colorbar(im, ax=ax, label="count")
    return fig, ax


def plot_volcano(sig: pd.DataFrame):
    """Compact q-value volcano plot retained for backward compatibility."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if sig is None or sig.empty:
        ax.text(0.5, 0.5, "No significance results", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    df = classify_rag_entries_q(sig, q_threshold=0.05)
    tiny = np.finfo(float).tiny
    df["neg_log10_q"] = -np.log10(pd.to_numeric(df["q_value_bh"], errors="coerce").clip(lower=tiny))
    colors = df["classification"].map(RAG_EFFECT_COLORS).fillna(RAG_EFFECT_COLORS["neutral"])
    ax.scatter(
        df["delta_pass_rate"], df["neg_log10_q"], s=60, c=colors,
        alpha=0.75, edgecolors="black", linewidths=0.5,
    )
    for _, row in df.nsmallest(10, "q_value_bh").iterrows():
        ax.annotate(
            row["entry_id"], (row["delta_pass_rate"], row["neg_log10_q"]),
            fontsize=8, xytext=(4, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.9, lw=0.4),
        )
    ax.axvline(0, linestyle="--", linewidth=1, color="0.35")
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, color="0.35")
    ax.set_xlabel("Δ pass rate when retrieved")
    ax.set_ylabel(r"$-\log_{10}$(BH q-value)")
    ax.set_title("RAG entry association volcano plot")
    return fig, ax


# =============================================================================
# =============================================================================
# CONSOLIDATED EXTENSIONS
#
# Everything below this line was merged in from the previous notebooks
# (sweep_analysis, agent_eval_surrogate_modeling, rag_retrieval_analysis)
# and from screen_eval_outliers.py, so the master notebook can stay thin.
# The methods are preserved; only the packaging changed.
# =============================================================================
# =============================================================================


# -----------------------------------------------------------------------------
# Publication figure style
# -----------------------------------------------------------------------------

PUBLICATION_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.55,
    "grid.color": "#B8C4D0",
    "axes.edgecolor": "#31485F",
    "axes.linewidth": 0.75,
    "patch.linewidth": 0.70,
    "hatch.linewidth": 0.45,
    "legend.edgecolor": "#B8C4D0",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,   # editable text in Illustrator/Inkscape
    "ps.fonttype": 42,
}

# NOTE: `set_publication_style()` is defined once, in the "UNIFIED PLOTTING
# FRAMEWORK" section near the end of this file -- it applies PUBLICATION_RC
# above plus a shared seaborn base theme, so every notebook (including this
# one) renders with the same fonts/spines/grid *and* the same seaborn theme.


def save_figure(fig, name: str, plot_dir: str | Path, formats: Sequence[str] = ("png", "pdf"), dpi: int = 300) -> list[Path]:
    """Save one figure as publication-ready PNG (raster) and PDF (vector)."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = plot_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths


# -----------------------------------------------------------------------------
# Objective metadata for config-level Pareto analysis
# (port of the OBJECTIVES dict from sweep_analysis, renamed to the canonical
#  summarize_by_config() column names)
# -----------------------------------------------------------------------------

OBJECTIVE_INFO: dict[str, dict[str, str]] = {
    "pass_rate":           {"direction": "maximize", "label": "Pass Rate",         "fmt": ".0%"},
    "duration_s_mean":     {"direction": "minimize", "label": "Duration (s)",      "fmt": ".1f"},
    "total_tokens_mean":   {"direction": "minimize", "label": "Tokens (mean)",     "fmt": ".0f"},
    "estimated_cost_sum":  {"direction": "minimize", "label": "Total Cost (USD)",  "fmt": ".4f"},
    "llm_calls_mean":      {"direction": "minimize", "label": "LLM Calls (mean)",  "fmt": ".1f"},
    "tool_calls_mean":     {"direction": "minimize", "label": "Tool Calls (mean)", "fmt": ".1f"},
    "num_failures_sum":    {"direction": "minimize", "label": "Failures (total)",  "fmt": ".0f"},
}

# Default composite weights, preserved from sweep_analysis section 13.
DEFAULT_EFFICIENCY_WEIGHTS: dict[str, float] = {
    "pass_rate":          0.4,
    "duration_s_mean":    0.15,
    "total_tokens_mean":  0.15,
    "estimated_cost_sum": 0.0,
    "llm_calls_mean":     0.1,
    "tool_calls_mean":    0.1,
    "num_failures_sum":   0.1,
}


def active_objectives(configs: pd.DataFrame, objectives: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, dict[str, str]]:
    objectives = dict(objectives or OBJECTIVE_INFO)
    return {k: dict(v) for k, v in objectives.items() if k in configs.columns}


def objective_cost_matrix(configs: pd.DataFrame, objectives: Mapping[str, Mapping[str, str]]) -> np.ndarray:
    """Build a cost matrix where every column is minimize-is-better."""
    cols = []
    for name, info in objectives.items():
        vals = pd.to_numeric(configs[name], errors="coerce").astype(float)
        vals = vals.fillna(vals.max() if np.isfinite(vals.max()) else 0.0)
        if info.get("direction") == "maximize":
            vals = -vals
        cols.append(vals.to_numpy())
    return np.column_stack(cols)


def domination_count(costs: np.ndarray) -> np.ndarray:
    """Count how many other points each point is dominated by."""
    n = costs.shape[0]
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        dominated_by = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        counts[i] = int(dominated_by.sum())
    return counts


def add_objective_pareto_columns(configs: pd.DataFrame, objectives: Mapping[str, Mapping[str, str]] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Pareto rank / domination count / is_pareto using OBJECTIVE_INFO directions."""
    cfg = configs.copy()
    objs = active_objectives(cfg, objectives)
    if not objs:
        return cfg, objs
    costs = objective_cost_matrix(cfg, objs)
    cfg["pareto_rank"] = pareto_rank(costs)
    cfg["dominated_by"] = domination_count(costs)
    cfg["is_pareto"] = cfg["pareto_rank"] == 0
    return cfg, objs


def weighted_efficiency_score(configs: pd.DataFrame, weights: Mapping[str, float] | None = None, objectives: Mapping[str, Mapping[str, str]] | None = None) -> pd.DataFrame:
    """
    Composite weighted efficiency score (sweep_analysis section 13 method):
    min-max normalize each objective in its 'better = 1' direction, then take a
    weighted average with weights normalized to sum to 1.
    """
    cfg = configs.copy()
    objs = active_objectives(cfg, objectives)
    weights = {k: v for k, v in (weights or DEFAULT_EFFICIENCY_WEIGHTS).items() if k in cfg.columns}
    if not weights:
        return cfg
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    cfg["efficiency_score"] = 0.0
    for obj, w in weights.items():
        vals = pd.to_numeric(cfg[obj], errors="coerce").astype(float)
        lo, hi = vals.min(), vals.max()
        direction = objs.get(obj, {}).get("direction", "minimize")
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            normed = pd.Series(1.0, index=cfg.index)
        elif direction == "maximize":
            normed = (vals - lo) / (hi - lo)
        else:
            normed = 1.0 - (vals - lo) / (hi - lo)
        cfg["efficiency_score"] = cfg["efficiency_score"] + w * normed.fillna(0.0)
    return cfg


def short_config_label(
    full_label: str,
    max_parts: int = 3,
    prioritize: Sequence[str] = ("model", "T", "episodic_top_k"),
) -> str:
    """Shorten 'llm_model=gpt4 | llm_temperature=0.3 | episodic_top_k=5' -> compact stacked label.

    Parts whose abbreviated key is in ``prioritize`` are moved to the front so
    they are never lost to the ``max_parts`` truncation (e.g. the LLM model).
    """

    parts = str(full_label).split(" | ")
    short = []
    for p in parts:
        k, _, v = p.partition("=")
        abbr = (
            k.replace("llm_model", "model")
             .replace("llm_temperature", "temperature")
             .replace("episodic_top_k", "RAG_top_k")
            #  .replace("top_k", "k")
            #  .replace("max_tokens", "mt")
            #  .strip("max_messages")
            #  .strip("max_retries")
            .strip("_")
        )
        short.append(f"{abbr}={v}" if abbr else v)
    rank = {key: i for i, key in enumerate(prioritize)}
    short.sort(key=lambda s: rank.get(s.partition("=")[0], len(rank)))
    return "\n".join(short[:max_parts])


# -----------------------------------------------------------------------------
# Publication figures: config-level Pareto suite (paper Figure 2)
# -----------------------------------------------------------------------------

def plot_pareto_radar(configs: pd.DataFrame, objectives: Mapping[str, Mapping[str, str]] | None = None, max_configs: int = 8, label_parts: int = 3, title: str = "Pareto-Optimal Config Profiles\n(outer = better)"):
    """Radar/spider chart of the Pareto-front configurations (paper Fig 2a)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cfg, objs = add_objective_pareto_columns(configs, objectives)
    radar_objectives = list(objs.keys())
    if len(radar_objectives) < 3:
        raise ValueError("Need at least 3 objectives present in the config table for a radar chart.")

    top = cfg[cfg["pareto_rank"] == 0].copy()
    if len(top) > max_configs:
        top = top.nsmallest(max_configs, "dominated_by")
    if len(top) == 0:
        top = cfg.nsmallest(min(5, len(cfg)), "pareto_rank")

    n_axes = len(radar_objectives)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    norm_data = {}
    for obj in radar_objectives:
        vals = pd.to_numeric(cfg[obj], errors="coerce").astype(float)
        lo, hi = vals.min(), vals.max()
        if hi == lo or not np.isfinite(hi - lo):
            norm_data[obj] = pd.Series(1.0, index=cfg.index)
        elif objs[obj]["direction"] == "maximize":
            norm_data[obj] = (vals - lo) / (hi - lo)
        else:
            norm_data[obj] = 1 - (vals - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = get_blue_palette(len(top))
    for i, (idx, row) in enumerate(top.iterrows()):
        values = [float(norm_data[obj].loc[idx]) for obj in radar_objectives]
        values += values[:1]
        lbl = short_config_label(row.get("config_label", row.get("config_hash", "")), label_parts)
        ax.plot(angles, values, "o-", linewidth=2, label=lbl, color=palette[i], markersize=6)
        ax.fill(angles, values, alpha=0.08, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([objs[o]["label"] for o in radar_objectives], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Worst", "", "", "Best"], fontsize=7, alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.05), fontsize=7.5, frameon=True, borderpad=0.8, labelspacing=0.6)
    fig.tight_layout()
    return fig, ax


DEFAULT_PARETO_PAIRS = [
    ("pass_rate", "duration_s_mean"),
    ("pass_rate", "estimated_cost_sum"),
    ("pass_rate", "total_tokens_mean"),
    ("duration_s_mean", "estimated_cost_sum"),
    ("duration_s_mean", "total_tokens_mean"),
    ("total_tokens_mean", "estimated_cost_sum"),
    ("llm_calls_mean", "duration_s_mean"),
    ("tool_calls_mean", "duration_s_mean"),
]


def _try_adjust_text(texts, ax):
    try:
        from adjustText import adjust_text
        if texts:
            adjust_text(texts, ax=ax)
    except Exception:
        pass


def plot_pairwise_pareto_fronts(configs: pd.DataFrame, pairs: Sequence[tuple[str, str]] | None = None, objectives: Mapping[str, Mapping[str, str]] | None = None, annotate_top: int = 3, label_parts: int = 6):
    """One scatter per objective pair with the 2D Pareto front highlighted (Fig 2 d-f style). Returns {(x, y): (fig, ax)}."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cfg, objs = add_objective_pareto_columns(configs, objectives)
    pairs = [(a, b) for a, b in (pairs or DEFAULT_PARETO_PAIRS) if a in objs and b in objs]
    cmap = plt.get_cmap(BLUE_CMAP)
    norm = Normalize(vmin=0, vmax=1)
    out = {}

    for obj_a, obj_b in pairs:
        fig, ax = plt.subplots(figsize=(7, 6))
        x = pd.to_numeric(cfg[obj_a], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(cfg[obj_b], errors="coerce").to_numpy(dtype=float)
        pair_costs = np.column_stack([
            -x if objs[obj_a]["direction"] == "maximize" else x,
            -y if objs[obj_b]["direction"] == "maximize" else y,
        ])
        pair_pareto = pareto_front_mask(pair_costs)
        colors = cmap(norm(pd.to_numeric(cfg["pass_rate"], errors="coerce").fillna(0).to_numpy()))

        ax.scatter(x[~pair_pareto], y[~pair_pareto], c=colors[~pair_pareto], s=70, alpha=0.5, edgecolors="gray", linewidths=0.5, zorder=2)
        ax.scatter(x[pair_pareto], y[pair_pareto], c=colors[pair_pareto], s=160, alpha=0.95, edgecolors="black", linewidths=1.5, zorder=3, marker="*")

        pareto_pts = np.column_stack([x[pair_pareto], y[pair_pareto]])
        if len(pareto_pts) > 1:
            order = np.argsort(pareto_pts[:, 0])
            ax.plot(pareto_pts[order, 0], pareto_pts[order, 1], "k--", alpha=0.4, lw=1.2, zorder=1)

        texts = []
        for i in np.where(pair_pareto)[0][: max(0, annotate_top)]:
            lbl = short_config_label(cfg.iloc[i].get("config_label", ""), label_parts)
            texts.append(ax.annotate(lbl, (x[i], y[i]), fontsize=7, alpha=0.85, textcoords="offset points", xytext=(12, 8),
                                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7, lw=0.5),
                                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6)))
        _try_adjust_text(texts, ax)

        ax.set_xlabel(objs[obj_a]["label"])
        ax.set_ylabel(objs[obj_b]["label"])
        ax.set_title(f"{objs[obj_a]['label']}  vs  {objs[obj_b]['label']}")
        fig.tight_layout()
        out[(obj_a, obj_b)] = (fig, ax)
    return out


DEFAULT_TRADEOFF_SPECS = [
    {"x": "duration_s_mean", "y": "total_tokens_mean", "size": "pass_rate", "color": "estimated_cost_sum",
     "title": "Speed vs Token Efficiency (size = pass rate, color = cost)"},
    {"x": "pass_rate", "y": "estimated_cost_sum", "size": "total_tokens_mean", "color": "duration_s_mean",
     "title": "Success vs Cost (size = tokens, color = duration)"},
    {"x": "llm_calls_mean", "y": "duration_s_mean", "size": "pass_rate", "color": "total_tokens_mean",
     "title": "LLM Calls vs Duration (size = pass rate, color = tokens)"},
]


def plot_tradeoff_bubbles(configs: pd.DataFrame, specs: Sequence[Mapping[str, str]] | None = None, objectives: Mapping[str, Mapping[str, str]] | None = None, annotate_top: int = 3, annotate_min_pass_rate: float | None = None, label_parts: int = 6):
    """Bubble tradeoff deep-dives with the 2D Pareto-optimal points ringed in gold (paper Fig 2 d-f). Returns {title: (fig, ax)}."""
    import matplotlib.pyplot as plt

    cfg, objs = add_objective_pareto_columns(configs, objectives)
    specs = [s for s in (specs or DEFAULT_TRADEOFF_SPECS) if all(c in cfg.columns for c in (s["x"], s["y"], s["size"], s["color"]))]
    out = {}
    for spec in specs:
        fig, ax = plt.subplots(figsize=(10, 7))
        x = pd.to_numeric(cfg[spec["x"]], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(cfg[spec["y"]], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(cfg[spec["size"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        c = pd.to_numeric(cfg[spec["color"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        s_min, s_max = s.min(), s.max()
        s_norm = 80 + 520 * (s - s_min) / (s_max - s_min) if s_max > s_min else np.full_like(s, 300)

        scatter = ax.scatter(x, y, s=s_norm, c=c, cmap=BLUE_CMAP, alpha=0.8, edgecolors="black", linewidths=0.8)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(objs.get(spec["color"], {}).get("label", spec["color"]))

        x_dir = objs.get(spec["x"], {"direction": "minimize"}).get("direction")
        y_dir = objs.get(spec["y"], {"direction": "minimize"}).get("direction")
        pair_costs = np.column_stack([-x if x_dir == "maximize" else x, -y if y_dir == "maximize" else y])
        pair_pareto = pareto_front_mask(pair_costs)
        ax.scatter(x[pair_pareto], y[pair_pareto], facecolors="none", edgecolors=BLUE_EDGE_COLOR, linewidths=2.5, s=s_norm[pair_pareto] + 100, zorder=5, label="Pareto-optimal")

        pareto_idxs = np.where(pair_pareto)[0]
        if annotate_min_pass_rate is not None and "pass_rate" in cfg.columns:
            pareto_idxs = [i for i in pareto_idxs if cfg.iloc[i]["pass_rate"] > annotate_min_pass_rate]
        texts = []
        for i in list(pareto_idxs)[: max(0, annotate_top)]:
            lbl = short_config_label(cfg.iloc[i].get("config_label", ""), label_parts)
            texts.append(ax.annotate(lbl, (x[i], y[i]), fontsize=7, alpha=0.85, textcoords="offset points", xytext=(14, 10),
                                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.75, lw=0.5),
                                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6), zorder=10))
        _try_adjust_text(texts, ax)

        ax.set_xlabel(objs.get(spec["x"], {}).get("label", spec["x"]))
        ax.set_ylabel(objs.get(spec["y"], {}).get("label", spec["y"]))
        ax.set_title(spec["title"], fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out[spec["title"]] = (fig, ax)
    return out


def plot_efficiency_frontier(configs: pd.DataFrame, x: str = "total_tokens_mean", y: str = "pass_rate", group_col: str = "cfg_llm_model", label_col: str | None = "config_label", label_parts: int = 2, title: str = "Token Usage vs. Accuracy Efficiency Frontier"):
    """Scatter of every configuration colored/marked by a categorical column (paper Fig 2b)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter

    cfg = configs.copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    groups = cfg[group_col].fillna("unknown").astype(str) if group_col in cfg.columns else pd.Series("all", index=cfg.index)
    palette = get_categorical_colors(groups.unique(), palette=BLUE_CMAP)
    markers = ["s", "o", "^", "D", "v", "P", "X", "*"]
    texts = []
    for gi, (g, sub) in enumerate(cfg.groupby(groups)):
        ax.scatter(pd.to_numeric(sub[x], errors="coerce"), pd.to_numeric(sub[y], errors="coerce"),
                   s=90, color=palette[str(g)], marker=markers[gi % len(markers)], alpha=0.9,
                   edgecolors="black", linewidths=0.6, label=str(g))
        if label_col and label_col in sub.columns:
            for _, row in sub.iterrows():
                texts.append(ax.annotate(short_config_label(row[label_col], label_parts).replace("\n", " / "),
                                         (row[x], row[y]), fontsize=6.5, alpha=0.8, xytext=(5, 4), textcoords="offset points"))
    _try_adjust_text(texts, ax)
    if y == "pass_rate":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel(OBJECTIVE_INFO.get(x, {}).get("label", x))
    ax.set_ylabel(OBJECTIVE_INFO.get(y, {}).get("label", y))
    ax.set_title(title)
    ax.legend(title=group_col.replace("cfg_", ""), fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_hyperparameter_impact(configs: pd.DataFrame, params: Sequence[str] | None = None, score_col: str = "efficiency_score", pass_col: str = "pass_rate", ncols: int = 4, suptitle: str = "Hyperparameter → Efficiency & Pass Rate"):
    """Small-multiple line plots of mean efficiency and pass rate vs each numeric hyperparameter (paper Fig 3d)."""
    import matplotlib.pyplot as plt

    cfg = configs.copy()
    if score_col not in cfg.columns:
        cfg = weighted_efficiency_score(cfg)
    if params is None:
        cfg_cols = sorted(c for c in cfg.columns if c.startswith("cfg_"))
        params = [c for c in cfg_cols if cfg[c].nunique(dropna=False) > 1 and pd.api.types.is_numeric_dtype(_numeric_if_possible(cfg[c]))]
    params = [p for p in params if p in cfg.columns]
    if not params:
        raise ValueError("No varying numeric hyperparameters available.")

    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    for ax, param in zip(axes.ravel(), params):
        short = param.replace("cfg_", "")
        tmp = cfg.copy()
        tmp[param] = pd.to_numeric(tmp[param], errors="coerce")
        agg = tmp.groupby(param).agg(mean_eff=(score_col, "mean"), mean_pass=(pass_col, "mean")).reset_index().sort_values(param)
        ax.plot(agg[param], agg["mean_eff"], "s-", color="#4292C6", lw=2, markersize=7, label="Efficiency")
        ax.plot(agg[param], agg["mean_pass"], "o--", color="#08519C", lw=1.5, markersize=6, label="Pass Rate")
        ax.set_xlabel(short)
        ax.set_ylabel("Score (0-1)")
        ax.set_title(f"Impact of {short}")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_config_outcome_correlation(runs: pd.DataFrame, outcome_cols: Sequence[str] | None = None, title: str = "Correlation: Config Parameters → Test Outcomes", ignore_inputs = ["max_messages", "recursion_limit"]):
    """Annotated heatmap of Pearson correlation between numeric hyperparameters and run-level outcomes (paper Fig 3c)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = runs.copy()
    if outcome_cols is None:
        outcome_cols = ["passed", "duration_ms", "total_tokens", "input_tokens", "output_tokens", "llm_calls", "tool_calls", "estimated_cost", "tasks_completed", "num_failures"]
    outcome_cols = [c for c in outcome_cols if c in df.columns]
    cfg_cols = sorted(c for c in df.columns if c.startswith("cfg_"))
    varying = [c for c in cfg_cols if df[c].nunique(dropna=False) > 1]
    numeric_cfgs = []
    for c in varying:
        if c in ignore_inputs:
            continue
        s = _numeric_if_possible(df[c])
        if pd.api.types.is_numeric_dtype(s):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_cfgs.append(c)
    if not numeric_cfgs or len(outcome_cols) < 2:
        raise ValueError("Not enough varying numeric config parameters or outcomes for the correlation heatmap.")

    corr_df = df[numeric_cfgs + outcome_cols].copy()
    if "passed" in corr_df.columns:
        corr_df["passed"] = corr_df["passed"].astype(float)
    corr = corr_df.corr().loc[numeric_cfgs, outcome_cols]

    fig, ax = plt.subplots(figsize=(max(9, len(outcome_cols) * 1.1), max(3, len(numeric_cfgs) * 0.7)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=BLUE_CMAP, center=0, linewidths=1, linecolor="white", ax=ax, vmin=-1, vmax=1)
    ax.set_yticklabels([c.replace("cfg_", "") for c in numeric_cfgs], rotation=0)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax, corr


def config_test_pass_matrix(runs: pd.DataFrame) -> pd.DataFrame:
    """test_id × config pass-rate matrix for quickly sorting through test results."""
    return runs.pivot_table(index="test_id", columns="config_label" if "config_label" in runs.columns else "config_hash", values="passed_int", aggfunc="mean")


def plot_config_test_pass_matrix(runs: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    mat = config_test_pass_matrix(runs)
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * mat.shape[1]), max(5, 0.3 * mat.shape[0])))
    sns.heatmap(mat, cmap=BLUE_CMAP, vmin=0, vmax=1, linewidths=0.4, linecolor="white", ax=ax, cbar_kws={"label": "Pass rate"})
    ax.set_xticklabels([short_config_label(t.get_text(), 3).replace("\n", " / ") for t in ax.get_xticklabels()], rotation=90, fontsize=6.5)
    ax.set_title("Pass rate by test and configuration")
    fig.tight_layout()
    return fig, ax, mat


# -----------------------------------------------------------------------------
# Surrogate modeling extensions (paper Figure 3a/3b, exploitative recommendations)
# Ported from agent_eval_surrogate_modeling.
# -----------------------------------------------------------------------------

def family_from_feature(c: str) -> str:
    if c.startswith("test_id_"):
        return "task identity"
    if c.startswith("cfg_llm_model"):
        return "LLM model"
    if c.startswith("cfg_llm_provider"):
        return "LLM provider"
    if c.startswith("cfg_llm_temperature"):
        return "temperature"
    if c.startswith("cfg_max_tokens"):
        return "max tokens"
    if c.startswith("cfg_chunk_size"):
        return "chunk size"
    if c.startswith("cfg_chunk_overlap"):
        return "chunk overlap"
    if c.startswith("cfg_similarity_threshold"):
        return "similarity threshold"
    if c.startswith("cfg_episodic_top_k"):
        return "episodic top-k"
    if c.startswith("cfg_contextual_top_k"):
        return "contextual top-k"
    if c.startswith("cfg_embedding_model"):
        return "embedding model"
    if c.startswith("cfg_rag_cache"):
        return "RAG cache"
    if c.startswith(("cfg_max_messages", "cfg_preserve_recent_messages", "cfg_pruning_strategy")):
        return "message pruning"
    if c.startswith(("cfg_agent_timeout", "cfg_max_retries", "cfg_recursion_limit")):
        return "agent limits"
    if c.startswith(("cfg_test_mock_mode", "cfg_test_timeout_seconds")):
        return "test harness"
    if c.startswith("cfg_cost_per_1k"):
        return "cost constants"
    if c.startswith("cfg_tool_") or c.startswith("tool_"):
        return "tool availability"
    if c.startswith("rag_entry_"):
        return "RAG entry"
    if c.startswith("rag_"):
        return "RAG summary"
    if c.startswith("cfg_graph_topology_"):
        return "# of agents"
    if c.startswith("cfg_max_workflows_to_keep"):
        return "Max # of Workflows"
    return "other"


def feature_family_audit(model_result: Mapping[str, Any], model_name: str = "random_forest") -> pd.DataFrame:
    """Per-feature breakdown of the family grouping behind Fig 3b.

    Returns one row per design-matrix feature with its assigned family and
    (when available) its random-forest importance and logistic coefficient,
    sorted so 'other' appears first — use this to audit what fell through the
    family mapping rather than guessing.
    """
    features = list(model_result["spec"].feature_columns)
    out = pd.DataFrame({"feature": features})
    out["family"] = out["feature"].map(family_from_feature)
    rf = model_result["models"].get(model_name)
    if rf is not None and hasattr(rf, "feature_importances_"):
        out["rf_importance"] = rf.feature_importances_
    logit = model_result["models"].get("logistic_l2")
    if logit is not None:
        clf = logit.named_steps.get("clf", logit) if hasattr(logit, "named_steps") else logit
        if hasattr(clf, "coef_"):
            out["logistic_coef"] = clf.coef_.ravel()
    out["is_other"] = out["family"].eq("other")
    sort_cols = ["is_other"] + (["rf_importance"] if "rf_importance" in out.columns else [])
    out = out.sort_values(sort_cols, ascending=[False, False][: len(sort_cols)]).drop(columns="is_other")
    return out.reset_index(drop=True)


def wilson_interval(k, n, z: float = 1.96):
    """Wilson confidence interval for binomial proportions."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    lo = np.where(n > 0, center - half, np.nan)
    hi = np.where(n > 0, center + half, np.nan)
    return lo, hi


def coefficient_table(model_result: Mapping[str, Any], model_name: str = "logistic_l2") -> pd.DataFrame:
    """Logistic coefficients (+ odds ratios + feature family) from a fit_surrogate_models() result."""
    model = model_result["models"].get(model_name)
    if model is None:
        return pd.DataFrame()
    out = feature_effect_table(model, model_result["spec"].feature_columns, model_name)
    out["family"] = out["feature"].map(family_from_feature)
    return out


def family_importance_table(model_result: Mapping[str, Any], model_name: str = "random_forest") -> pd.DataFrame:
    """Random-forest feature importance summed by feature family (paper Fig 3b table)."""
    model = model_result["models"].get(model_name)
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    imp = pd.DataFrame({"feature": model_result["spec"].feature_columns, "importance": model.feature_importances_})
    imp["family"] = imp["feature"].map(family_from_feature)
    fam = imp.groupby("family", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    return fam


def plot_family_importance(family_importance: pd.DataFrame, top: int = 10, title: str = "Pre-run feature importance by family"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sub = family_importance.head(top).iloc[::-1]
    cats = sub["family"].astype(str).tolist()
    cmap = get_categorical_colors(cats)
    bars = ax.barh(sub["family"], sub["importance"], color=[cmap[c] for c in cats])
    apply_bar_hatches(bars.patches, cats)
    ax.set_xlabel("Summed random-forest importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_calibration(model_result: Mapping[str, Any], n_bins: int = 8, title: str = "Pre-run surrogate calibration\nGrouped CV by config", exclude: Sequence[str] | None = None):
    """Reliability diagram from grouped-CV out-of-fold predictions (paper Fig 3a)."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y = model_result["y"]
    exclude = set(exclude or [])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    plotted = False
    for name in model_result.get("cv_predictions", pd.DataFrame()).columns:
        if name in exclude:
            continue
        pred = model_result["cv_predictions"][name].to_numpy(dtype=float)
        mask = ~np.isnan(pred)
        if mask.sum() < n_bins or pd.Series(y[mask]).nunique() < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y[mask], pred[mask], n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
        plotted = True
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=BLUE_EDGE_COLOR, alpha=0.7)
    ax.set_xlabel("Predicted pass probability")
    ax.set_ylabel("Observed pass frequency")
    ax.set_title(title)
    if plotted:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No usable CV predictions", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig, ax


def predict_config_task_grid(runs: pd.DataFrame, model_result: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """
    Predict pass probability for every observed configuration on every observed task,
    using the pre-run surrogate (exploitative use of the model).
    """
    if model_result is None:
        model_result = fit_surrogate_models(runs, include_rag=False)
    if not model_result.get("models"):
        raise ValueError(f"Surrogate model is not usable: status={model_result.get('status')}")
    spec = model_result["spec"]

    cfg_cols = sorted(c for c in runs.columns if c.startswith("cfg_"))
    cfg_reps = runs.drop_duplicates("config_hash")[["config_hash", "config_label"] + cfg_cols] if "config_label" in runs.columns else runs.drop_duplicates("config_hash")[["config_hash"] + cfg_cols]
    test_ids = sorted(runs["test_id"].dropna().unique().tolist())

    rows = []
    for _, cfg_row in cfg_reps.iterrows():
        for tid in test_ids:
            row = cfg_row.to_dict()
            row["test_id"] = tid
            rows.append(row)
    grid = pd.DataFrame(rows)
    X = transform_features(grid, spec)

    preds = {}
    for name, model in model_result["models"].items():
        preds[f"pred_pass_{name}"] = model.predict_proba(X)[:, 1]
    for k, v in preds.items():
        grid[k] = v
    pred_cols = list(preds.keys())
    grid["pred_pass_blend"] = grid[pred_cols].mean(axis=1)
    return grid


# Penalty weights preserved from the surrogate-modeling notebook recommendation cell.
DEFAULT_RECOMMENDATION_PENALTIES = {
    "duration_s_mean": 0.05,
    "estimated_cost_mean": 0.03,
    "total_tokens_mean": 0.02,
}


def surrogate_recommendations(pred_grid: pd.DataFrame, configs: pd.DataFrame | None = None, penalty_weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """
    Rank configurations by predicted mean pass probability across all tasks
    minus small log-scaled penalties for slow/expensive/token-hungry configs.
    """
    penalty_weights = dict(penalty_weights or DEFAULT_RECOMMENDATION_PENALTIES)
    rec = (
        pred_grid.groupby("config_hash", as_index=False)
        .agg(pred_pass_mean=("pred_pass_blend", "mean"), pred_pass_min=("pred_pass_blend", "min"), pred_pass_std=("pred_pass_blend", "std"))
    )
    if configs is not None:
        keep = [c for c in ["config_hash", "config_label", "pass_rate", "n_tests", "duration_s_mean", "total_tokens_mean", "estimated_cost_mean", "estimated_cost_sum"] if c in configs.columns]
        rec = rec.merge(configs[keep].drop_duplicates("config_hash"), on="config_hash", how="left")

    penalty = pd.Series(0.0, index=rec.index)
    for col, weight in penalty_weights.items():
        if col in rec.columns:
            x = np.log1p(pd.to_numeric(rec[col], errors="coerce").fillna(rec[col].median()))
            denom = x.max() - x.min()
            norm = pd.Series(0.0, index=rec.index) if denom == 0 or not np.isfinite(denom) else (x - x.min()) / denom
            rec[col + "_norm"] = norm
            penalty = penalty + weight * norm
    rec["balanced_score"] = rec["pred_pass_mean"] - penalty
    return rec.sort_values(["balanced_score", "pred_pass_mean"], ascending=False).reset_index(drop=True)


def recommendations_by_task_group(pred_grid: pd.DataFrame, configs: pd.DataFrame | None = None, group_pattern: str = r"^([^_]+)_") -> pd.DataFrame:
    """Per-task-family config recommendations. Task family defaults to the prefix before the first underscore."""
    grid = pred_grid.copy()
    grid["task_group"] = grid["test_id"].astype(str).str.extract(group_pattern, expand=False).fillna(grid["test_id"])
    rec = (
        grid.groupby(["task_group", "config_hash"], as_index=False)
        .agg(pred_pass_mean=("pred_pass_blend", "mean"), pred_pass_min=("pred_pass_blend", "min"), n_tasks=("test_id", "nunique"))
    )
    if configs is not None:
        keep = [c for c in ["config_hash", "config_label", "pass_rate"] if c in configs.columns]
        rec = rec.merge(configs[keep].drop_duplicates("config_hash"), on="config_hash", how="left")
    rec["rank_in_group"] = rec.groupby("task_group")["pred_pass_mean"].rank(ascending=False, method="first")
    return rec.sort_values(["task_group", "rank_in_group"]).reset_index(drop=True)


def plot_predicted_config_task_heatmap(pred_grid: pd.DataFrame, top_configs: Sequence[str] | int = 15, label_col: str = "config_label"):
    """Heatmap of predicted pass probability per config × task for the top recommended configs."""
    import matplotlib.pyplot as plt
    if isinstance(top_configs, int):
        order = surrogate_recommendations(pred_grid)["config_hash"].head(top_configs).tolist()
    else:
        order = list(top_configs)
    heat = (
        pred_grid[pred_grid["config_hash"].isin(order)]
        .pivot_table(index="config_hash", columns="test_id", values="pred_pass_blend", aggfunc="mean")
        .loc[[c for c in order if c in set(pred_grid["config_hash"])]]
    )
    labels = heat.index
    if label_col in pred_grid.columns:
        m = pred_grid.drop_duplicates("config_hash").set_index("config_hash")[label_col]
        labels = [short_config_label(m.get(h, h), 4).replace("\n", " / ") for h in heat.index]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * heat.shape[1]), max(4, 0.32 * heat.shape[0])))
    im = ax.imshow(heat.values, aspect="auto", vmin=0, vmax=1, cmap=BLUE_CMAP)
    fig.colorbar(im, ax=ax, label="Predicted pass probability")
    ax.set_yticks(np.arange(len(heat.index)), labels=labels, fontsize=7)
    ax.set_xticks(np.arange(len(heat.columns)), labels=heat.columns, rotation=90, fontsize=7)
    ax.set_title("Predicted pass probability by config and task")
    fig.tight_layout()
    return fig, ax, heat


# -----------------------------------------------------------------------------
# RAG analysis extensions (paper Figure 4 and triage tools)
# Ported from rag_retrieval_analysis.
# -----------------------------------------------------------------------------

def plot_entry_passfail_diverging(entry_summary: pd.DataFrame):
    """Diverging pass/fail counts and pass-rate bars per RAG entry."""
    import matplotlib.pyplot as plt
    s = entry_summary.copy()
    s["n_passed"] = (s["pass_rate"] * s["n_runs"]).round().astype(int)
    s["n_failed"] = s["n_runs"] - s["n_passed"]
    s = s.sort_values("pass_rate")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(s) * 0.4)), gridspec_kw={"width_ratios": [2, 1]})
    y_pos = range(len(s))
    passed_bars = axes[0].barh(y_pos, s["n_passed"], color=STATUS_COLORS["passed"], label="Passed", edgecolor=BLUE_EDGE_COLOR, hatch=STATUS_HATCHES["passed"], linewidth=0.70)
    failed_bars = axes[0].barh(y_pos, -s["n_failed"], color=STATUS_COLORS["failed"], label="Failed", edgecolor=BLUE_EDGE_COLOR, hatch=STATUS_HATCHES["failed"], linewidth=0.70)
    axes[0].set_yticks(y_pos, labels=s["entry_id"], fontsize=8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("← Failed    |    Passed →")
    axes[0].set_title("Pass / Fail Counts per RAG Entry")
    axes[0].legend(loc="lower right")
    colors = ["#C6DBEF" if pr < 0.5 else "#6BAED6" if pr < 0.75 else "#2171B5" for pr in s["pass_rate"]]
    patterns = ["xx" if pr < 0.5 else ".." if pr < 0.75 else "///" for pr in s["pass_rate"]]
    rate_bars = axes[1].barh(y_pos, s["pass_rate"], color=colors, edgecolor=BLUE_EDGE_COLOR, linewidth=0.70)
    for patch, hatch in zip(rate_bars.patches, patterns):
        patch.set_hatch(hatch)
    axes[1].set_yticks([])
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Pass Rate")
    axes[1].set_title("Pass Rate")
    axes[1].axvline(0.5, color="#6BAED6", linestyle="--", alpha=0.65)
    axes[1].axvline(0.75, color="#2171B5", linestyle=":", alpha=0.75)
    for i, pr in enumerate(s["pass_rate"]):
        axes[1].text(pr + 0.02, i, f"{pr:.0%}", va="center", fontsize=8)
    fig.tight_layout()
    return fig, axes


def rag_cooccurrence_matrix(runs: pd.DataFrame, failed_only: bool = True) -> pd.DataFrame:
    """Entry × entry co-occurrence counts (diagonal = solo appearance count) in failed runs."""
    df = runs[~runs["passed"]] if failed_only and "passed" in runs.columns else runs
    all_ids = sorted(set(itertools.chain.from_iterable(runs["rag_entry_ids"].apply(_as_list))))
    cooccur = pd.DataFrame(0, index=all_ids, columns=all_ids, dtype=int)
    for _, row in df.iterrows():
        eids = [e for e in set(_as_list(row.get("rag_entry_ids"))) if e in cooccur.index]
        for a in eids:
            cooccur.loc[a, a] += 1
            for b in eids:
                if a != b:
                    cooccur.loc[a, b] += 1
    active = cooccur.index[cooccur.sum(axis=1) > 0]
    return cooccur.loc[active, active]


def plot_rag_cooccurrence(runs: pd.DataFrame, failed_only: bool = True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    mat = rag_cooccurrence_matrix(runs, failed_only=failed_only)
    fig, ax = plt.subplots(figsize=(max(8, len(mat) * 0.6), max(6, len(mat) * 0.5)))
    if mat.empty:
        ax.text(0.5, 0.5, "No failed runs with RAG entries", ha="center", va="center")
        ax.axis("off")
        return fig, ax, mat
    mask = np.zeros_like(mat, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(mat, mask=mask, annot=True, fmt="d", cmap=BLUE_CMAP, linewidths=0.5, ax=ax, cbar_kws={"label": "Co-occurrences in failed runs"})
    ax.set_title("RAG Entry Co-occurrence in Failed Runs\n(diagonal = solo failure count)")
    fig.tight_layout()
    return fig, ax, mat


def plot_entry_hyperparameter_interaction(runs: pd.DataFrame, params: Sequence[str] | None = None, top_entries: int = 8, max_params: int = 4):
    """Entry pass rate vs numeric hyperparameter value, one figure per hyperparameter. Returns {param: (fig, ax)}."""
    import matplotlib.pyplot as plt

    cfg_cols = sorted(c for c in runs.columns if c.startswith("cfg_"))
    if params is None:
        params = [c for c in cfg_cols if runs[c].nunique(dropna=False) > 1 and pd.api.types.is_numeric_dtype(_numeric_if_possible(runs[c]))]
    params = list(params)[:max_params]

    rows = []
    for _, row in runs.iterrows():
        for eid in set(_as_list(row.get("rag_entry_ids"))):
            rows.append({"entry_id": eid, "passed_int": int(bool(row.get("passed"))), **{p: row.get(p) for p in params}})
    if not rows:
        return {}
    edf = pd.DataFrame(rows)
    top_eids = edf["entry_id"].value_counts().head(top_entries).index.tolist()

    out = {}
    for p in params:
        edf[p] = pd.to_numeric(edf[p], errors="coerce")
        if edf[p].dropna().nunique() < 2:
            continue
        sub = edf[edf["entry_id"].isin(top_eids)]
        pr = sub.groupby(["entry_id", p])["passed_int"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "pass_rate", "count": "n"})
        if pr.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for eid in top_eids:
            d = pr[pr["entry_id"] == eid].sort_values(p)
            if not d.empty:
                ax.plot(d[p], d["pass_rate"], marker="o", label=eid, linewidth=1.5, alpha=0.8)
        ax.set_xlabel(p.replace("cfg_", ""))
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Entry Pass Rate vs {p.replace('cfg_', '')}")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        out[p] = (fig, ax)
    return out


def rag_risk_table(runs: pd.DataFrame, entry_summary: pd.DataFrame | None = None, sig: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Consolidated RAG-entry risk ranking (rag_retrieval_analysis section 10 method):
    risk = 0.40*fail_rate + 0.25*normalized total failures + 0.20*(1 - p) + 0.15*normalized mean tokens.
    """
    from collections import Counter

    if entry_summary is None:
        entry_summary = rag_entry_summary(runs)
    if sig is None:
        sig = rag_significance_table(runs)
    if entry_summary.empty:
        return pd.DataFrame()

    risk = entry_summary.set_index("entry_id").copy()
    # total check failures across runs that retrieved each entry
    totals = {}
    dominant = {}
    for eid in risk.index:
        mask = runs["rag_entry_ids"].apply(lambda xs: eid in set(_as_list(xs)))
        sub = runs[mask]
        totals[eid] = float(pd.to_numeric(sub.get("num_failures"), errors="coerce").fillna(0).sum()) if not sub.empty else 0.0
        ft_counts = Counter(ft for fts in sub.get("failure_types", pd.Series(dtype=object)) for ft in _as_list(fts))
        dominant[eid] = ft_counts.most_common(1)[0][0] if ft_counts else "none"
    risk["total_failures"] = pd.Series(totals)
    risk["dominant_failure_type"] = pd.Series(dominant)

    if sig is not None and not sig.empty:
        risk = risk.join(sig.set_index("entry_id")[["p_value", "q_value_bh", "delta_pass_rate"]], how="left")

    tf_max = risk["total_failures"].max()
    tok_max = pd.to_numeric(risk.get("mean_total_tokens"), errors="coerce").max()
    risk["risk_score"] = (
        0.40 * risk["fail_rate"].fillna(0)
        + 0.25 * (risk["total_failures"] / tf_max if tf_max and tf_max > 0 else 0)
        + 0.20 * (1 - risk.get("p_value", pd.Series(1.0, index=risk.index)).fillna(1))
        + 0.15 * (pd.to_numeric(risk.get("mean_total_tokens"), errors="coerce").fillna(0) / tok_max if tok_max and tok_max > 0 else 0)
    )
    return risk.sort_values("risk_score", ascending=False).reset_index()


def plot_rag_risk(risk: pd.DataFrame, top: int = 30):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, max(4, min(top, len(risk)) * 0.4)))
    sub = risk.head(top).iloc[::-1]
    norm = sub["risk_score"] / max(risk["risk_score"].max(), 1e-9)
    colors = plt.get_cmap(BLUE_CMAP)(0.30 + 0.65 * norm)
    cats = sub["entry_id"].astype(str).tolist()
    bars = ax.barh(sub["entry_id"], sub["risk_score"], color=colors, edgecolor=BLUE_EDGE_COLOR, linewidth=0.70)
    apply_bar_hatches(bars.patches, cats)
    ax.set_xlabel("Composite risk score")
    ax.set_title("RAG entry risk ranking (higher = more concerning)")
    fig.tight_layout()
    return fig, ax


def classify_rag_entries(sig: pd.DataFrame, p_threshold: float = 0.05, delta_threshold: float = 0.0) -> pd.DataFrame:
    """Label each entry helpful (green) / toxic (red) / neutral (gray) from the significance table."""
    out = sig.copy()
    def label(row):
        if row["p_value"] <= p_threshold and row["delta_pass_rate"] > delta_threshold:
            return "helpful"
        if row["p_value"] <= p_threshold and row["delta_pass_rate"] < -delta_threshold:
            return "toxic"
        return "neutral"
    out["classification"] = out.apply(label, axis=1)
    return out

def classify_rag_entries_q(
    sig: pd.DataFrame,
    q_threshold: float = 0.05,
    delta_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Label each RAG entry using Benjamini-Hochberg q-values.

    Helpful: q <= q_threshold and delta_pass_rate > delta_threshold
    Toxic:   q <= q_threshold and delta_pass_rate < -delta_threshold
    Neutral: otherwise
    """
    out = sig.copy()

    if "q_value_bh" not in out.columns:
        if "p_value" not in out.columns:
            raise ValueError("Expected either 'q_value_bh' or 'p_value' in sig.")
        out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())

    def label(row):
        q = row["q_value_bh"]
        delta = row["delta_pass_rate"]

        if pd.isna(q) or pd.isna(delta):
            return "neutral"
        if q <= q_threshold and delta > delta_threshold:
            return "helpful"
        if q <= q_threshold and delta < -delta_threshold:
            return "toxic"
        return "neutral"

    out["classification"] = out.apply(label, axis=1)
    return out

def _volcano_annotation_candidates(
    df: pd.DataFrame,
    significance_col: str,
    annotate_top: int,
) -> pd.DataFrame:
    if annotate_top <= 0 or df.empty:
        return pd.DataFrame(columns=df.columns)
    frames = [
        df.nsmallest(max(1, annotate_top // 2), significance_col),
        df.nsmallest(3, "delta_pass_rate"),
        df.nlargest(3, "delta_pass_rate"),
    ]
    return pd.concat(frames).drop_duplicates("entry_id").head(annotate_top)


def _spread_label_positions(
    values: Sequence[float],
    lower: float,
    upper: float,
    min_separation: float,
) -> np.ndarray:
    """Return ordered y positions that stay in bounds and do not overlap."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        return np.asarray([float(np.clip(values[0], lower, upper))])

    order = np.argsort(values)
    ordered = np.clip(values[order], lower, upper)
    available = max(upper - lower, 0.0)
    sep = min(float(min_separation), available / max(len(ordered) - 1, 1))

    # Forward pass.
    positioned = ordered.copy()
    for i in range(1, len(positioned)):
        positioned[i] = max(positioned[i], positioned[i - 1] + sep)

    # Pull the stack back into the top boundary, then make a backward pass.
    if positioned[-1] > upper:
        positioned -= positioned[-1] - upper
    for i in range(len(positioned) - 2, -1, -1):
        positioned[i] = min(positioned[i], positioned[i + 1] - sep)
    if positioned[0] < lower:
        positioned += lower - positioned[0]

    result = np.empty_like(positioned)
    result[order] = positioned
    return result


def _set_symmetric_volcano_xlim(ax, df: pd.DataFrame) -> None:
    """Use a symmetric effect-size axis with space reserved for labels."""
    values = pd.to_numeric(df["delta_pass_rate"], errors="coerce").dropna()
    extent = max(float(values.abs().max()) if not values.empty else 0.0, 0.05)
    ax.set_xlim(-1.24 * extent, 1.24 * extent)


def _annotate_volcano_labels(ax, anno: pd.DataFrame, y_col: str) -> None:
    """Place labels in deterministic left/right lanes inside the axes.

    The previous offset/``adjustText`` approach could push long entry names far
    outside the plotting region when the figure was saved with a tight bounding
    box.  This layout keeps every label inside the axes, aligns labels in two
    clean columns, and uses leader lines back to the corresponding points.
    """
    if anno.empty:
        return

    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    x_span = max(x_right - x_left, np.finfo(float).eps)
    y_span = max(y_top - y_bottom, np.finfo(float).eps)

    label_x = {
        "left": x_left + 0.018 * x_span,
        "right": x_right - 0.018 * x_span,
    }
    y_lower = y_bottom + 0.045 * y_span
    y_upper = y_top - 0.055 * y_span
    min_sep = max(0.055 * y_span, 0.16)

    work = anno.copy()
    work["_side"] = np.where(work["delta_pass_rate"] >= 0, "right", "left")

    for side, sub in work.groupby("_side", sort=False):
        sub = sub.sort_values(y_col).copy()
        target_y = _spread_label_positions(
            sub[y_col].to_numpy(dtype=float),
            lower=y_lower,
            upper=y_upper,
            min_separation=min_sep,
        )
        ha = "right" if side == "right" else "left"

        for (_, row), y_text in zip(sub.iterrows(), target_y):
            x = float(row["delta_pass_rate"])
            y = float(row[y_col])
            ax.annotate(
                str(row["entry_id"]),
                xy=(x, y),
                xytext=(label_x[side], float(y_text)),
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=7.1,
                fontweight="semibold",
                color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.17",
                    facecolor="white",
                    edgecolor="#777777",
                    linewidth=0.42,
                    alpha=0.94,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#8A8A8A",
                    lw=0.52,
                    alpha=0.85,
                    shrinkA=2,
                    shrinkB=3,
                    connectionstyle="arc3,rad=0",
                ),
                annotation_clip=True,
                zorder=6,
            )


def _draw_volcano_points(ax, df: pd.DataFrame, sizes: np.ndarray) -> None:
    # Neutral points first so significant helpful/toxic associations stay visible.
    for classification in ("neutral", "toxic", "helpful"):
        sub = df[df["classification"] == classification]
        if sub.empty:
            continue
        idx = sub.index.to_numpy()
        pos = df.index.get_indexer(idx)
        ax.scatter(
            sub["delta_pass_rate"],
            sub[df.attrs["y_col"]],
            s=sizes[pos],
            c=RAG_EFFECT_COLORS[classification],
            alpha=0.72 if classification != "neutral" else 0.58,
            edgecolors="#333333",
            linewidths=0.55,
            zorder=4 if classification != "neutral" else 3,
        )


def _volcano_legend(ax, threshold_label: str):
    import matplotlib.pyplot as plt

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=RAG_EFFECT_COLORS["helpful"],
            markeredgecolor="#333333", markersize=8, label=f"Positive ΔP ({threshold_label})",
        ),
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=RAG_EFFECT_COLORS["neutral"],
            markeredgecolor="#333333", markersize=8, label="No clear association",
        ),
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=RAG_EFFECT_COLORS["toxic"],
            markeredgecolor="#333333", markersize=8, label=f"Negative ΔP ({threshold_label})",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8,
        frameon=True,
        framealpha=0.94,
        ncol=1,
        borderaxespad=0.35,
    )


def plot_rag_volcano_publication(
    sig: pd.DataFrame,
    min_with_entry: int = 1,
    p_lines: Sequence[float] = (0.05, 0.10),
    p_threshold: float = 0.05,
    delta_threshold: float = 0.0,
    annotate_top: int = 12,
    title: str = "Volcano Plot: RAG Entry Impact on Pass Rate\n(p-value; size = sample count with entry)",
):
    """Publication p-value volcano plot with green/gray/red effect encoding."""
    import matplotlib.pyplot as plt

    required = {"entry_id", "n_with_entry", "delta_pass_rate", "p_value"}
    missing = required - set(sig.columns)
    if missing:
        raise ValueError(f"sig is missing required columns: {sorted(missing)}")

    df = sig[sig["n_with_entry"] >= min_with_entry].copy()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if df.empty:
        ax.text(0.5, 0.5, "No significance results", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    tiny = np.finfo(float).tiny
    df["neg_log10_p"] = -np.log10(pd.to_numeric(df["p_value"], errors="coerce").clip(lower=tiny))
    df = classify_rag_entries(df, p_threshold=p_threshold, delta_threshold=delta_threshold)
    df = df.reset_index(drop=True)
    df.attrs["y_col"] = "neg_log10_p"

    size_denominator = max(float(df["n_with_entry"].max()), 1.0)
    sizes = 55 + 850 * (df["n_with_entry"].to_numpy(dtype=float) / size_denominator)
    _draw_volcano_points(ax, df, sizes)

    line_styles = ["--", ":", "-."]
    for p_value, ls in zip(p_lines, line_styles):
        if p_value <= 0:
            continue
        y = -np.log10(max(float(p_value), tiny))
        ax.axhline(y, linestyle=ls, linewidth=0.9, color="#666666", zorder=1)
        ax.text(
            0.50, y, f"p={p_value:g}", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#555555", va="bottom", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.4),
        )
    ax.axvline(0, linestyle="-", linewidth=0.8, color="black", alpha=0.45, zorder=1)

    ymax = max(float(df["neg_log10_p"].max()), max([-np.log10(p) for p in p_lines if p > 0], default=0.0))
    ax.set_ylim(-0.05, ymax * 1.10 + 0.25)
    _set_symmetric_volcano_xlim(ax, df)
    anno = _volcano_annotation_candidates(df, "p_value", annotate_top)
    _annotate_volcano_labels(ax, anno, "neg_log10_p")
    _volcano_legend(ax, f"p ≤ {p_threshold:g}")

    ax.set_xlabel("Δ Pass Rate (with entry − without entry)")
    ax.set_ylabel(r"$-\log_{10}$(p-value)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_rag_q_volcano_publication(
    sig: pd.DataFrame,
    min_with_entry: int = 1,
    q_lines: Sequence[float] = (0.05, 0.10),
    q_threshold: float = 0.05,
    delta_threshold: float = 0.0,
    annotate_top: int = 12,
    title: str = "Volcano Plot: RAG Entry Impact on Pass Rate\n(BH q-value; size = sample count with entry)",
):
    """Publication BH q-value volcano plot with bounded, aligned annotations."""
    import matplotlib.pyplot as plt

    required_cols = {"entry_id", "n_with_entry", "delta_pass_rate"}
    missing = required_cols - set(sig.columns)
    if missing:
        raise ValueError(f"sig is missing required columns: {sorted(missing)}")

    df = sig.copy()
    if "q_value_bh" not in df.columns:
        if "p_value" not in df.columns:
            raise ValueError("Expected either 'q_value_bh' or 'p_value' in sig.")
        df["q_value_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
    df = df[df["n_with_entry"] >= min_with_entry].copy()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    if df.empty:
        ax.text(0.5, 0.5, "No q-value significance results", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    tiny = np.finfo(float).tiny
    df["neg_log10_q"] = -np.log10(pd.to_numeric(df["q_value_bh"], errors="coerce").clip(lower=tiny))
    df = classify_rag_entries_q(df, q_threshold=q_threshold, delta_threshold=delta_threshold)
    df = df.reset_index(drop=True)
    df.attrs["y_col"] = "neg_log10_q"

    size_denominator = max(float(df["n_with_entry"].max()), 1.0)
    sizes = 55 + 850 * (df["n_with_entry"].to_numpy(dtype=float) / size_denominator)
    _draw_volcano_points(ax, df, sizes)

    line_styles = ["--", ":", "-."]
    for q_value, ls in zip(q_lines, line_styles):
        if q_value <= 0:
            continue
        y = -np.log10(max(float(q_value), tiny))
        ax.axhline(y, linestyle=ls, linewidth=0.9, color="#666666", zorder=1)
        ax.text(
            0.50, y, f"q={q_value:g}", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#555555", va="bottom", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.4),
        )
    ax.axvline(0, linestyle="-", linewidth=0.8, color="black", alpha=0.45, zorder=1)

    ymax = max(float(df["neg_log10_q"].max()), max([-np.log10(q) for q in q_lines if q > 0], default=0.0))
    ax.set_ylim(-0.05, ymax * 1.10 + 0.25)
    _set_symmetric_volcano_xlim(ax, df)
    anno = _volcano_annotation_candidates(df, "q_value_bh", annotate_top)
    _annotate_volcano_labels(ax, anno, "neg_log10_q")
    _volcano_legend(ax, f"q ≤ {q_threshold:g}")

    ax.set_xlabel("Δ Pass Rate (with entry − without entry)")
    ax.set_ylabel(r"$-\log_{10}$(BH q-value)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Test-quality screening: which TESTS (not configs) look broken or pathological
# -----------------------------------------------------------------------------

def summarize_by_test(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-test aggregate: pass rate, configs covered, error rate, dominant failure types."""
    from collections import Counter
    df = runs.copy()
    if "passed_int" not in df and "passed" in df:
        df["passed_int"] = df["passed"].astype(int)
    df["is_error"] = df.get("status", pd.Series("", index=df.index)).astype(str).str.lower().eq("error")
    grp = df.groupby("test_id")
    out = grp.agg(
        n_runs=("run_key", "count"),
        n_configs=("config_hash", "nunique"),
        n_passed=("passed_int", "sum"),
        pass_rate=("passed_int", "mean"),
        error_rate=("is_error", "mean"),
        mean_duration_s=("duration_s", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        mean_num_failures=("num_failures", "mean"),
    ).reset_index()
    lo, hi = wilson_interval(out["n_passed"], out["n_runs"])
    out["pass_ci_low"], out["pass_ci_high"] = lo, hi
    dom = {}
    for tid, sub in grp:
        counts = Counter(ft for fts in sub.get("failure_types", pd.Series(dtype=object)) for ft in _as_list(fts))
        dom[tid] = counts.most_common(1)[0][0] if counts else "none"
    out["dominant_failure_type"] = out["test_id"].map(dom)
    return out.sort_values("pass_rate")


def flag_suspect_tests(test_summary: pd.DataFrame, min_configs: int = 1, universal_fail_threshold: float = 0.0, near_universal_fail_threshold: float = 0.1, error_rate_threshold: float = 0.5, pass_rate_threshold: float = 1.0) -> pd.DataFrame:
    """
    Flag tests that look like they contain bugs or unreasonable expectations rather
    than informative benchmarks:
      - fail (or nearly fail) across every configuration tested,
      - mostly end in execution errors instead of check failures,
      - pass for every configuration (no discriminative power).
    """
    out = test_summary.copy()
    reasons = []
    for _, r in out.iterrows():
        rs = []
        if r["n_configs"] >= min_configs and r["pass_rate"] <= universal_fail_threshold:
            rs.append("fails_for_every_config_possible_test_bug")
        elif r["n_configs"] >= min_configs and r["pass_rate"] <= near_universal_fail_threshold:
            rs.append("near_universal_failure")
        if r["error_rate"] >= error_rate_threshold:
            rs.append("dominated_by_execution_errors")
        if r["n_configs"] >= min_configs and r["pass_rate"] >= pass_rate_threshold:
            rs.append("passes_everywhere_no_signal")
        reasons.append(";".join(rs))
    out["suspect_reasons"] = reasons
    out["is_suspect"] = out["suspect_reasons"].astype(bool)
    return out.sort_values(["is_suspect", "pass_rate"], ascending=[False, True])


def plot_test_pass_rates(
    test_summary: pd.DataFrame,
    show_counts: bool = True,
    count_label_template: str = "{n_runs:g} runs | {n_configs:g} configs",
):
    """Per-test pass rate with Wilson 95% CIs and benchmark coverage labels.

    Parameters
    ----------
    test_summary:
        Output from :func:`summarize_by_test`.
    show_counts:
        When true, add a right-aligned metadata column showing the number of
        executions and distinct configurations represented by each bar.
    count_label_template:
        Format string receiving ``n_runs`` and ``n_configs``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    required = {
        "test_id", "pass_rate", "pass_ci_low", "pass_ci_high",
        "n_runs", "n_configs",
    }
    missing = required - set(test_summary.columns)
    if missing:
        raise ValueError(f"test_summary is missing required columns: {sorted(missing)}")

    s = test_summary.sort_values("pass_rate").reset_index(drop=True)
    figure_width = 10.0 if show_counts else 8.0
    fig, ax = plt.subplots(figsize=(figure_width, max(4, len(s) * 0.34)))

    if s.empty:
        ax.text(0.5, 0.5, "No benchmark tests available", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    y = np.arange(len(s))
    err = np.vstack([
        (s["pass_rate"] - s["pass_ci_low"]).clip(lower=0),
        (s["pass_ci_high"] - s["pass_rate"]).clip(lower=0),
    ])
    colors = ["#C6DBEF" if pr < 0.5 else "#6BAED6" if pr < 0.75 else "#2171B5" for pr in s["pass_rate"]]
    patterns = ["xx" if pr < 0.5 else ".." if pr < 0.75 else "///" for pr in s["pass_rate"]]
    bars = ax.barh(
        y,
        s["pass_rate"],
        xerr=err,
        color=colors,
        edgecolor=BLUE_EDGE_COLOR,
        linewidth=0.70,
        error_kw={
            "elinewidth": 0.9,
            "alpha": 0.75,
            "ecolor": BLUE_EDGE_COLOR,
            "capsize": 1.8,
        },
    )
    for patch, hatch in zip(bars.patches, patterns):
        patch.set_hatch(hatch)

    ax.set_yticks(y, labels=s["test_id"], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Pass rate across all configurations (95% Wilson CI)")
    ax.set_title("Benchmark test difficulty")

    if show_counts:
        # x in axes coordinates and y in data coordinates creates a clean,
        # aligned metadata column without changing the 0--1 pass-rate scale.
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        for yi, row in s.iterrows():
            label = count_label_template.format(
                n_runs=float(row["n_runs"]),
                n_configs=float(row["n_configs"]),
            )
            ax.text(
                1.025,
                yi,
                label,
                transform=transform,
                ha="left",
                va="center",
                fontsize=7.6,
                color="#333333",
                clip_on=False,
            )
        ax.text(
            1.025,
            1.012,
            "Benchmark coverage",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.8,
            fontweight="semibold",
            color="#333333",
            clip_on=False,
        )
        fig.subplots_adjust(right=0.78)
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    else:
        fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Configuration filtering: drop or keep-only runs by cfg_* parameter value(s)
# -----------------------------------------------------------------------------

def _resolve_cfg_param_map(runs: pd.DataFrame, params: Mapping[str, Any] | None) -> dict[str, list]:
    """Normalize a {param: value_or_values} mapping to {cfg_column: [values]}.

    Accepts bare names (``llm_model``) or full column names (``cfg_llm_model``).
    Values may be a single scalar or any list/tuple/set of scalars. Raises a
    KeyError (listing available cfg_ columns) if a param doesn't resolve to a
    real column, so typos fail loudly instead of silently filtering nothing.
    """
    out: dict[str, list] = {}
    for key, values in (params or {}).items():
        col = key if key.startswith("cfg_") else f"cfg_{key}"
        if col not in runs.columns:
            available = sorted(c for c in runs.columns if c.startswith("cfg_"))
            raise KeyError(f"Unknown config parameter '{key}' (looked for column '{col}'). Available cfg_ columns: {available}")
        out[col] = list(values) if isinstance(values, (list, tuple, set)) else [values]
    return out


def filter_by_config_params(
    runs: pd.DataFrame,
    exclude: Mapping[str, Any] | None = None,
    include_only: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop or keep-only runs based on their cfg_* configuration values.

    Parameters
    ----------
    runs : DataFrame with cfg_* columns (e.g. cfg_llm_model, cfg_llm_temperature).
    exclude : {param: value(s)} to remove, e.g. {"llm_model": ["gpt-35", "claude-sonnet-3-7"]}
        drops every run whose cfg_llm_model is one of those values.
    include_only : {param: value(s)} to keep, e.g. {"llm_temperature": [1.0, 0.75]}
        drops every run whose cfg_llm_temperature is NOT one of those values.
        Multiple params in `include_only` are combined with AND (a run must satisfy
        every listed param's criterion to be kept).
    verbose : print how many runs each rule removed.

    Keys accept either the bare parameter name ("llm_model") or the full column
    name ("cfg_llm_model"). Values accept a single value or a list of values.
    Nothing is mutated in place; a filtered copy is returned.
    """
    df = runs.copy()
    n0 = len(df)

    for col, values in _resolve_cfg_param_map(runs, exclude).items():
        mask = df[col].isin(values)
        if verbose and mask.any():
            print(f"[filter_by_config_params] excluding {int(mask.sum())} runs where {col.replace('cfg_', '')} in {values}")
        df = df[~mask]

    for col, values in _resolve_cfg_param_map(runs, include_only).items():
        n_before = len(df)
        df = df[df[col].isin(values)]
        if verbose:
            print(f"[filter_by_config_params] keeping only {len(df)}/{n_before} runs where {col.replace('cfg_', '')} in {values}")

    if verbose:
        print(f"[filter_by_config_params] kept {len(df)} / {n0} runs overall.")

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Outlier screening: in-memory port of screen_eval_outliers.py
# (hard thresholds + robust MAD/IQR z-scores, globally and within each test_id)
# -----------------------------------------------------------------------------

OUTLIER_RESULT_METRICS = [
    "duration_s", "total_tokens", "input_tokens", "output_tokens",
    "llm_calls", "tool_calls", "estimated_cost", "tasks_completed",
    "num_rag_retrievals", "num_unique_rag_entries",
]


@dataclass
class OutlierScreenConfig:
    max_duration_s: float | None = 50_000.0
    max_total_tokens: float | None = None
    max_input_tokens: float | None = None
    max_output_tokens: float | None = None
    max_llm_calls: float | None = None
    max_tool_calls: float | None = None
    max_cost: float | None = None
    max_rag_retrievals: float | None = None
    flag_errors: bool = False
    flag_budget_exceeded: bool = True
    flag_zero_metric_failures: bool = False
    flag_no_pass_suites: bool = False
    min_suite_pass_rate: float | None = None
    max_suite_duration_s: float | None = 50_000.0
    robust: bool = True
    robust_z: float = 6.0
    min_n_for_robust: int = 8


def robust_zscore(s: pd.Series) -> pd.Series:
    """MAD-based robust z, falling back to IQR, then standard z."""
    x = pd.to_numeric(s, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if pd.isna(med):
        return pd.Series(np.nan, index=s.index)
    if mad and not pd.isna(mad):
        return 0.6745 * (x - med) / mad
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr and not pd.isna(iqr):
        return (x - med) / (iqr / 1.349)
    std = x.std(skipna=True)
    if std and not pd.isna(std):
        return (x - x.mean(skipna=True)) / std
    return pd.Series(np.nan, index=s.index)


def _append_reasons(reasons: list[list[str]], mask: pd.Series, label: str) -> None:
    for i in np.where(mask.fillna(False).to_numpy())[0]:
        reasons[i].append(label)


def screen_outliers(runs: pd.DataFrame, config: OutlierScreenConfig | None = None) -> dict[str, pd.DataFrame]:
    """
    Screen the canonical runs table for pathological/outlier runs and bad suites.

    Returns dict with:
        runs_flagged        all rows + flag columns
        result_outliers     flagged rows only
        suites_flagged      per-suite summary + flags
        suite_outliers      flagged suites only
        removal_candidates  combined review targets (never deletes anything)
    """
    cfg = config or OutlierScreenConfig()
    df = runs.copy().reset_index(drop=True)

    # --- hard threshold flags ---
    reasons: list[list[str]] = [[] for _ in range(len(df))]
    thresholds = {
        "duration_s": cfg.max_duration_s,
        "total_tokens": cfg.max_total_tokens,
        "input_tokens": cfg.max_input_tokens,
        "output_tokens": cfg.max_output_tokens,
        "llm_calls": cfg.max_llm_calls,
        "tool_calls": cfg.max_tool_calls,
        "estimated_cost": cfg.max_cost,
        "num_rag_retrievals": cfg.max_rag_retrievals,
    }
    for col, threshold in thresholds.items():
        if threshold is not None and col in df.columns:
            _append_reasons(reasons, pd.to_numeric(df[col], errors="coerce") > threshold, f"{col}>{threshold:g}")
    if cfg.flag_errors and "status" in df.columns:
        _append_reasons(reasons, df["status"].astype(str).str.lower().eq("error"), "status=error")
    if cfg.flag_budget_exceeded and "error_message" in df.columns:
        _append_reasons(reasons, df["error_message"].fillna("").astype(str).str.contains("budget.*exceeded|budget_exceeded", case=False, regex=True), "budget_exceeded_error")
    if cfg.flag_zero_metric_failures and "status" in df.columns:
        status_bad = df["status"].astype(str).str.lower().isin(["failed", "error"])
        zeroish = pd.Series(False, index=df.index)
        for col in ["duration_s", "total_tokens", "llm_calls", "tool_calls"]:
            if col in df.columns:
                zeroish = zeroish | pd.to_numeric(df[col], errors="coerce").fillna(0).eq(0)
        _append_reasons(reasons, status_bad & zeroish, "failed_or_error_with_zero_metrics")
    df["hard_outlier_reasons"] = [";".join(r) for r in reasons]
    df["hard_outlier"] = df["hard_outlier_reasons"].astype(bool)

    # --- robust statistical flags, globally and within each test_id ---
    reasons = [[] for _ in range(len(df))]
    if cfg.robust:
        for metric in OUTLIER_RESULT_METRICS:
            if metric not in df.columns:
                continue
            if df[metric].notna().sum() >= cfg.min_n_for_robust:
                rz = robust_zscore(df[metric])
                df[f"{metric}_robust_z_global"] = rz
                _append_reasons(reasons, rz > cfg.robust_z, f"robust_high_{metric}_global_z>{cfg.robust_z:g}")
            if "test_id" in df.columns:
                z_by_test = pd.Series(np.nan, index=df.index, dtype=float)
                for _, idx in df.groupby("test_id", dropna=False).groups.items():
                    idx = list(idx)
                    if df.loc[idx, metric].notna().sum() >= cfg.min_n_for_robust:
                        z_by_test.loc[idx] = robust_zscore(df.loc[idx, metric])
                df[f"{metric}_robust_z_by_test"] = z_by_test
                _append_reasons(reasons, z_by_test > cfg.robust_z, f"robust_high_{metric}_within_test_z>{cfg.robust_z:g}")
    df["robust_outlier_reasons"] = [";".join(r) for r in reasons]
    df["robust_outlier"] = df["robust_outlier_reasons"].astype(bool)
    df["outlier_reasons"] = (df["hard_outlier_reasons"] + ";" + df["robust_outlier_reasons"]).str.strip(";").str.replace(";;", ";", regex=False)
    df["is_outlier"] = df["outlier_reasons"].astype(bool)

    # --- per-suite summary and flags ---
    suites = pd.DataFrame()
    if "suite_run_id" in df.columns:
        status = df.get("status", pd.Series("", index=df.index)).astype(str).str.lower()
        suites = df.assign(_failed=status.eq("failed"), _error=status.eq("error")).groupby("suite_run_id").agg(
            source_file=("source_file", "first") if "source_file" in df.columns else ("suite_run_id", "first"),
            config_hash=("config_hash", "first"),
            workspace_dir=("workspace_dir", "first") if "workspace_dir" in df.columns else ("suite_run_id", "first"),
            n_tests=("run_key", "count"),
            n_passed=("passed", "sum"),
            n_failed=("_failed", "sum"),
            n_errors=("_error", "sum"),
            suite_duration_s=("suite_duration_ms", "first") if "suite_duration_ms" in df.columns else ("duration_s", "sum"),
        ).reset_index()
        if "suite_duration_ms" in df.columns:
            suites["suite_duration_s"] = pd.to_numeric(suites["suite_duration_s"], errors="coerce") / 1000.0
        suites["suite_pass_rate"] = suites["n_passed"] / suites["n_tests"].clip(lower=1)
        s_reasons: list[list[str]] = [[] for _ in range(len(suites))]
        if cfg.flag_no_pass_suites:
            _append_reasons(s_reasons, (suites["n_tests"] > 0) & (suites["n_passed"] == 0), "suite_has_zero_passes")
            _append_reasons(s_reasons, (suites["n_tests"] > 0) & (suites["n_errors"] == suites["n_tests"]), "suite_all_errors")
        if cfg.min_suite_pass_rate is not None:
            _append_reasons(s_reasons, suites["suite_pass_rate"] < cfg.min_suite_pass_rate, f"suite_pass_rate<{cfg.min_suite_pass_rate:g}")
        if cfg.max_suite_duration_s is not None:
            _append_reasons(s_reasons, pd.to_numeric(suites["suite_duration_s"], errors="coerce") > cfg.max_suite_duration_s, f"suite_duration_s>{cfg.max_suite_duration_s:g}")
        suites["suite_outlier_reasons"] = [";".join(r) for r in s_reasons]
        suites["is_outlier"] = suites["suite_outlier_reasons"].astype(bool)

    result_outliers = df[df["is_outlier"]].copy()
    suite_outliers = suites[suites["is_outlier"]].copy() if not suites.empty else pd.DataFrame()

    # --- removal/review candidates (conservative: report only) ---
    cand_rows = []
    for _, r in suite_outliers.iterrows():
        cand_rows.append({
            "scope": "suite",
            "suggested_review_target": r.get("workspace_dir") or r.get("source_file"),
            "source_file": r.get("source_file"), "suite_run_id": r.get("suite_run_id"),
            "config_hash": r.get("config_hash"), "test_id": None, "status": None,
            "reasons": r.get("suite_outlier_reasons"),
            "duration_s": None, "total_tokens": None, "estimated_cost": None,
        })
    for _, r in result_outliers.iterrows():
        cand_rows.append({
            "scope": "test_result",
            "suggested_review_target": r.get("source_file"),
            "source_file": r.get("source_file"), "suite_run_id": r.get("suite_run_id"),
            "config_hash": r.get("config_hash"), "test_id": r.get("test_id"), "status": r.get("status"),
            "reasons": r.get("outlier_reasons"),
            "duration_s": r.get("duration_s"), "total_tokens": r.get("total_tokens"), "estimated_cost": r.get("estimated_cost"),
        })
    candidates = pd.DataFrame(cand_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(["scope", "suggested_review_target", "suite_run_id", "test_id"], na_position="last")

    return {
        "runs_flagged": df,
        "result_outliers": result_outliers,
        "suites_flagged": suites,
        "suite_outliers": suite_outliers,
        "removal_candidates": candidates,
    }


def plot_outlier_overview(screen: Mapping[str, pd.DataFrame], metrics: Sequence[str] = ("duration_s", "total_tokens")):
    """Strip plots of key metrics per test with flagged outliers highlighted in red."""
    import matplotlib.pyplot as plt
    df = screen["runs_flagged"]
    metrics = [m for m in metrics if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), max(4, df["test_id"].nunique() * 0.28)), squeeze=False)
    test_order = sorted(df["test_id"].dropna().unique())
    y_map = {t: i for i, t in enumerate(test_order)}
    for ax, metric in zip(axes[0], metrics):
        good = df[~df["is_outlier"]]
        bad = df[df["is_outlier"]]
        ax.scatter(pd.to_numeric(good[metric], errors="coerce"), good["test_id"].map(y_map), s=18, alpha=0.5, color="#9ECAE1", label="kept")
        ax.scatter(pd.to_numeric(bad[metric], errors="coerce"), bad["test_id"].map(y_map), s=45, alpha=0.9, color="#08519C", marker="x", label="flagged")
        ax.set_yticks(range(len(test_order)), labels=test_order, fontsize=7)
        ax.set_xlabel(metric)
        ax.set_xscale("log")
        ax.set_title(f"{metric} by test (dark blue = flagged outlier)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


def write_outlier_report(screen: Mapping[str, pd.DataFrame], outdir: str | Path) -> dict[str, Path]:
    """Write the same CSV/Markdown reports as screen_eval_outliers.py."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    name_map = {
        "runs_flagged": "all_test_results_screened.csv",
        "suites_flagged": "all_suites_screened.csv",
        "result_outliers": "test_result_outliers.csv",
        "suite_outliers": "suite_outliers.csv",
        "removal_candidates": "removal_candidates.csv",
    }
    for key, fname in name_map.items():
        p = outdir / fname
        screen.get(key, pd.DataFrame()).to_csv(p, index=False)
        paths[fname] = p
    lines = ["# Evaluation outlier screening summary\n"]
    lines.append(f"- Test result rows screened: **{len(screen['runs_flagged'])}**")
    lines.append(f"- Flagged test result rows: **{len(screen['result_outliers'])}**")
    lines.append(f"- Flagged suites: **{len(screen['suite_outliers'])}**")
    cand = screen.get("removal_candidates", pd.DataFrame())
    if not cand.empty:
        lines.append("\n## Top removal/review candidates\n")
        for _, r in cand.head(30).iterrows():
            lines.append(f"- `{r.get('scope')}` `{r.get('suggested_review_target')}` suite=`{r.get('suite_run_id')}` test=`{r.get('test_id')}` reasons=`{r.get('reasons')}`")
    lines.append("\n## Notes\n")
    lines.append("- Nothing is deleted automatically; this report only lists candidates for manual review/quarantine.")
    lines.append("- Robust flags are computed globally and within each `test_id`, so intrinsically slow tests are not penalized.")
    p = outdir / "outlier_summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    paths["outlier_summary.md"] = p
    return paths

# =============================================================================
# =============================================================================
# UNIFIED PLOTTING FRAMEWORK
#
# Shared categorical palette, ONE publication style, confidence-interval
# helpers, and a generic grouped-bar-with-error-bars primitive. This is the
# single place that defines "what an agent_eval figure looks like" so that
# agent_eval_master_analysis.ipynb, framework_comparison.ipynb, and
# optimal_config_comparisons.ipynb render with the same colors, fonts, and
# uncertainty conventions instead of three different bespoke styles.
# =============================================================================
# =============================================================================

# -----------------------------------------------------------------------------
# Shared blue palette and black-and-white-safe texture system
# -----------------------------------------------------------------------------

#: One sequential colormap for quantitative encodings throughout the project.
#: Neutral black/gray is retained for text, axes, error bars, and reference lines.
BLUE_CMAP: str = "Blues"
BLUE_CMAP_R: str = "Blues_r"
BLUE_EDGE_COLOR: str = "#17324D"
BLUE_GRID_COLOR: str = "#B8C4D0"

#: Subtle, print-safe hatches. Every bar category receives a texture so the
#: figures remain legible when printed in grayscale. Patterns are deliberately
#: sparse: visible at journal scale without looking decorative or busy.
HATCH_CYCLE: list[str] = ["///", r"\\", "xx", "..", "++", "oo", "--", "**"]

#: Canonical ordering/labels/colors for the agent-architecture sweep. Colors
#: are sampled from light, medium, and dark portions of Matplotlib's Blues map.
FRAMEWORK_ORDER: list[str] = ["one_agent", "two_agent", "three_agent"]

FRAMEWORK_LABELS: dict[str, str] = {
    "one_agent": "1-Agent",
    "two_agent": "2-Agent (Sup+Worker)",
    "three_agent": "3-Agent (Sup+2 Workers)",
}

FRAMEWORK_COLORS: dict[str, str] = {
    "one_agent": "#9ECAE1",
    "two_agent": "#4292C6",
    "three_agent": "#08519C",
}

FRAMEWORK_HATCHES: dict[str, str] = {
    "one_agent": "///",
    "two_agent": r"\\",
    "three_agent": "xx",
}

#: Denser scatter-marker textures than the bar-chart hatches. Matplotlib
#: PathCollection objects support hatching, which lets framework identity stay
#: visible even when all architectures use shades from the Blues map.
FRAMEWORK_MARKER_HATCHES: dict[str, str] = {
    "one_agent": "///",
    "two_agent": "...",
    "three_agent": "xx",
}

#: Deliberate exception to the project-wide Blues palette for RAG association
#: plots: direction of association uses a conventional green/gray/red scheme.
RAG_EFFECT_COLORS: dict[str, str] = {
    "helpful": "#2CA25F",
    "neutral": "#9E9E9E",
    "toxic": "#D73027",
}


FRAMEWORK_LINESTYLES: dict[str, Any] = {
    "one_agent": (0, (1.5, 1.5)),
    "two_agent": "--",
    "three_agent": "-",
}

#: Pass/fail/error remain within the same blue family. Hatches and marker
#: shapes carry the categorical distinction in black-and-white reproduction.
STATUS_COLORS: dict[Any, str] = {
    "passed": "#08519C", "pass": "#08519C", "Passed": "#08519C", True: "#08519C",
    "failed": "#6BAED6", "fail": "#6BAED6", "Failed": "#6BAED6", False: "#6BAED6",
    "error": "#C6DBEF", "Error": "#C6DBEF",
}

STATUS_HATCHES: dict[Any, str] = {
    "passed": "///", "pass": "///", "Passed": "///", True: "///",
    "failed": "xx", "fail": "xx", "Failed": "xx", False: "xx",
    "error": "..", "Error": "..",
}

#: Marker shapes cycled deterministically for categorical scatter series.
MARKER_CYCLE: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]


def get_blue_palette(
    n: int,
    low: float = 0.34,
    high: float = 0.90,
    reverse: bool = False,
) -> list[Any]:
    """Return well-separated samples from Matplotlib's ``Blues`` colormap."""
    import matplotlib as mpl

    if n <= 0:
        return []
    positions = np.linspace(low, high, n)
    if reverse:
        positions = positions[::-1]
    cmap = mpl.colormaps[BLUE_CMAP]
    return [cmap(float(v)) for v in positions]


def get_categorical_colors(
    categories: Iterable[Any],
    palette: str = BLUE_CMAP,
    existing: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministically map category names to colors.

    ``Blues`` is treated specially: colors are sampled from the useful middle
    and dark range rather than the nearly-white end of the colormap. This
    preserves contrast on white backgrounds and in reduced-size figures.
    """
    import seaborn as sns

    cats = sorted({str(c) for c in categories if c is not None and not (isinstance(c, float) and math.isnan(c))})
    existing = {str(k): v for k, v in (existing or {}).items()}
    out: dict[str, Any] = {}
    remaining = []
    for c in cats:
        if c in existing:
            out[c] = existing[c]
        else:
            remaining.append(c)
    if remaining:
        if str(palette).lower() in {"blues", "blues_r"}:
            pal = get_blue_palette(len(remaining), reverse=str(palette).lower().endswith("_r"))
        else:
            pal = sns.color_palette(palette, len(remaining))
        for c, col in zip(remaining, pal):
            out[c] = col
    return out


def get_hatch_map(
    categories: Iterable[Any],
    existing: Mapping[Any, str] | None = None,
    hatches: Sequence[str] = HATCH_CYCLE,
) -> dict[str, str]:
    """Deterministically map category names to subtle hatch patterns."""
    cats = sorted({str(c) for c in categories if c is not None and not (isinstance(c, float) and math.isnan(c))})
    existing = {str(k): v for k, v in (existing or {}).items()}
    out: dict[str, str] = {}
    remaining = []
    for c in cats:
        if c in existing:
            out[c] = existing[c]
        else:
            remaining.append(c)
    for i, c in enumerate(remaining):
        out[c] = hatches[i % len(hatches)]
    return out


def apply_bar_hatches(
    patches: Iterable[Any],
    categories: Iterable[Any],
    hatch_map: Mapping[Any, str] | None = None,
    edgecolor: str = BLUE_EDGE_COLOR,
    linewidth: float = 0.70,
) -> dict[str, str]:
    """Apply category hatches and a crisp dark-blue outline to bar patches."""
    cats = [str(c) for c in categories]
    hmap = {str(k): v for k, v in (hatch_map or get_hatch_map(cats)).items()}
    for patch, category in zip(patches, cats):
        patch.set_hatch(hmap.get(category, ""))
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(linewidth)
    return hmap


def make_legend_patches(
    categories: Iterable[Any],
    labels: Mapping[Any, str] | None = None,
    color_map: Mapping[Any, Any] | None = None,
    hatch_map: Mapping[Any, str] | None = None,
):
    """Create legend patches matching the project's blue + hatch bar style."""
    import matplotlib.patches as mpatches

    cats = [str(c) for c in categories]
    cmap = {str(k): v for k, v in (color_map or get_categorical_colors(cats)).items()}
    hmap = {str(k): v for k, v in (hatch_map or get_hatch_map(cats)).items()}
    labels = {str(k): str(v) for k, v in (labels or {}).items()}
    return [
        mpatches.Patch(
            facecolor=cmap.get(c, "#9ECAE1"), edgecolor=BLUE_EDGE_COLOR,
            linewidth=0.70, hatch=hmap.get(c, ""), label=labels.get(c, c),
        )
        for c in cats
    ]


def get_marker_map(categories: Iterable[Any], markers: Sequence[str] = MARKER_CYCLE) -> dict[str, str]:
    """Deterministically map category names to marker shapes (sorted, cycled)."""
    cats = sorted({str(c) for c in categories if c is not None})
    return {c: markers[i % len(markers)] for i, c in enumerate(cats)}


def set_publication_style(extra_rc: Mapping[str, Any] | None = None, seaborn_theme: str | None = "whitegrid", font_scale: float = 1.0) -> None:
    """
    Apply ONE consistent, journal-friendly Matplotlib/Seaborn style for every
    figure in every notebook that imports this module.

    This intentionally folds in the seaborn theme call that used to be
    duplicated (with slightly different settings) in framework_comparison.ipynb,
    so every notebook starts from the same base style. Pass ``extra_rc`` for
    figure-specific tweaks (e.g. a bigger base font for a single full-width
    figure) without forking the underlying palette/spines/grid conventions.
    """
    import matplotlib as mpl
    from cycler import cycler

    blue_cycle = get_blue_palette(8)
    if seaborn_theme:
        try:
            import seaborn as sns
            sns.set_theme(style=seaborn_theme, font_scale=font_scale, palette=blue_cycle)
        except Exception:
            pass
    mpl.rcParams.update(PUBLICATION_RC)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=blue_cycle)
    if extra_rc:
        mpl.rcParams.update(dict(extra_rc))


# -----------------------------------------------------------------------------
# Confidence-interval helpers
#
# Three regimes, chosen so every bar/point in every notebook can show how
# much it would plausibly move if a different sample of runs had been drawn:
#   - proportions (pass/fail rates)   -> Wilson interval (already used by
#                                        plot_test_pass_rates / summarize_by_test)
#   - means of a continuous metric    -> normal-approximation CI on the mean
#                                        (sample SEM x t-critical value)
#   - sums / composite statistics     -> bootstrap (no clean closed form)
# -----------------------------------------------------------------------------

def mean_ci(values: Iterable[Any], ci: float = 0.95) -> dict[str, float]:
    """Normal-approximation confidence interval for a mean (t critical value, sample SEM)."""
    from scipy import stats as _stats

    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(x.mean())
    if n < 2:
        return {"mean": mean, "se": np.nan, "ci_low": mean, "ci_high": mean, "n": n}
    se = float(x.std(ddof=1) / np.sqrt(n))
    tcrit = float(_stats.t.ppf(0.5 + ci / 2, df=n - 1))
    return {"mean": mean, "se": se, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se, "n": n}


def proportion_ci(k: float, n: float, ci: float = 0.95) -> dict[str, float]:
    """Wilson interval for a binomial proportion. Thin scalar-friendly wrapper around `wilson_interval`."""
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(ci, 2), 1.96)
    lo, hi = wilson_interval(np.array([k], dtype=float), np.array([n], dtype=float), z=z)
    p = (k / n) if n else np.nan
    return {"mean": float(p) if n else np.nan, "se": np.nan, "ci_low": float(lo[0]), "ci_high": float(hi[0]), "n": int(n)}


def bootstrap_ci(values: Iterable[Any], stat=np.mean, n_boot: int = 2000, ci: float = 0.95, random_state: int = 0) -> dict[str, float]:
    """Percentile bootstrap CI for an arbitrary statistic (sum, mean, count, ...)."""
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    point = float(stat(x))
    if n == 1:
        return {"mean": point, "se": np.nan, "ci_low": point, "ci_high": point, "n": n}
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat(x[idx], axis=1) if stat in (np.mean, np.sum, np.median, np.std) else np.array([stat(x[i]) for i in idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return {"mean": point, "se": float(np.std(boots, ddof=1)), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def summarize_with_ci(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    value_col: str,
    kind: str = "auto",
    ci: float = 0.95,
    n_boot: int = 2000,
    stat=np.mean,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    One row per group with a point estimate and a confidence interval for ``value_col``.

    This is the single statistics primitive behind every error bar in
    framework_comparison.ipynb and optimal_config_comparisons.ipynb (and is
    available to agent_eval_master_analysis.ipynb too) so "how much uncertainty
    to show" is answered the same way everywhere instead of being re-derived
    ad hoc per chart.

    kind:
      "auto"      - Wilson CI if every value is 0/1/bool (a rate), else a
                    normal-approximation CI on the mean.
      "wilson"    - force proportion/Wilson CI (value_col must be 0/1 or bool).
      "mean"      - normal-approximation CI on the mean.
      "sum"       - bootstrap CI on the group sum.
      "bootstrap" - bootstrap CI on `stat` (default mean).

    Returns columns: <group_cols...>, n, estimate, se, ci_low, ci_high, ci_kind.
    """
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    work = df.copy()
    numeric_vals = pd.to_numeric(work[value_col], errors="coerce") if kind != "wilson" else work[value_col]

    non_null = numeric_vals.dropna()
    looks_like_rate = len(non_null) > 0 and set(pd.unique(non_null)).issubset({0, 1, True, False})
    resolved_kind = kind
    if kind == "auto":
        resolved_kind = "wilson" if looks_like_rate else "mean"

    work = work.assign(__value__=numeric_vals if resolved_kind != "wilson" else work[value_col].astype(float))

    rows = []
    for key, sub in work.groupby(group_cols, dropna=False):
        vals = sub["__value__"].dropna()
        n = len(vals)
        if resolved_kind == "wilson":
            res = proportion_ci(float(vals.sum()), n, ci=ci)
        elif resolved_kind == "sum":
            res = bootstrap_ci(vals, stat=np.sum, n_boot=n_boot, ci=ci, random_state=random_state)
        elif resolved_kind == "bootstrap":
            res = bootstrap_ci(vals, stat=stat, n_boot=n_boot, ci=ci, random_state=random_state)
        else:
            res = mean_ci(vals, ci=ci)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row.update(n=n, estimate=res["mean"], se=res.get("se", np.nan), ci_low=res["ci_low"], ci_high=res["ci_high"], ci_kind=resolved_kind)
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


# -----------------------------------------------------------------------------
# Generic grouped-bar-with-error-bars primitive
# -----------------------------------------------------------------------------

def _ci_to_yerr(estimate: Sequence[float], ci_low: Sequence[float], ci_high: Sequence[float]) -> np.ndarray:
    est = np.asarray(estimate, dtype=float)
    lo = np.asarray(ci_low, dtype=float)
    hi = np.asarray(ci_high, dtype=float)
    lower = np.nan_to_num(est - lo, nan=0.0)
    upper = np.nan_to_num(hi - est, nan=0.0)
    return np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)])


def plot_grouped_bars(
    ax,
    summary: pd.DataFrame,
    category_col: str,
    estimate_col: str = "estimate",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    group_col: str | None = None,
    category_order: Sequence[Any] | None = None,
    group_order: Sequence[Any] | None = None,
    color_map: Mapping[str, Any] | None = None,
    hatch_map: Mapping[str, str] | None = None,
    palette: str = BLUE_CMAP,
    use_hatches: bool = True,
    bar_label_fmt: str | Any | None = "{:.2f}",
    bar_label_fontsize: float = 7.5,
    bar_width: float = 0.72,
    edgecolor: str = BLUE_EDGE_COLOR,
    hatch_linewidth: float = 0.70,
    capsize: float = 3.5,
    err_color: str = "black",
    legend_title: str | None = None,
    show_legend: bool = True,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    rotate_xticks: float = 0,
    ylim_padding: float | None = None,
) -> dict[str, Any]:
    """
    Draw a (optionally grouped) bar chart with 95%-CI error bars from a tidy
    summary table such as the output of `summarize_with_ci`.

    This is the ONE bar-drawing routine shared by every notebook: bar width,
    error-bar style, color palette, value labels, and grid all look the same
    wherever it is called, and every call necessarily plots an interval
    because `ci_low_col`/`ci_high_col` are required inputs.

    Returns {'x': x positions of each category, 'categories': ..., 'color_map': ...}
    for callers that want to add extra annotations (hlines, brackets, etc.).
    """
    if category_order is not None:
        categories = list(category_order)
    elif hasattr(summary[category_col], "cat"):
        categories = list(summary[category_col].cat.categories)
    else:
        categories = sorted(summary[category_col].dropna().unique().tolist(), key=str)
    x = np.arange(len(categories))
    fmt = bar_label_fmt if callable(bar_label_fmt) else (lambda v, _f=bar_label_fmt: _f.format(v))

    all_hi = pd.to_numeric(summary[ci_high_col], errors="coerce")
    all_est = pd.to_numeric(summary[estimate_col], errors="coerce")
    _label_pad = 0.018 * float(np.nanmax(np.where(all_hi.notna(), all_hi, all_est))) if len(summary) else 0.0

    def _label_bars(xpos, est, hi=None):
        if bar_label_fmt is None:
            return
        hi = hi if hi is not None else est
        for xi, v, h in zip(xpos, est, hi):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            top = h if (h is not None and not (isinstance(h, float) and np.isnan(h))) else v
            ax.text(xi, top + _label_pad, fmt(v), ha="center", va="bottom", fontsize=bar_label_fontsize, zorder=5)

    if group_col and group_col in summary.columns:
        groups = list(group_order) if group_order is not None else sorted(summary[group_col].dropna().unique().tolist(), key=str)
        n_groups = max(len(groups), 1)
        sub_w = bar_width / n_groups
        cmap = {str(k): v for k, v in (dict(color_map) if color_map else get_categorical_colors(groups, palette=palette)).items()}
        hmap = {str(k): v for k, v in (dict(hatch_map) if hatch_map else get_hatch_map(groups)).items()}
        for gi, g in enumerate(groups):
            sub = summary[summary[group_col] == g].set_index(category_col)
            est = [sub.loc[c, estimate_col] if c in sub.index else np.nan for c in categories]
            lo = [sub.loc[c, ci_low_col] if c in sub.index else np.nan for c in categories]
            hi = [sub.loc[c, ci_high_col] if c in sub.index else np.nan for c in categories]
            xpos = x + (gi - (n_groups - 1) / 2) * sub_w
            bars = ax.bar(
                xpos, est, sub_w * 0.88, color=cmap.get(str(g), "#9ECAE1"), label=str(g),
                edgecolor=edgecolor, linewidth=hatch_linewidth,
                hatch=(hmap.get(str(g), "") if use_hatches else ""), zorder=3,
            )
            ax.errorbar(xpos, est, yerr=_ci_to_yerr(est, lo, hi), fmt="none", color=err_color,
                        capsize=capsize, linewidth=1.1, zorder=4)
            _label_bars(xpos, est, hi)
        if show_legend:
            ax.legend(title=legend_title, fontsize=8, frameon=False)
    else:
        sub = summary.set_index(category_col)
        est = [sub.loc[c, estimate_col] if c in sub.index else np.nan for c in categories]
        lo = [sub.loc[c, ci_low_col] if c in sub.index else np.nan for c in categories]
        hi = [sub.loc[c, ci_high_col] if c in sub.index else np.nan for c in categories]
        cmap = {str(k): v for k, v in (dict(color_map) if color_map else get_categorical_colors(categories, palette=palette)).items()}
        hmap = {str(k): v for k, v in (dict(hatch_map) if hatch_map else get_hatch_map(categories)).items()}
        colors = [cmap.get(str(c), "#9ECAE1") for c in categories]
        bars = ax.bar(x, est, bar_width, color=colors, edgecolor=edgecolor, linewidth=hatch_linewidth, zorder=3)
        if use_hatches:
            apply_bar_hatches(bars.patches, categories, hmap, edgecolor=edgecolor, linewidth=hatch_linewidth)
        ax.errorbar(x, est, yerr=_ci_to_yerr(est, lo, hi), fmt="none", color=err_color,
                    capsize=capsize, linewidth=1.1, zorder=4)
        _label_bars(x, est, hi)
        cmap = {str(k): v for k, v in cmap.items()}

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=rotate_xticks, ha=("right" if rotate_xticks else "center"))
    if show_grid:
        ax.yaxis.grid(True, alpha=grid_alpha, zorder=0)
    ax.set_axisbelow(True)
    if ylim_padding is not None:
        all_hi = pd.to_numeric(summary[ci_high_col], errors="coerce").fillna(pd.to_numeric(summary[estimate_col], errors="coerce"))
        ymax = float(all_hi.max()) if len(all_hi) else 1.0
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * (1 + ylim_padding))
    return {"x": x, "categories": categories, "color_map": cmap, "hatch_map": hmap}


def errorbar_scatter(
    ax,
    x: Sequence[float], y: Sequence[float],
    xerr: Sequence[float] | None = None,
    yerr: Sequence[float] | None = None,
    color: Any = "#4C72B0",
    marker: str = "o",
    s: float = 90,
    alpha: float = 0.9,
    label: str | None = None,
    hatch: str | None = None,
    edgecolor: Any = "black",
    linewidth: float = 0.8,
    **kwargs,
):
    """Scatter point(s) with optional error bars and print-safe marker texture."""
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
        elinewidth=1.0, capsize=3, alpha=0.6, zorder=3,
    )
    return ax.scatter(
        x, y, color=color, marker=marker, s=s, alpha=alpha,
        edgecolors=edgecolor, linewidths=linewidth, label=label,
        hatch=hatch, zorder=4, **kwargs,
    )


# -----------------------------------------------------------------------------
# Canonical loaders for multi-source comparison notebooks
#
# Both wrap the same parsers used everywhere else in this module
# (`load_evaluation_data` / `load_suite_tables`) so framework_comparison.ipynb
# and optimal_config_comparisons.ipynb never re-implement suite/jsonl parsing.
# -----------------------------------------------------------------------------

def load_framework_sweep(
    root: str | Path | Sequence[str | Path],
    labels: Mapping[str, str] | None = None,
    expected_frameworks: Sequence[str] = tuple(FRAMEWORK_ORDER),
) -> dict[str, Any]:
    """
    Canonical loader for a "framework sweep": the same benchmark suite run
    under several agent architectures.

    Unlike the old convention (one hardcoded subdirectory per architecture,
    e.g. `root/one_agent/`, `root/two_agent/`, ...), the architecture for
    each individual test result is read directly from that result's own
    `cfg_graph_topology` config field (populated by `flatten_config_snapshot`,
    which backfills `DEFAULT_GRAPH_TOPOLOGY` for older suites that predate the
    field). `root` can therefore be a single directory containing suite files
    from any/all architectures mixed together -- nothing about the folder
    layout is assumed. `root` may also be a single file or a list of
    files/directories, same as `load_evaluation_data`.

    `labels` maps a raw topology value (e.g. "one_agent") to a display label
    (e.g. "1-Agent"); defaults to `FRAMEWORK_LABELS`. Topology values with no
    entry in `labels` are labeled with their raw value, so architectures added
    later (e.g. "four_agent") work without editing this module.

    `expected_frameworks` is only used to populate the `missing` list (frameworks
    you expected to see but that had zero matching rows) -- it does not filter
    or restrict which topologies get loaded.

    Returns {'runs': ..., 'rag_events': ..., 'missing': [expected frameworks with no data]}.
    """
    labels = dict(labels) if labels else dict(FRAMEWORK_LABELS)
    try:
        runs, rag_events = load_evaluation_data(root, prefer_suite_json=True)
    except FileNotFoundError:
        runs, rag_events = pd.DataFrame(), pd.DataFrame()

    if runs.empty:
        return {"runs": runs, "rag_events": rag_events, "missing": list(expected_frameworks)}

    runs = runs.copy()
    # flatten_config_snapshot already backfills cfg_graph_topology with
    # DEFAULT_GRAPH_TOPOLOGY, but guard here too in case `runs` came from a
    # source (e.g. hand-built jsonl) that skipped that step.
    topology = runs["cfg_graph_topology"] if "cfg_graph_topology" in runs.columns else pd.Series([None] * len(runs), index=runs.index)
    runs["framework"] = topology.fillna(DEFAULT_GRAPH_TOPOLOGY)
    runs["framework_label"] = runs["framework"].map(labels).fillna(runs["framework"])

    if rag_events is not None and not rag_events.empty and "run_key" in rag_events.columns and "run_key" in runs.columns:
        rag_events = rag_events.copy()
        fw_lookup = runs.drop_duplicates("run_key").set_index("run_key")[["framework", "framework_label"]]
        rag_events = rag_events.merge(fw_lookup, on="run_key", how="left")

    present = set(runs["framework"].dropna())
    missing = [fw for fw in expected_frameworks if fw not in present]
    if "cfg_llm_model" not in runs.columns:
        runs["cfg_llm_model"] = "unknown"
    return {"runs": runs, "rag_events": rag_events, "missing": missing}


def _resolve_label_source(source: str | Path | Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the runs/rag_events for ONE label's source spec, which may be:
      - a directory containing one or more suite_*.json / regression_data_*.jsonl
        files (one per repeated run) -- the recommended way to point at "all
        the runs of this configuration",
      - a single suite-result .json or regression .jsonl file,
      - or a list mixing either of the above (loaded and concatenated).
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return load_evaluation_data(p, prefer_suite_json=True)
        if p.suffix == ".jsonl":
            return load_regression_jsonl(p), pd.DataFrame()
        return load_suite_tables(p)
    run_frames, event_frames = [], []
    for item in source:
        r, e = _resolve_label_source(item)
        if not r.empty:
            run_frames.append(r)
        if e is not None and not e.empty:
            event_frames.append(e)
    runs = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return runs, events


def load_named_configs(
    sources: Mapping[str, Any] | Sequence[str | Path],
    labels: Sequence[str] | Mapping[str, str] | None = None,
    label_col: str = "config_plot_label",
) -> pd.DataFrame:
    """
    Canonical loader for a handful of named configurations to compare
    side-by-side (e.g. several surrogate-model-recommended "optimal" configs
    plus a baseline), where EACH configuration may have been run **multiple
    times** over the same test suite.

    `sources` is either:
      - a dict ``{label: source}`` -- the recommended form when you have
        repeated runs to average over. Each ``source`` can be a directory
        (every suite/regression file inside is treated as one run of that
        configuration), a single file, or an explicit list of files/directories.
      - a flat list of single files, paired with `labels` (the original
        one-run-per-config form; still supported for backward compatibility).

    Every individual test result from every run is kept as its own row,
    tagged with its label and a `run_id` (the run's `suite_run_id`, falling
    back to its source file), so `summarize_named_configs` can compute
    confidence intervals across repeated runs rather than just across the
    tests within a single run.
    """
    if isinstance(sources, Mapping):
        items = list(sources.items())
    else:
        sources = list(sources)
        if labels is None:
            resolved_labels = [Path(p).stem if isinstance(p, (str, Path)) else f"config_{i}" for i, p in enumerate(sources)]
        elif isinstance(labels, Mapping):
            resolved_labels = [labels.get(str(p), labels.get(Path(p).name, Path(p).stem)) for p in sources]
        else:
            resolved_labels = list(labels)
            if len(resolved_labels) != len(sources):
                raise ValueError("labels must have the same length as sources")
        items = list(zip(resolved_labels, sources))

    frames = []
    for label, source in items:
        runs, _ = _resolve_label_source(source)
        if runs.empty:
            warnings.warn(f"No results parsed for label {label!r} from {source!r}")
            continue
        runs = runs.copy()
        runs[label_col] = label
        runs["run_id"] = runs["suite_run_id"] if "suite_run_id" in runs.columns else runs.get("source_file", label)
        runs["run_id"] = runs["run_id"].fillna(runs.get("source_file", label))
        frames.append(runs)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_named_configs_multi(
    sources_by_group: Mapping[str, Mapping[str, Any] | Sequence[Any]],
    label_col: str = "config_plot_label",
    group_col: str = "test_set",
    labels: Sequence[str] | Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load named configs across multiple *parallel* test-set groups -- e.g. the
    standard 53-test optimization suite and a held-out validation suite run
    against the same named configurations -- tagging every row with which
    group it came from.

    `sources_by_group` is ``{group_value: sources}`` where each `sources`
    value is exactly what `load_named_configs` accepts on its own (typically
    a ``{config_label: source}`` dict, using the *same* config labels in
    every group so results line up on the same x-axis category later).

    Each group is loaded independently via `load_named_configs` -- so a
    config that's missing from one group (e.g. a validation run that hasn't
    finished yet) simply produces no rows for that (label, group) pair
    rather than raising -- then tagged with `group_col` and concatenated.
    The result is the direct input to `summarize_named_configs(..., group_col=...)`.
    """
    frames = []
    for group_value, sources in sources_by_group.items():
        runs = load_named_configs(sources, labels=labels, label_col=label_col)
        if runs.empty:
            warnings.warn(f"No results parsed for test-set group {group_value!r}")
            continue
        runs = runs.copy()
        runs[group_col] = group_value
        frames.append(runs)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_named_configs(
    runs: pd.DataFrame,
    label_col: str = "config_plot_label",
    group_col: str | None = None,
    run_id_col: str = "run_id",
    common_tests_only: bool = True,
    ci: float = 0.95,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """
    Per-configuration summary (pass rate, mean/total tokens, mean duration,
    total cost) with a 95% confidence interval on every metric, **averaged
    over repeated runs when more than one run is present for a configuration**.

    Two CI regimes, chosen automatically per configuration:
      - **n_runs >= 2 ("across_run")**: compute each metric once per run, then
        take the mean +/- a t-distribution CI *across those per-run values*.
        This is the statistically appropriate way to combine repeated
        full-suite runs -- the tests inside a single run share that run's
        conditions, so measuring spread only *within* one run would
        pseudo-replicate and understate the true run-to-run uncertainty.
      - **n_runs == 1 ("within_run")**: there's no run-to-run spread to
        measure, so this falls back to a CI across the individual test
        results within that single run (Wilson for the pass rate; a
        normal-approximation or bootstrap interval for the continuous
        metrics) -- the same behavior as when this function was first
        written for one-run-per-config comparisons.

    The configuration-level `ci_basis` column records which regime applied.

    `group_col`, if given (e.g. `"test_set"` distinguishing a standard suite
    from a held-out validation suite -- see `load_named_configs_multi`),
    adds a second grouping dimension: one summary row is produced per
    (label, group) pair instead of per label alone, and `common_tests_only`
    restricts to common test_ids *within each group value independently*
    (so a validation suite's disjoint test IDs never interact with the
    standard suite's "common tests" computation, and vice versa).
    """
    df = runs.copy()
    if run_id_col not in df.columns:
        df[run_id_col] = df.get("suite_run_id", df.get("source_file"))
    df[run_id_col] = df[run_id_col].fillna(df.get("source_file", "run_0"))

    gcols = [label_col] + ([group_col] if group_col else [])
    label_order = list(dict.fromkeys(df[label_col]))  # first-seen order, preserved for plotting

    if common_tests_only and "test_id" in df.columns:
        if group_col and group_col in df.columns:
            # Compute "common to every config" separately per group_col value,
            # since e.g. standard-suite and validation-suite test_ids are
            # disjoint by construction and should never be intersected together.
            keep = pd.Series(False, index=df.index)
            for _, idx in df.groupby(group_col).groups.items():
                block = df.loc[idx]
                per_label = block.groupby(label_col)["test_id"].apply(lambda s: set(s.dropna()))
                common = set.intersection(*per_label.tolist()) if len(per_label) else set()
                keep.loc[idx] = block["test_id"].isin(common).values if common else True
            df = df[keep].copy()
        else:
            per_label_tests = df.groupby(label_col)["test_id"].apply(lambda s: set(s.dropna()))
            common = set.intersection(*per_label_tests.tolist()) if len(per_label_tests) else set()
            if common:
                df = df[df["test_id"].isin(common)].copy()

    per_run = df.groupby(gcols + [run_id_col]).agg(
        n_tests=("test_id", "count"),
        pass_rate=("passed_int", "mean"),
        mean_duration_s=("duration_s", "mean"),
        mean_tokens_per_test=("total_tokens", "mean"),
        total_tokens=("total_tokens", "sum"),
        total_estimated_cost=("estimated_cost", "sum") if "estimated_cost" in df.columns else ("total_tokens", "size"),
    ).reset_index()
    if "estimated_cost" not in df.columns:
        per_run = per_run.drop(columns=["total_estimated_cost"])

    # out_name -> (raw column used for the n_runs==1 within-run fallback, its CI kind)
    metric_fallback = {
        "pass_rate": ("passed_int", "wilson"),
        "mean_duration_s": ("duration_s", "mean"),
        "mean_tokens_per_test": ("total_tokens", "mean"),
        "total_tokens": ("total_tokens", "sum"),
        "total_estimated_cost": ("estimated_cost", "sum"),
    }
    metric_names = [m for m in metric_fallback if m in per_run.columns]

    rows = []
    for key, sub in per_run.groupby(gcols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        n_runs = sub[run_id_col].nunique()
        row: dict[str, Any] = dict(zip(gcols, key_tuple))
        row.update(n_runs=n_runs, n_tests=int(sub["n_tests"].sum()))
        if n_runs >= 2:
            row["ci_basis"] = "across_run"
            for m in metric_names:
                res = mean_ci(sub[m], ci=ci)
                row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = res["mean"], res["ci_low"], res["ci_high"]
        else:
            row["ci_basis"] = "within_run"
            mask = pd.Series(True, index=df.index)
            for c, v in zip(gcols, key_tuple):
                mask &= (df[c] == v)
            raw = df[mask]
            for m in metric_names:
                raw_col, kind = metric_fallback[m]
                res = summarize_with_ci(raw, gcols, raw_col, kind=kind, ci=ci, n_boot=n_boot).iloc[0]
                row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = res["estimate"], res["ci_low"], res["ci_high"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out["pass_rate_percent"] = out["pass_rate"] * 100
    out["pass_rate_percent_ci_low"] = out["pass_rate_ci_low"] * 100
    out["pass_rate_percent_ci_high"] = out["pass_rate_ci_high"] * 100
    out[label_col] = pd.Categorical(out[label_col], categories=label_order, ordered=True)
    sort_cols = [label_col]
    if group_col:
        group_order = list(dict.fromkeys(df[group_col]))
        out[group_col] = pd.Categorical(out[group_col], categories=group_order, ordered=True)
        sort_cols.append(group_col)
    return out.sort_values(sort_cols).reset_index(drop=True)


def plot_named_config_summary(
    summary: pd.DataFrame,
    label_col: str = "config_plot_label",
    group_col: str | None = None,
    group_order: Sequence[Any] | None = None,
    title: str = "Benchmark Results Across Configurations",
    color_map: Mapping[str, Any] | None = None,
    legend_title: str | None = None,
    figsize: tuple[float, float] = (13.5, 8.5),
    bar_label_fontsize: int = 9,
    rotate_xticks: float = 0,
    show_grid: bool = True,
    ylim_padding: float = 0.18,
):
    """
    The 2x2 "pass rate / mean tokens / total tokens / mean duration" summary
    figure from optimal_config_comparisons.ipynb, redrawn with
    `plot_grouped_bars` so every panel carries a confidence interval.

    Pass `group_col` (e.g. `"test_set"`, matching what was passed to
    `summarize_named_configs`) to draw each configuration as side-by-side
    sub-bars -- one per group value (such as "Standard" vs "Validation") --
    each with its own CI and a shared color per group value across every
    configuration. `color_map` is then keyed by group value rather than by
    configuration label.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if group_col and group_col in summary.columns:
        groups = list(group_order) if group_order is not None else sorted(summary[group_col].dropna().unique().tolist(), key=str)
        cmap = dict(color_map) if color_map else get_categorical_colors(groups, palette=BLUE_CMAP)
    else:
        cmap = dict(color_map) if color_map else get_categorical_colors(summary[label_col].astype(str), palette=BLUE_CMAP)
    panels = [
        (axes[0, 0], "pass_rate_percent", "Pass rate", "Pass rate (%)", "{:.1f}%"),
        (axes[0, 1], "mean_tokens_per_test", "Mean tokens per test", "Tokens / test", "{:,.0f}"),
        (axes[1, 0], "total_tokens", "Total tokens", "Total tokens", "{:,.0f}"),
        (axes[1, 1], "mean_duration_s", "Average duration per test", "Seconds / test", "{:.1f}s"),
    ]
    for ax, col, ttl, ylabel, fmt in panels:
        if col not in summary.columns:
            ax.axis("off")
            continue
        s = summary.rename(columns={col: "estimate", f"{col}_ci_low": "ci_low", f"{col}_ci_high": "ci_high"})
        plot_grouped_bars(
            ax, s, category_col=label_col, group_col=group_col, group_order=group_order,
            color_map=cmap, bar_label_fmt=fmt, bar_label_fontsize=bar_label_fontsize,
            legend_title=legend_title, show_legend=bool(group_col), rotate_xticks=rotate_xticks,
            show_grid=show_grid, ylim_padding=ylim_padding,
        )
        ax.set_title(ttl, pad=10)
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontweight="bold")
    return fig, axes