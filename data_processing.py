# data_processing.py
# Read questionnaire JSONs, pair parent/teen answers, and expose:
# - pairs_df: per-question paired records (incl. free text)
# - ts_df: per-dimension monthly time series using analysis_ready_data (fallback: computed mean)
from __future__ import annotations

import os, re, json, glob, math
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# -------------------- Tunables --------------------
DELTA_THRESHOLD = 1.0           # month-to-month gap change threshold
SLOPE_THRESHOLD = 0.25          # slope threshold for a trend
FUZZY_JACCARD_THRESHOLD = 0.6   # fuzzy pair threshold (same dimension)
EXTS = (".json",)

# -------------------- Small utils --------------------
def _parse_dt(s: Any) -> datetime:
    """Parse various timestamp formats; return datetime.min on failure."""
    if isinstance(s, datetime):
        return s
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return datetime.min

def _month_str(dt: datetime) -> str:
    return "unknown" if dt == datetime.min else f"{dt.year:04d}-{dt.month:02d}"

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _canon_qid(qid: Optional[str]) -> Optional[str]:
    """Normalize question ids, remove leading role hints."""
    if not qid:
        return None
    s = str(qid).strip().lower()
    s = re.sub(r"^(p|t|teen|parent|c)_+", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or None

def _letter_index(ch: Any) -> Optional[int]:
    """Parse A/B/C/D..., or variants like 'B)', 'b.' into 0-based index."""
    if not ch:
        return None
    s = str(ch).strip()
    m = re.match(r"^\s*([A-Za-z])(?:[\)\.\s].*)?$", s)
    if not m:
        return None
    idx = ord(m.group(1).upper()) - ord("A")
    return idx if 0 <= idx < 26 else None

def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _fmt_num(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    try:
        xf = float(x)
        return f"{xf:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def _jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity for fuzzy question text match."""
    ta = set(_norm_text(a).split())
    tb = set(_norm_text(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

# -------------------- Data model --------------------
@dataclass
class PairRecord:
    """One paired record per question (latest per (qid_canon, dimension) downstream when needed)."""
    child_name: str
    month: str
    ts: datetime
    dimension: str
    # ids/texts
    qid_parent: Optional[str]
    qid_teen: Optional[str]
    qid_canon: Optional[str]
    qtext_parent: Optional[str]
    qtext_teen: Optional[str]
    # types
    type_parent: Optional[str]
    type_teen: Optional[str]
    # numeric or text
    value_parent: Optional[float]
    value_teen: Optional[float]
    label_parent: Optional[str]
    label_teen: Optional[str]
    free_parent: Optional[str]
    free_teen: Optional[str]

# -------------------- Main store --------------------
class MHDataStore:
    """Load JSONs, pair answers, and provide retrieval/summaries for the app."""
    def __init__(self):
        self.pairs_df: Optional[pd.DataFrame] = None
        self.ts_df: Optional[pd.DataFrame] = None
        self._children: List[str] = []

    # ---------- read analysis_ready_data helpers ----------
    def _extract_precomputed_ts(self, data: Dict[str, Any], child_name: str, month: str) -> List[Dict[str, Any]]:
        """Read parent/teen per-dimension averages from analysis_ready_data.dimension_scores."""
        out: List[Dict[str, Any]] = []
        dscores = (data.get("analysis_ready_data") or {}).get("dimension_scores") or {}
        if not dscores:
            return out
        parent_map = (dscores.get("parent") or {})
        teen_map   = (dscores.get("teenager") or dscores.get("teen") or {})
        dims = set(parent_map.keys()) | set(teen_map.keys())
        for dim in dims:
            pa = parent_map.get(dim) or {}
            ta = teen_map.get(dim) or {}
            out.append({
                "child_name": child_name,
                "month": month,
                "dimension": dim,
                "parent_avg": (float(pa["average_score"]) if pa.get("average_score") is not None else None),
                "teen_avg":   (float(ta["average_score"]) if ta.get("average_score") is not None else None),
            })
        return out

    def _extract_free_text_summary_df(self, data: Dict[str, Any], child_name: str, month: str) -> pd.DataFrame:
        """Read analysis_ready_data.free_text_summary and return a flat dataframe for merging."""
        rows: List[Dict[str, Any]] = []
        fts = (data.get("analysis_ready_data") or {}).get("free_text_summary") or {}
        for side_key, arr in (("parent", fts.get("parent") or []),
                              ("teenager", fts.get("teenager") or [])):
            for item in arr or []:
                qid = item.get("question_id")
                txt = (item.get("free_text_response") or "").strip()
                if not qid or not txt:
                    continue
                rows.append({
                    "child_name": child_name,
                    "month": month,
                    "side": side_key,
                    "qid_canon": _canon_qid(qid),
                    "free_text_summary": txt
                })
        return pd.DataFrame(rows)

    # ---------- public API ----------
    def load_root(self, root_dir: str) -> "MHDataStore":
        """Load all children from root_dir (expects data/<child_slug>/*.json)."""
        all_pairs: List[PairRecord] = []
        pre_ts_rows: List[Dict[str, Any]] = []
        free_summ_dfs: List[pd.DataFrame] = []

        for child_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
            if not os.path.isdir(child_dir):
                continue
            child_name_from_dir = os.path.basename(child_dir).replace("_", " ").title()

            for fp in sorted(glob.glob(os.path.join(child_dir, "*"))):
                if not fp.lower().endswith(EXTS):
                    continue
                data = self._safe_load_json(fp)
                if not data:
                    continue

                # timestamps / identity
                assess_dt = _parse_dt(
                    data.get("assessment_session", {}).get("assessment_date")
                    or data.get("metadata", {}).get("assessment_date")
                )
                month = _month_str(assess_dt)
                child_name = (
                    data.get("participant_info", {}).get("child", {}).get("name")
                    or child_name_from_dir
                )

                # 1) read analysis_ready_data (preferred source for time series + free-text summary)
                pre_ts_rows.extend(self._extract_precomputed_ts(data, child_name, month))
                df_ft = self._extract_free_text_summary_df(data, child_name, month)
                if not df_ft.empty:
                    free_summ_dfs.append(df_ft)

                # 2) flatten both sides (raw responses) -> for pairing and evidence
                opt_parent = self._build_option_map(
                    data.get("questionnaire_data", {}).get("parent_questionnaire", [])
                )
                opt_teen = self._build_option_map(
                    data.get("questionnaire_data", {}).get("teenager_questionnaire", [])
                )

                parent_rows = self._flatten_side(
                    side="parent",
                    responses=(data.get("responses") or {}).get("parent", []) or [],
                    opt_map=opt_parent
                )
                teen_rows = self._flatten_side(
                    side="teenager",
                    responses=(data.get("responses") or {}).get("teenager", []) or [],
                    opt_map=opt_teen
                )

                # 3) inject free-text from analysis_ready_data into per-item rows
                parent_rows, teen_rows = self._inject_free_text_from_ard(data, parent_rows, teen_rows)

                # 4) pair two sides (qid -> text -> fuzzy within dimension)
                for p in self._pair_two_sides(parent_rows, teen_rows):
                    all_pairs.append(PairRecord(
                        child_name=child_name,
                        month=month,
                        ts=assess_dt,
                        dimension=p.get("dimension") or "Misc",
                        qid_parent=p.get("qid_parent"),
                        qid_teen=p.get("qid_teen"),
                        qid_canon=p.get("qid_canon"),
                        qtext_parent=p.get("qtext_parent"),
                        qtext_teen=p.get("qtext_teen"),
                        type_parent=p.get("type_parent"),
                        type_teen=p.get("type_teen"),
                        value_parent=p.get("value_parent"),
                        value_teen=p.get("value_teen"),
                        label_parent=p.get("label_parent"),
                        label_teen=p.get("label_teen"),
                        free_parent=p.get("free_parent"),
                        free_teen=p.get("free_teen"),
                    ))

                if child_name not in self._children:
                    self._children.append(child_name)

        # --------- build dataframes ---------
        if all_pairs:
            self.pairs_df = pd.DataFrame([asdict(x) for x in all_pairs])

            # merge analysis_ready free-text summary (as fallback into free_parent/free_teen)
            if free_summ_dfs:
                free_summ = pd.concat(free_summ_dfs, ignore_index=True)
                pivot = (free_summ
                         .pivot_table(index=["child_name","month","qid_canon"],
                                      columns="side", values="free_text_summary",
                                      aggfunc="last")
                         .reset_index())
                pivot.columns.name = None
                pivot = pivot.rename(columns={"parent":"free_parent_summ",
                                              "teenager":"free_teen_summ"})
                key = ["child_name","month","qid_canon"]
                self.pairs_df = self.pairs_df.merge(pivot, on=key, how="left")
                self.pairs_df["free_parent"] = self.pairs_df["free_parent"].combine_first(self.pairs_df["free_parent_summ"])
                self.pairs_df["free_teen"]   = self.pairs_df["free_teen"].combine_first(self.pairs_df["free_teen_summ"])
                self.pairs_df = self.pairs_df.drop(columns=["free_parent_summ","free_teen_summ"])

            # normalized text for quick filtering
            self.pairs_df["qtext_parent_norm"] = self.pairs_df["qtext_parent"].fillna("").map(_norm_text)
            self.pairs_df["qtext_teen_norm"]   = self.pairs_df["qtext_teen"].fillna("").map(_norm_text)

            # time series: prefer precomputed averages; fallback to per-item means
            calc_ts = (self.pairs_df
                       .assign(parent_num=lambda d: pd.to_numeric(d["value_parent"], errors="coerce"),
                               teen_num=lambda d: pd.to_numeric(d["value_teen"], errors="coerce"))
                       .groupby(["child_name", "month", "dimension"], as_index=False)
                       .agg(parent_avg=("parent_num", "mean"),
                            teen_avg=("teen_num", "mean")))

            pre_ts = pd.DataFrame(pre_ts_rows,
                                  columns=["child_name","month","dimension","parent_avg","teen_avg"])
            if pre_ts.empty:
                ts_combined = calc_ts.copy()
            else:
                key = ["child_name","month","dimension"]
                ts_combined = (pre_ts
                               .merge(calc_ts, on=key, how="outer", suffixes=("_pre","_cmp"))
                               .assign(
                                   parent_avg=lambda d: d["parent_avg_pre"].combine_first(d["parent_avg_cmp"]),
                                   teen_avg=lambda d: d["teen_avg_pre"].combine_first(d["teen_avg_cmp"]),
                               )[key + ["parent_avg","teen_avg"]])

            self.ts_df = ts_combined.assign(gap=lambda d: d["parent_avg"] - d["teen_avg"])
        else:
            self.pairs_df = pd.DataFrame(columns=[f.name for f in fields(PairRecord)])
            self.ts_df = pd.DataFrame(columns=["child_name","month","dimension","parent_avg","teen_avg","gap"])

        return self

    # ---------- small public helpers ----------
    def children(self) -> List[str]:
        return sorted(self._children)

    def by_child_pairs(self, child_name: str) -> pd.DataFrame:
        if self.pairs_df is None or self.pairs_df.empty:
            return pd.DataFrame()
        return self.pairs_df[self.pairs_df["child_name"].str.lower() == child_name.strip().lower()].copy()

    def compute_timeseries(self, child_name: str) -> pd.DataFrame:
        if self.ts_df is None or self.ts_df.empty:
            return pd.DataFrame()
        out = self.ts_df[self.ts_df["child_name"].str.lower() == child_name.strip().lower()].copy()
        return out.sort_values(["dimension", "month"])

    # ---------- summaries for the chat context ----------
    def summarize_child(self, child_name: str) -> Dict[str, Any]:
        ts = self.compute_timeseries(child_name)
        if ts.empty:
            return {"child": child_name, "months": [], "latest_snapshot": [], "top_gaps": [], "notable_changes": [], "trends": []}

        months = sorted(ts["month"].unique().tolist())
        last_m = months[-1]

        latest = (ts[ts["month"] == last_m]
                  .sort_values("dimension")
                  .assign(parent=lambda d: d["parent_avg"],
                          teen=lambda d: d["teen_avg"])
                  [["dimension", "parent", "teen", "gap"]])

        latest2 = latest.assign(abs_gap=lambda d: d["gap"].abs())
        top_gaps = latest2.sort_values("abs_gap", ascending=False).drop(columns=["abs_gap"]).head(5)
        top_gaps = top_gaps.to_dict(orient="records")

        # month-to-month notable changes
        notable = []
        for dim, g in ts.groupby("dimension"):
            g = g.sort_values("month").reset_index(drop=True)
            for i in range(1, len(g)):
                prev_gap = g.loc[i-1, "gap"]
                cur_gap = g.loc[i, "gap"]
                if pd.notnull(prev_gap) and pd.notnull(cur_gap):
                    delta = float(cur_gap - prev_gap)
                    if abs(delta) >= DELTA_THRESHOLD:
                        notable.append({
                            "dimension": dim,
                            "month_from": g.loc[i-1, "month"],
                            "month_to": g.loc[i, "month"],
                            "delta_gap": round(delta, 2),
                            "label": "widened" if delta > 0 else "narrowed"
                        })

        # simple linear trend on gap
        trends = []
        for dim, g in ts.groupby("dimension"):
            g2 = g.sort_values("month").reset_index(drop=True)
            xs = list(range(len(g2)))
            ys = [0.0 if pd.isna(v) else float(v) for v in g2["gap"].tolist()]
            slope = self._slope(xs, ys)
            if abs(slope) >= SLOPE_THRESHOLD:
                trends.append({
                    "dimension": dim,
                    "slope": round(float(slope), 3),
                    "direction": "up" if slope > 0 else "down"
                })

        return {
            "child": child_name,
            "months": months,
            "latest_snapshot": latest.to_dict(orient="records"),
            "top_gaps": top_gaps,
            "notable_changes": notable,
            "trends": trends
        }

    # ---------- retrieval for routed evidence ----------
    def retrieve_dual_perspective(
        self,
        child_name: str,
        query: str = "",
        dimension: Optional[str] = None,
        month: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return most recent paired Q/A items (with numeric/text), optionally filtered."""
        df = self.by_child_pairs(child_name)
        if df.empty:
            return []
        g = df.copy()
        if dimension:
            g = g[g["dimension"].str.lower() == dimension.strip().lower()]
        if month:
            g = g[g["month"] == month]
        if query:
            q = _norm_text(query)
            g = g[g["qtext_parent_norm"].str.contains(q, regex=False) |
                  g["qtext_teen_norm"].str.contains(q, regex=False)]

        # keep latest per (qid_canon, dimension); then take newest top_k
        g = g.sort_values("ts").groupby(["qid_canon", "dimension"], dropna=False).tail(1)
        g = g.sort_values("ts", ascending=False).head(top_k)

        out = []
        for _, r in g.iterrows():
            teen_text = r["free_teen"] or r["label_teen"] or _fmt_num(r["value_teen"])
            parent_text = r["free_parent"] or r["label_parent"] or _fmt_num(r["value_parent"])
            qtext = r["qtext_teen"] or r["qtext_parent"]
            out.append({
                "month": r["month"],
                "dimension": r["dimension"],
                "question_text": qtext,
                "teen_answer_text": teen_text,
                "parent_answer_text": parent_text,
                "teen_score": r["value_teen"],
                "parent_score": r["value_parent"],
                "gap": (None if (r["value_parent"] is None or r["value_teen"] is None)
                        else float(r["value_parent"]) - float(r["value_teen"]))
            })
        return out

    def build_chat_context(
        self,
        child_name: str,
        max_items: int = 6,
        only_significant_changes: bool = True,
        max_snapshot_dims: int = 8,
        include_text_evidence: bool = True,
    ) -> str:
        """Build a compact, LLM-friendly summary with optional free-text highlights."""
        summ = self.summarize_child(child_name)
        if not summ.get("months"):
            return f"Data note: No questionnaire records found for {child_name}."

        lines: List[str] = []
        lines.append(f"Child: {child_name}; Months covered: {', '.join(summ['months'])}.")

        # latest snapshot (top |gap|)
        latest_list = summ.get("latest_snapshot", [])
        if latest_list:
            latest_df = pd.DataFrame(latest_list)
            latest_df = (latest_df
                         .assign(abs_gap=lambda d: d["gap"].abs())
                         .sort_values("abs_gap", ascending=False)
                         .drop(columns=["abs_gap"])
                         .head(max_snapshot_dims))
            n_show = min(max_snapshot_dims, len(latest_df))
            lines.append(f"Latest snapshot (top {n_show} by absolute gap; gap = parent - teen):")
            for _, row in latest_df.iterrows():
                lines.append(f"- {row['dimension']}: parent={_fmt_num(row['parent'])}, "
                             f"teen={_fmt_num(row['teen'])}, gap={_fmt_num(row['gap'])}")

        if summ["top_gaps"]:
            lines.append("Largest perception gaps (latest month):")
            for r in summ["top_gaps"][:max_items]:
                lines.append(f"- {r['dimension']}: gap={_fmt_num(r['gap'])}")

        changes = summ["notable_changes"]
        if changes:
            lines.append("Notable month-to-month changes in gap:")
            for c in changes[:max_items]:
                lines.append(f"- {c['dimension']} | {c['month_from']}→{c['month_to']}: "
                             f"Δgap={_fmt_num(c['delta_gap'])} ({c['label']})")

        if summ["trends"]:
            lines.append("Trends across months (gap slope):")
            for t in summ["trends"][:max_items]:
                lines.append(f"- {t['dimension']}: slope={_fmt_num(t['slope'])} ({t['direction']})")

        if include_text_evidence:
            evid = self.gather_free_text_evidence(
                child_name=child_name, max_dims=6, top_k_per_dim=1, max_snippet_len=160
            )
            if evid:
                lines.append("Recent free-text highlights:")
                for e in evid[:max_items]:
                    lines.append(f"- [{e['dimension']}] {e['question_text']} ({e['month']})")
                    if e.get("teen_text"):
                        lines.append(f"  • Teen: \"{e['teen_text']}\"")
                    if e.get("parent_text"):
                        lines.append(f"  • Parent: \"{e['parent_text']}\"")

        lines.append("Guidance: Use dual-perspective answers, highlight gaps with brief explanations, "
                     "and offer actionable parenting suggestions (routines, communication, activities).")
        return "\n".join(lines)

    # ---------- internal parsing/pairing ----------
    def _safe_load_json(self, fp: str) -> Optional[Dict[str, Any]]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _build_option_map(self, qs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """qid -> list of options (each with value/text); used to map letters or reverse-lookup labels."""
        mp: Dict[str, List[Dict[str, Any]]] = {}
        for q in qs or []:
            qid = q.get("id") or q.get("question_id")
            if not qid:
                continue
            mp[str(qid)] = q.get("options") or []
        return mp

    def _flatten_side(self, side: str, responses: List[Dict[str, Any]], opt_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize one side's answers into a consistent structure."""
        out = []
        for r in responses or []:
            qid = r.get("question_id")
            qtext = r.get("question_text")
            dimension = r.get("dimension") or "Misc"
            rtype = r.get("response_type")
            selected_value = r.get("selected_value")
            selected_opt = r.get("selected_option")
            free_text = (r.get("free_text_response") or "").strip() or None

            value = _try_float(selected_value)
            label = None

            if value is None and selected_opt:
                # 1) interpret letter "A/B/C..." by index into options
                idx = _letter_index(selected_opt)
                opts = opt_map.get(str(qid), [])
                if idx is not None and 0 <= idx < len(opts):
                    value = _try_float(opts[idx].get("value"))
                    label = (opts[idx].get("text") or "").strip() or None
                else:
                    # 2) try matching selected_option text to option.text
                    so = str(selected_opt).strip().lower()
                    for opt in opts:
                        if so == (opt.get("text") or "").strip().lower():
                            value = _try_float(opt.get("value"))
                            label = (opt.get("text") or "").strip() or None
                            break

            if label is None and value is not None:
                # reverse lookup label by numeric value
                for opt in opt_map.get(str(qid), []):
                    if _try_float(opt.get("value")) == value:
                        label = (opt.get("text") or "").strip() or None
                        break

            out.append({
                "side": side,
                "qid": str(qid) if qid is not None else None,
                "qid_canon": _canon_qid(qid),
                "qtext": qtext,
                "dimension": dimension,
                "type": rtype,
                "value": value,
                "label": label,
                "free": free_text
            })
        return out

    def _pair_two_sides(self, parent_rows: List[Dict[str, Any]], teen_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pair logic: by qid_canon -> by normalized question text -> fuzzy within same dimension."""
        res: List[Dict[str, Any]] = []

        by_qid_p = {r["qid_canon"]: r for r in parent_rows if r.get("qid_canon")}
        by_qid_t = {r["qid_canon"]: r for r in teen_rows if r.get("qid_canon")}
        used_p, used_t = set(), set()

        # 1) exact by qid_canon
        for qid_c in set(by_qid_p.keys()) & set(by_qid_t.keys()):
            p = by_qid_p[qid_c]; t = by_qid_t[qid_c]
            res.append(self._mk_pair(p, t, qid_c))
            used_p.add(id(p)); used_t.add(id(t))

        # 2) exact by normalized text (prefer same dimension)
        txt_ix_p: Dict[str, List[Dict[str, Any]]] = {}
        txt_ix_t: Dict[str, List[Dict[str, Any]]] = {}
        for r in parent_rows:
            if id(r) in used_p: continue
            k = _norm_text(r.get("qtext") or "")
            txt_ix_p.setdefault(k, []).append(r)
        for r in teen_rows:
            if id(r) in used_t: continue
            k = _norm_text(r.get("qtext") or "")
            txt_ix_t.setdefault(k, []).append(r)

        for k in set(txt_ix_p.keys()) & set(txt_ix_t.keys()):
            for p in txt_ix_p[k]:
                if id(p) in used_p: continue
                cand = None
                for t in txt_ix_t[k]:
                    if id(t) in used_t: continue
                    if (t.get("dimension") == p.get("dimension")):
                        cand = t; break
                if not cand:
                    for t in txt_ix_t[k]:
                        if id(t) not in used_t:
                            cand = t; break
                if cand:
                    res.append(self._mk_pair(p, cand, p.get("qid_canon") or cand.get("qid_canon")))
                    used_p.add(id(p)); used_t.add(id(cand))

        # 3) fuzzy match within dimension; output unpaired as single-sided
        remain_p = [r for r in parent_rows if id(r) not in used_p]
        remain_t = [r for r in teen_rows if id(r) not in used_t]

        bucket_p: Dict[str, List[Dict[str, Any]]] = {}
        bucket_t: Dict[str, List[Dict[str, Any]]] = {}
        for r in remain_p:
            bucket_p.setdefault(r.get("dimension") or "Misc", []).append(r)
        for r in remain_t:
            bucket_t.setdefault(r.get("dimension") or "Misc", []).append(r)

        for dim in set(bucket_p.keys()) | set(bucket_t.keys()):
            Ps = bucket_p.get(dim, [])
            Ts = bucket_t.get(dim, [])
            taken_t = set()
            for p in Ps:
                best = None; best_sim = 0.0; best_idx = -1
                for i, t in enumerate(Ts):
                    if i in taken_t: continue
                    sim = _jaccard(p.get("qtext") or "", t.get("qtext") or "")
                    if sim > best_sim:
                        best_sim, best, best_idx = sim, t, i
                if best and best_sim >= FUZZY_JACCARD_THRESHOLD:
                    res.append(self._mk_pair(p, best, p.get("qid_canon") or best.get("qid_canon")))
                    taken_t.add(best_idx)
                else:
                    res.append(self._mk_pair(p, None, p.get("qid_canon")))
            for i, t in enumerate(Ts):
                if i not in taken_t:
                    res.append(self._mk_pair(None, t, t.get("qid_canon")))
        return res

    def _mk_pair(self, p: Optional[Dict[str, Any]], t: Optional[Dict[str, Any]], qid_canon: Optional[str]) -> Dict[str, Any]:
        return {
            "dimension": (p or t or {}).get("dimension") or "Misc",
            "qid_parent": (p or {}).get("qid"),
            "qid_teen": (t or {}).get("qid"),
            "qid_canon": qid_canon,
            "qtext_parent": (p or {}).get("qtext"),
            "qtext_teen": (t or {}).get("qtext"),
            "type_parent": (p or {}).get("type"),
            "type_teen": (t or {}).get("type"),
            "value_parent": (p or {}).get("value"),
            "value_teen": (t or {}).get("value"),
            "label_parent": (p or {}).get("label"),
            "label_teen": (t or {}).get("label"),
            "free_parent": (p or {}).get("free"),
            "free_teen": (t or {}).get("free"),
        }

    # ---------- text evidence ----------
    def _shorten(self, s: Optional[str], n: int = 160) -> Optional[str]:
        if not s:
            return None
        s = str(s).strip()
        return (s[:n] + "…") if len(s) > n else s

    def gather_free_text_evidence(
        self, child_name: str, max_dims: int = 6, top_k_per_dim: int = 1, max_snippet_len: int = 160
    ) -> List[Dict[str, Any]]:
        """Collect recent free-text/label snippets per dimension for LLM prompt evidence."""
        df = self.by_child_pairs(child_name)
        if df.empty:
            return []
        g = df.copy()
        has_text = (
            g["free_parent"].notna() | g["free_teen"].notna() |
            g["label_parent"].notna() | g["label_teen"].notna()
        )
        g = g[has_text].copy()
        if g.empty:
            return []

        g = g.sort_values("ts").groupby(["qid_canon", "dimension"], dropna=False).tail(1)
        g = g.sort_values("ts", ascending=False)

        out = []
        for dim, gd in g.groupby("dimension", sort=False):
            gd = gd.sort_values("ts", ascending=False).head(top_k_per_dim)
            for _, r in gd.iterrows():
                teen_text = r["free_teen"] or r["label_teen"]
                parent_text = r["free_parent"] or r["label_parent"]
                if not teen_text and not parent_text:
                    continue
                out.append({
                    "dimension": dim,
                    "month": r["month"],
                    "question_text": r["qtext_teen"] or r["qtext_parent"],
                    "teen_text": self._shorten(teen_text, max_snippet_len),
                    "parent_text": self._shorten(parent_text, max_snippet_len),
                })
            if len(out) >= max_dims * top_k_per_dim:
                break
        return out

    def _inject_free_text_from_ard(
        self,
        data: Dict[str, Any],
        parent_rows: List[Dict[str, Any]],
        teen_rows: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Write free_text from analysis_ready_data.free_text_summary back to item rows."""
        ard = (data or {}).get("analysis_ready_data", {})
        fts = (ard.get("free_text_summary") or {})

        for side_name, bag, rows in [
            ("parent", fts.get("parent") or [], parent_rows),
            ("teenager", fts.get("teenager") or [], teen_rows)
        ]:
            for item in bag:
                qid = item.get("question_id")
                qtext = item.get("question_text")
                dim = item.get("dimension") or "Misc"
                free = (item.get("free_text_response") or "").strip()
                if not free:
                    continue
                canon = _canon_qid(qid)
                norm_q = _norm_text(qtext or "")

                # try to update existing row by canon or (text+dimension)
                hit = None
                for r in rows:
                    if (_canon_qid(r.get("qid")) == canon) or (
                        _norm_text(r.get("qtext") or "") == norm_q and (r.get("dimension") or "Misc") == dim
                    ):
                        hit = r; break
                if hit:
                    if not hit.get("free"):
                        hit["free"] = free
                    hit["qtext"] = hit.get("qtext") or qtext
                    hit["dimension"] = hit.get("dimension") or dim
                else:
                    rows.append({
                        "side": side_name,
                        "qid": str(qid) if qid is not None else None,
                        "qid_canon": canon,
                        "qtext": qtext,
                        "dimension": dim,
                        "type": "free_text",
                        "value": None,
                        "label": None,
                        "free": free
                    })
        return parent_rows, teen_rows

    # ---------- math ----------
    def _slope(self, xs: List[float], ys: List[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        xm = sum(xs) / n
        ym = sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs) or 1e-9
        return num / den

# -------------------- module-level helpers --------------------
_GLOBAL_STORE: Optional[MHDataStore] = None

def load_store(data_root: str) -> MHDataStore:
    """External entry: load_store('./data')."""
    global _GLOBAL_STORE
    ds = MHDataStore().load_root(data_root)
    _GLOBAL_STORE = ds
    return ds

def get_store() -> MHDataStore:
    assert _GLOBAL_STORE is not None, "Call load_store(data_root) first."
    return _GLOBAL_STORE
