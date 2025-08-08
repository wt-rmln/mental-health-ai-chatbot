from __future__ import annotations
import os, re, json, glob, math
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# -------------------- Adjustable Parameters --------------------

# Significant change threshold: absolute month-to-month gap change ≥ this value is marked as significant
DELTA_THRESHOLD = 1.0

# Trend threshold: absolute slope of 6-month linear regression ≥ this value is considered a trend
SLOPE_THRESHOLD = 0.25

# Jaccard threshold for approximate text matching (used for pairing questions within the same dimension)
FUZZY_JACCARD_THRESHOLD = 0.6

# Allowed file extensions
EXTS = (".json",)

# -------------------- Utility Functions --------------------

def _parse_dt(s: str) -> datetime:
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
    if dt == datetime.min:
        return "unknown"
    return f"{dt.year:04d}-{dt.month:02d}"

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _canon_qid(qid: Optional[str]) -> Optional[str]:
    if not qid:
        return None
    s = str(qid).strip().lower()
    s = re.sub(r"^(p|t|teen|parent|c)_+", "", s)  # Remove possible role prefixes
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or None

def _letter_index(ch: str) -> Optional[int]:
    """
    More robust parsing of option letters: accepts formats like "B", "b", "B)", "b.", "B ) Something", etc.
    """
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
    ta = set(_norm_text(a).split())
    tb = set(_norm_text(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

# -------------------- Data Class --------------------

@dataclass
class PairRecord:
    child_name: str
    month: str
    ts: datetime
    dimension: str

    # Question IDs / text (for both sides)
    qid_parent: Optional[str]
    qid_teen: Optional[str]
    qid_canon: Optional[str]

    qtext_parent: Optional[str]
    qtext_teen: Optional[str]

    # Question types
    type_parent: Optional[str]
    type_teen: Optional[str]

    # Numerical and text responses (for both sides)
    value_parent: Optional[float]
    value_teen: Optional[float]
    label_parent: Optional[str]      # Option text
    label_teen: Optional[str]
    free_parent: Optional[str]       # Free text
    free_teen: Optional[str]

# -------------------- Main Class --------------------

class MHDataStore:
    def __init__(self):
        # Long-form paired table
        self.pairs_df: Optional[pd.DataFrame] = None
        # Dimension-level time series (parent/teen averages + gap)
        self.ts_df: Optional[pd.DataFrame] = None
        # List of loaded children
        self._children: List[str] = []

    # ---------- Public API ----------
    def load_root(self, root_dir: str) -> "MHDataStore":
        """
        Load all children's data from the root directory. Example structure:
        data/
          alex_chen/
            response_data_20250115_initial_assessment.json
            ...
        """
        all_pairs: List[PairRecord] = []

        # Iterate over child subdirectories: data/*/
        for child_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
            if not os.path.isdir(child_dir):
                continue
            # Try to read child's name from file (participant_info.child.name), otherwise use directory name
            child_name_from_dir = os.path.basename(child_dir).replace("_", " ").title()

            for fp in sorted(glob.glob(os.path.join(child_dir, "*"))):
                if not fp.lower().endswith(EXTS):
                    continue
                data = self._safe_load_json(fp)
                if not data:
                    continue

                # Parse assessment date and child name
                assess_dt = _parse_dt(
                    data.get("assessment_session", {}).get("assessment_date")
                    or data.get("metadata", {}).get("assessment_date")
                )
                month = _month_str(assess_dt)

                child_name = (
                    data.get("participant_info", {})
                        .get("child", {})
                        .get("name")
                    or child_name_from_dir
                )

                # Build option maps (qid -> options list) for mapping letters to numeric/text
                opt_parent = self._build_option_map(
                    data.get("questionnaire_data", {}).get("parent_questionnaire", [])
                )
                opt_teen = self._build_option_map(
                    data.get("questionnaire_data", {}).get("teenager_questionnaire", [])
                )

                # Flatten both sides' responses
                parent_rows = self._flatten_side(
                    side="parent",
                    responses=data.get("responses", {}).get("parent", []) or [],
                    opt_map=opt_parent
                )
                teen_rows = self._flatten_side(
                    side="teenager",
                    responses=data.get("responses", {}).get("teenager", []) or [],
                    opt_map=opt_teen
                )

                # Pair questions: by canonical qid, then text, then fuzzy match
                pairs = self._pair_two_sides(parent_rows, teen_rows)

                # Create PairRecord entries
                for p in pairs:
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

                # Record child name
                if child_name not in self._children:
                    self._children.append(child_name)

        # Build DataFrames
        if all_pairs:
            self.pairs_df = pd.DataFrame([asdict(x) for x in all_pairs])

            # Precompute normalized question text for faster search
            self.pairs_df["qtext_parent_norm"] = self.pairs_df["qtext_parent"].fillna("").map(_norm_text)
            self.pairs_df["qtext_teen_norm"]   = self.pairs_df["qtext_teen"].fillna("").map(_norm_text)

            # Build time series (numeric questions only)
            self.ts_df = (self.pairs_df
                          .assign(parent_num=lambda d: pd.to_numeric(d["value_parent"], errors="coerce"),
                                  teen_num=lambda d: pd.to_numeric(d["value_teen"], errors="coerce"))
                          .groupby(["child_name", "month", "dimension"], as_index=False)
                          .agg(parent_avg=("parent_num", "mean"),
                               teen_avg=("teen_num", "mean"))
                          .assign(gap=lambda d: d["parent_avg"] - d["teen_avg"]))
        else:
            self.pairs_df = pd.DataFrame(columns=[f.name for f in fields(PairRecord)])
            self.ts_df = pd.DataFrame(columns=["child_name", "month", "dimension", "parent_avg", "teen_avg", "gap"])

        return self

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
        out = out.sort_values(["dimension", "month"])
        return out

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

        # Top differences (by dimension)
        latest2 = latest.assign(abs_gap=lambda d: d["gap"].abs())
        top_gaps = latest2.sort_values("abs_gap", ascending=False).drop(columns=["abs_gap"]).head(5)
        top_gaps = top_gaps.to_dict(orient="records")

        # Significant month-to-month gap changes
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

        # Trends: slope per dimension
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

    def retrieve_dual_perspective(
        self,
        child_name: str,
        query: str = "",
        dimension: Optional[str] = None,
        month: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent dual-perspective answers (with numeric and text data),
        filtered optionally by keyword/dimension/month.
        For questions that cannot be numerically scored, returns free_text/label instead.
        """
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
            # Use precomputed normalized text columns to avoid repeated .apply(_norm_text)
            g = g[g["qtext_parent_norm"].str.contains(q, regex=False) |
                  g["qtext_teen_norm"].str.contains(q, regex=False)]

        # For each (qid_canon, dimension), take the latest record (by ts), then overall take top_k newest
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
        max_snapshot_dims: int = 8,    # Limit snapshot dimensions to reduce prompt size
    ) -> str:
        """
        Generate a context summary string to embed into the LLM prompt:
        - Latest month snapshot by dimension (parent/teen/gap), top max_snapshot_dims by |gap|
        - Largest gaps (latest month)
        - Significant changes (month-to-month Δgap)
        - Trends (slope per dimension)
        - Short usage guidance
        """
        summ = self.summarize_child(child_name)
        if not summ.get("months"):
            return f"Data note: No questionnaire records found for {child_name}."

        lines: List[str] = []
        lines.append(f"Child: {child_name}; Months covered: {', '.join(summ['months'])}.")

        # Latest snapshot: keep only top N dimensions by |gap|
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
        if changes:   # Fixed original 'or True' bug so flag works
            lines.append("Notable month-to-month changes in gap:")
            for c in changes[:max_items]:
                lines.append(f"- {c['dimension']} | {c['month_from']}→{c['month_to']}: "
                             f"Δgap={_fmt_num(c['delta_gap'])} ({c['label']})")

        if summ["trends"]:
            lines.append("Trends across months (gap slope):")
            for t in summ["trends"][:max_items]:
                lines.append(f"- {t['dimension']}: slope={_fmt_num(t['slope'])} ({t['direction']})")

        lines.append("Guidance: Use dual-perspective answers, highlight gaps with brief explanations, "
                     "and offer actionable parenting suggestions (routines, communication, activities).")
        return "\n".join(lines)

    # ---------- Internal: parsing and pairing ----------

    def _safe_load_json(self, fp: str) -> Optional[Dict[str, Any]]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _build_option_map(self, qs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return mapping: qid -> list of options (each with at least value/text).
        """
        mp: Dict[str, List[Dict[str, Any]]] = {}
        for q in qs or []:
            qid = q.get("id") or q.get("question_id")
            if not qid:
                continue
            opts = q.get("options") or []
            mp[str(qid)] = opts
        return mp

    def _flatten_side(
        self,
        side: str,  # "parent" or "teenager"
        responses: List[Dict[str, Any]],
        opt_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Flatten one side's responses into a uniform structure.
        Output fields:
        - qid / qid_canon / qtext / dimension / type
        - value (numeric; from option value if applicable)
        - label (option text)
        - free (free text)
        """
        out = []
        for r in responses or []:
            qid = r.get("question_id")
            qtext = r.get("question_text")
            dimension = r.get("dimension") or "Misc"
            rtype = r.get("response_type")
            selected_value = r.get("selected_value")
            selected_opt = r.get("selected_option")
            free_text = (r.get("free_text_response") or "").strip() or None

            # Map option -> numeric value and text label
            value = _try_float(selected_value)
            label = None

            if value is None:
                # Try: if only option letter is given, map by option order
                idx = _letter_index(selected_opt)
                if idx is not None:
                    opts = opt_map.get(str(qid), [])
                    if 0 <= idx < len(opts):
                        value = _try_float(opts[idx].get("value"))
                        label = (opts[idx].get("text") or "").strip() or None

            if label is None and value is not None:
                # Reverse lookup: if we have value, find corresponding option text
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

    def _pair_two_sides(
        self,
        parent_rows: List[Dict[str, Any]],
        teen_rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Pair parent and teenager questions:
        1) Exact match by qid_canon
        2) Exact match by normalized question text
        3) Fuzzy match within the same dimension (Jaccard ≥ threshold)
        Unpaired items on one side will still be output with the other side empty.
        """
        res: List[Dict[str, Any]] = []

        # Index by canonical qid
        by_qid_p = {r["qid_canon"]: r for r in parent_rows if r.get("qid_canon")}
        by_qid_t = {r["qid_canon"]: r for r in teen_rows if r.get("qid_canon")}

        used_p, used_t = set(), set()

        # 1) Exact match by qid_canon
        for qid_c in set(by_qid_p.keys()) & set(by_qid_t.keys()):
            p = by_qid_p[qid_c]; t = by_qid_t[qid_c]
            res.append(self._mk_pair(p, t, qid_c))
            used_p.add(id(p)); used_t.add(id(t))

        # 2) Exact match by normalized question text
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
            # When one-to-many, prefer matching dimensions
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

        # 3) Fuzzy match within dimension (Jaccard)
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

    # ---------- Math helper ----------

    def _slope(self, xs: List[float], ys: List[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        xm = sum(xs) / n
        ym = sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs) or 1e-9
        return num / den

# -------------------- Module-level helper functions --------------------

_GLOBAL_STORE: Optional[MHDataStore] = None

def load_store(data_root: str) -> MHDataStore:
    """
    External call: load_store("./data").
    Directory structure: ./data/<child_slug>/*.json
    """
    global _GLOBAL_STORE
    ds = MHDataStore().load_root(data_root)
    _GLOBAL_STORE = ds
    return ds

def get_store() -> MHDataStore:
    assert _GLOBAL_STORE is not None, "Call load_store(data_root) first."
    return _GLOBAL_STORE