"""
Australian Political Compass — Streamlit survey app.

Reads the respondent-facing question text from `compass_survey.md` and the
alignment matrix from `compass_scoring_matrix.csv`, presents the 30 questions in
shuffled order (party labels and scoring hidden), and reports each party's
"match %" using the instrument's scoring method.

Run with:  streamlit run compass_app.py
"""

from __future__ import annotations

import csv
import html
import json
import random
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent
SURVEY_MD = HERE / "compass_survey.md"
MATRIX_CSV = HERE / "compass_scoring_matrix.csv"

PARTIES = ["Labor", "Coalition", "One Nation", "Greens"]
# CSV column name -> display name (CSV uses "OneNation")
CSV_PARTY_COLS = {"Labor": "Labor", "Coalition": "Coalition", "OneNation": "One Nation", "Greens": "Greens"}

PARTY_COLOURS = {
    "Labor": "#E13B3B",       # red
    "Coalition": "#1F6FB2",   # blue
    "One Nation": "#F36C21",  # orange
    "Greens": "#39A935",      # green
}

ANSWER_OPTIONS = [
    "Strongly Agree",
    "Somewhat Agree",
    "Neutral",
    "Somewhat Disagree",
    "Strongly Disagree",
]
# Strongly/Somewhat collapsed for scoring, per the instrument design.
ANSWER_VALUE = {
    "Strongly Agree": 1,
    "Somewhat Agree": 1,
    "Neutral": 0,
    "Somewhat Disagree": -1,
    "Strongly Disagree": -1,
}
DEFAULT_ANSWER = "Neutral"

# Implementation lead-ins to underline in the statement (separates the big idea
# from the concrete "how").
CONNECTORS = ("To achieve this", "To follow this principle", "To rectify this")


def _format_statement(text: str) -> str:
    """HTML-escape the statement and underline the implementation lead-in."""
    safe = html.escape(text, quote=False)
    for c in CONNECTORS:
        safe = safe.replace(c, f"<u>{c}</u>")
    return safe


# --------------------------------------------------------------------------- #
# Loading / parsing
# --------------------------------------------------------------------------- #
@st.cache_data
def load_alignment(path: str) -> dict[str, dict[str, int]]:
    """qid -> {party: alignment}."""
    align: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            align[row["qid"]] = {disp: int(row[col]) for col, disp in CSV_PARTY_COLS.items()}
    return align


@st.cache_data
def load_meta(path: str) -> dict[str, dict[str, str]]:
    """qid -> {domain, big_idea} for labelling results."""
    meta: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["qid"]] = {"domain": row["domain"], "big_idea": row["big_idea"]}
    return meta


@st.cache_data
def load_questions(path: str) -> list[dict]:
    """
    Parse compass_survey.md into a list of respondent-facing questions:
    {qid, domain, text (stem + implementation), tradeoff}. Scoring rows and
    party labels are deliberately not surfaced to the respondent.
    """
    md = Path(path).read_text(encoding="utf-8")
    # Split on question headers, capturing qid and the rest of the header line.
    parts = re.split(r"^### (Q\d+) · (.+)$", md, flags=re.M)
    questions: list[dict] = []
    for i in range(1, len(parts), 3):
        qid = parts[i]
        header_rest = parts[i + 1].strip()          # e.g. "Greens · platform · Climate"
        body = parts[i + 2]
        domain = [p.strip() for p in header_rest.split("·")][-1]

        stem_lines: list[str] = []
        tradeoff = ""
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("*Scoring"):
                continue
            if s.startswith("**Known trade-off:**"):
                tradeoff = s.replace("**Known trade-off:**", "").strip()
                continue
            if s.startswith("---") or s.startswith("##"):
                break
            stem_lines.append(s)
        questions.append(
            {"qid": qid, "domain": domain, "text": " ".join(stem_lines), "tradeoff": tradeoff}
        )
    return questions


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def compute_results(answers: dict[str, str], align: dict[str, dict[str, int]]) -> pd.DataFrame:
    """
    raw_p   = Σ answer_value × alignment(q, p)        over answered questions
    max_p   = Σ |alignment(q, p)|                     over ALL questions (the pool)
    match%  = (raw_p + max_p) / (2 · max_p) · 100     -> 0..100, 50 = neutral
    """
    raw = {p: 0 for p in PARTIES}
    pool = {p: 0 for p in PARTIES}
    for qid, al in align.items():
        for p in PARTIES:
            pool[p] += abs(al[p])
    for qid, ans in answers.items():
        v = ANSWER_VALUE[ans]
        if v == 0 or qid not in align:
            continue
        for p in PARTIES:
            raw[p] += v * align[qid][p]

    rows = []
    for p in PARTIES:
        match = (raw[p] + pool[p]) / (2 * pool[p]) * 100 if pool[p] else 50.0
        rows.append({"Party": p, "Match %": round(match, 1), "raw": raw[p], "pool": pool[p]})
    return pd.DataFrame(rows).sort_values("Match %", ascending=False).reset_index(drop=True)


def alignment_breakdown(answers: dict[str, str], align: dict[str, dict[str, int]]) -> dict[str, list[tuple]]:
    """party -> [(qid, answer, alignment_value), ...] for questions you lined up with."""
    result: dict[str, list[tuple]] = {p: [] for p in PARTIES}
    for qid, ans in answers.items():
        v = ANSWER_VALUE[ans]
        if v == 0 or qid not in align:
            continue
        for p in PARTIES:
            if v * align[qid][p] > 0:  # answer pushed this party's score up
                result[p].append((qid, ans, align[qid][p]))
    return result


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def results_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Match %:Q", scale=alt.Scale(domain=[0, 100]), title="Match %"),
            y=alt.Y("Party:N", sort="-x", title=None),
            color=alt.Color(
                "Party:N",
                scale=alt.Scale(domain=list(PARTY_COLOURS), range=list(PARTY_COLOURS.values())),
                legend=None,
            ),
            tooltip=["Party", "Match %"],
        )
        .properties(height=180)
    )


def _save_and_move(qid: str, delta: int, goto_results: bool = False) -> None:
    """Callback: persist the current question's answer, then navigate."""
    widget_key = f"ans_{qid}"
    if widget_key in st.session_state:
        st.session_state.setdefault("answers", {})[qid] = st.session_state[widget_key]
    if goto_results:
        st.session_state["phase"] = "results"
    else:
        total = st.session_state.get("num_q", 1)
        st.session_state["idx"] = min(max(st.session_state.get("idx", 0) + delta, 0), total - 1)


def _reset() -> None:
    """Clear all survey state (incl. per-question widgets) for a fresh run."""
    for key in list(st.session_state.keys()):
        if key in {"order", "idx", "answers", "phase", "results"} or key.startswith("ans_"):
            st.session_state.pop(key, None)


def _render_results(
    df: pd.DataFrame,
    answers: dict[str, str],
    align: dict[str, dict[str, int]],
    meta: dict[str, dict[str, str]],
) -> None:
    top = df.iloc[0]
    st.title("🧭 Your results")
    st.markdown(f"### Closest match: **{top['Party']}** ({top['Match %']}%)")
    st.altair_chart(results_chart(df), use_container_width=True)
    st.dataframe(df[["Party", "Match %"]].set_index("Party"), use_container_width=True)

    answered = [qid for qid, a in answers.items() if ANSWER_VALUE[a] != 0]
    breakdown = alignment_breakdown(answers, align)

    with st.expander("Details & data"):
        st.write(
            "Match % = (raw + pool) / (2 × pool) × 100, where *raw* is the sum of "
            "answer × alignment and *pool* is the party's maximum achievable points."
        )
        st.dataframe(df.set_index("Party"), use_container_width=True)

        st.markdown("#### Where you lined up with each party")
        st.caption(
            f"Of the {len(answered)} questions you didn't leave Neutral, here is where your "
            "answer matched each party's position (you agreed with what they back, or "
            "disagreed with what they oppose)."
        )
        for _, prow in df.iterrows():
            party = prow["Party"]
            items = sorted(breakdown[party], key=lambda t: int(t[0][1:]))
            st.markdown(
                f"**You are aligned with {party} on the following "
                f"{len(items)} of {len(answered)} questions:**"
            )
            if items:
                st.markdown(
                    "\n".join(
                        f"- **{meta[qid]['domain']}** — {meta[qid]['big_idea'].replace('$', r'\$')} "
                        f"*(you answered {ans})*"
                        for qid, ans, _av in items
                    )
                )
            else:
                st.markdown("- _none_")

        st.download_button(
            "Download my responses (JSON)",
            data=json.dumps({"answers": answers, "results": df.to_dict(orient="records")}, indent=2),
            file_name="compass_responses.json",
            mime="application/json",
        )
    st.button("Start over", on_click=_reset, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Australian Political Compass", page_icon="🧭", layout="centered")

    questions = load_questions(str(SURVEY_MD))
    align = load_alignment(str(MATRIX_CSV))
    total = len(questions)
    st.session_state["num_q"] = total

    # One-time per-session init (randomised question order).
    if "order" not in st.session_state:
        order = list(range(total))
        random.shuffle(order)
        st.session_state["order"] = order
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("phase", "intro")

    phase = st.session_state["phase"]

    # ----- Intro -----
    if phase == "intro":
        st.title("🧭 Australian Political Compass")
        st.write(
            f"Answer {total} short statements, one at a time, saying how much you agree. "
            "There are no party labels — your answers are matched against where the four "
            "parties stand. Every party can reach 100%, so a high score means close alignment."
        )
        st.caption("You can go back and change any answer before submitting.")
        st.button(
            "Start",
            type="primary",
            use_container_width=True,
            on_click=lambda: st.session_state.update(phase="survey"),
        )
        return

    # ----- Results -----
    if phase == "results":
        answers = {q["qid"]: st.session_state["answers"].get(q["qid"], DEFAULT_ANSWER) for q in questions}
        meta = load_meta(str(MATRIX_CSV))
        _render_results(compute_results(answers, align), answers, align, meta)
        return

    # ----- Survey: one question at a time -----
    idx = st.session_state["idx"]
    q = questions[st.session_state["order"][idx]]

    st.progress((idx + 1) / total, text=f"Question {idx + 1} of {total}")

    st.markdown(
        "**Do you agree or disagree with the statement below?** "
        "Please consider the known trade-off before answering."
    )
    st.markdown(
        f"<p style='font-size:1.05rem; line-height:1.5; margin:0.75rem 0;'>"
        f"{_format_statement(q['text'])}</p>",
        unsafe_allow_html=True,
    )
    if q["tradeoff"]:
        st.markdown(
            "<div style='border:1px solid #000; border-radius:6px; "
            "padding:0.55rem 0.8rem; font-size:0.85rem; line-height:1.45; margin:0.25rem 0 1.1rem;'>"
            f"<strong>Known trade-off:</strong> {html.escape(q['tradeoff'], quote=False)}</div>",
            unsafe_allow_html=True,
        )

    prev = st.session_state["answers"].get(q["qid"], DEFAULT_ANSWER)
    st.radio(
        "Your view",
        options=ANSWER_OPTIONS,
        index=ANSWER_OPTIONS.index(prev),
        key=f"ans_{q['qid']}",
        label_visibility="collapsed",
    )

    st.write("")
    back_col, next_col = st.columns(2)
    back_col.button(
        "← Back",
        use_container_width=True,
        disabled=(idx == 0),
        on_click=_save_and_move,
        args=(q["qid"], -1),
    )
    if idx < total - 1:
        next_col.button(
            "Next →", type="primary", use_container_width=True,
            on_click=_save_and_move, args=(q["qid"], +1),
        )
    else:
        next_col.button(
            "See my results", type="primary", use_container_width=True,
            on_click=_save_and_move, args=(q["qid"], 0, True),
        )


if __name__ == "__main__":
    main()
