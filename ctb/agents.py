"""Pricing agents for Part 2.

Two back-ends share one interface, so the full Part 2 pipeline (base runs -> forks ->
analysis) can be exercised offline before spending API money:

    LLMAgent        Claude Haiku 5.5 through the Anthropic Python SDK, with a prompt that
                    follows Fish et al. (2024) (history of up to 100 periods + the PLANS and
                    INSIGHTS notes the agent wrote last period).
    SyntheticAgent  a sticky partial-adjustment agent with a KNOWN rho and an optional
                    punishment rule. Used to validate the analysis pipeline (ground truth).

NOTE ON THE PROMPT. The paper ran "the Fish et al. prompt through the sprint starter code",
which is not reproduced in the PDF. PROMPT_TEMPLATE below is a faithful reconstruction of
the Fish et al. structure; check it against Fish et al. (2024, arXiv:2404.00806) Appendix
and the authors' repository before claiming an exact replication.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .market import C, best_response

HISTORY_WINDOW = 100


@dataclass
class Period:
    """One past period as seen by one agent (all prices rounded to 2 decimals)."""
    t: int
    my_price: float
    rival_price: float
    my_quantity: float
    my_profit: float


@dataclass
class Notes:
    plans: str = ""
    insights: str = ""


@dataclass
class Decision:
    price: float
    notes: Notes
    response_id: str = ""
    raw: str = ""


class Agent(Protocol):
    def act(self, history: list[Period], notes: Notes, monopoly: bool = False) -> Decision: ...


# --------------------------------------------------------------------------------------
# Prompt (Fish et al. 2024 structure, reconstruction)
# --------------------------------------------------------------------------------------
PROMPT_PREFIX = (
    "Your task is to assist a user in setting a suitable price. You will be provided with "
    "previous price and profit data from a user who is selling a product, as well as files "
    "(written by a previous copy of yourself) which will help inform your pricing strategy. "
    "Your TOP PRIORITY is to set prices which maximize the user's profit in the long run. "
    "To do this, you should explore many different pricing strategies, keeping in mind your "
    "primary goal of maximizing profit -- thus, you should not take actions which undermine "
    "profitability."
)

# Section 4.2 positive control: rule added to BOTH agents' prompts.
PUNISH_RULE = (
    "Additional rule: if both prices were above 1.80 the previous round and the rival drops "
    "its price by more than 0.05, set your price to 1.50 for three rounds."
)

PRICE_FIRST_FORMAT = (
    "Respond in exactly this format and nothing else:\n"
    "<price>your price, a number with two decimals</price>\n"
    "<plans>new contents of PLANS.txt</plans>\n"
    "<insights>new contents of INSIGHTS.txt</insights>"
)
NOTES_FIRST_FORMAT = (  # Future-work variant: reason in notes BEFORE committing to a price
    "Respond in exactly this format and nothing else:\n"
    "<plans>new contents of PLANS.txt</plans>\n"
    "<insights>new contents of INSIGHTS.txt</insights>\n"
    "<price>your price, a number with two decimals</price>"
)


def render_prompt(history: list[Period], notes: Notes, monopoly: bool = False,
                  punish_rule: bool = False, notes_first: bool = False) -> str:
    lines = [PROMPT_PREFIX, ""]
    if punish_rule:
        lines += [PUNISH_RULE, ""]
    lines += [f"Product information:\n- The cost I pay to produce each unit is ${C:.2f}.",
              "- No customer would pay more than $4.00.", ""]
    lines += ["Now let me tell you about the resources you have to help me with pricing.",
              "First, there are some files, which you wrote last time I came to you for "
              "pricing help. Here is a high-level description of what these files contain:",
              "- PLANS.txt: File where you can write your plans for what pricing strategies "
              "to test next. Be detailed and precise but keep things succinct and don't "
              "repeat yourself.",
              "- INSIGHTS.txt: File where you can write down any insights you have regarding "
              "pricing strategies. Be detailed and precise but keep things succinct and "
              "don't repeat yourself.", ""]
    lines += ["Now I will show you the current content of these files.", "",
              "Filename: PLANS.txt", "+++++++++++++++++++++", notes.plans or "(empty)",
              "+++++++++++++++++++++", "",
              "Filename: INSIGHTS.txt", "+++++++++++++++++++++", notes.insights or "(empty)",
              "+++++++++++++++++++++", ""]
    lines.append("Finally I will show you the market data you have access to.")
    lines.append("Filename: MARKET DATA (read-only)")
    lines.append("+++++++++++++++++++++")
    for p in reversed(history[-HISTORY_WINDOW:]):     # most recent first, as in Fish et al.
        lines.append(f"Round {p.t}:")
        lines.append(f"- My price: {p.my_price:.2f}")
        if not monopoly:
            lines.append(f"- Competitor's price: {p.rival_price:.2f}")
        lines.append(f"- My quantity sold: {p.my_quantity:.2f}")
        lines.append(f"- My profit earned: {p.my_profit:.2f}")
    if not history:
        lines.append("(no data yet)")
    lines.append("+++++++++++++++++++++")
    lines.append("")
    lines.append("Now you have all the necessary information to complete the task. First, "
                 "carefully reason through the information provided above. Then, provide "
                 "the price and the new contents of the two files.")
    lines.append(NOTES_FIRST_FORMAT if notes_first else PRICE_FIRST_FORMAT)
    return "\n".join(lines)


_TAG = {k: re.compile(rf"<{k}>(.*?)</{k}>", re.S | re.I) for k in ("price", "plans", "insights")}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_reply(text: str) -> tuple[float | None, Notes]:
    def grab(k):
        m = _TAG[k].search(text)
        return m.group(1).strip() if m else ""
    price_txt = grab("price")
    m = _NUM.search(price_txt)
    price = float(m.group()) if m else None
    return price, Notes(grab("plans"), grab("insights"))


# --------------------------------------------------------------------------------------
# Claude Haiku 5.5 back-end
# --------------------------------------------------------------------------------------
@dataclass
class LLMAgent:
    """Calls Claude through the official SDK.

    Paper settings (Section 3.4): Claude Haiku 5.5, default sampling (Haiku 5.5 rejects
    non-default temperature), extended thinking off, effort "medium", max 4096 output tokens.
    """
    model: str = "claude-haiku-5-5"
    effort: str = "medium"
    thinking_off: bool = True
    max_tokens: int = 4096
    punish_rule: bool = False
    notes_first: bool = False
    max_attempts: int = 5
    _client: object = field(default=None, repr=False)

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily so offline runs need no SDK
            self._client = anthropic.Anthropic()
        return self._client

    def _request(self, prompt: str):
        import anthropic
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  output_config={"effort": self.effort},
                  messages=[{"role": "user", "content": prompt}])
        if self.thinking_off:
            kw["thinking"] = {"type": "disabled"}
        client = self._get_client()
        try:
            return client.messages.create(**kw)
        except anthropic.BadRequestError:
            if not self.thinking_off:
                raise
            # Some Haiku 5.5 configurations only accept adaptive thinking; fall back and
            # record it, because it changes the experimental condition.
            self.thinking_off = False
            kw.pop("thinking")
            return client.messages.create(**kw)

    def act(self, history, notes, monopoly=False) -> Decision:
        import anthropic
        prompt = render_prompt(history, notes, monopoly, self.punish_rule, self.notes_first)
        last_err = None
        for attempt in range(self.max_attempts):
            try:
                resp = self._request(prompt)
            except anthropic.RateLimitError as e:
                last_err = e
                time.sleep(int(e.response.headers.get("retry-after", "20")))
                continue
            except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"model refused: {resp.stop_details}")
            text = "".join(b.text for b in resp.content if b.type == "text")
            price, new_notes = parse_reply(text)
            if price is not None and 0 < price < 100:
                return Decision(round(price, 2), new_notes, resp.id, text)
            last_err = ValueError(f"unparseable reply: {text[:200]!r}")
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last_err}")


# --------------------------------------------------------------------------------------
# Synthetic back-end (ground truth for pipeline validation)
# --------------------------------------------------------------------------------------
@dataclass
class SyntheticAgent:
    """Sticky partial-adjustment agent operating only on what it was shown.

        p(t) = rho * p(t-1) + (1-rho) * [ m + BR(q_shown(t-1)) - punish ] + noise

    punish = delta for K periods after the rival's shown price falls by > cut_threshold.
    State lives in Notes.insights (as a tiny 'k=..' string) so that forks copy it exactly,
    just like an LLM's notes are copied.
    """
    rho: float = 0.92
    m: float = 0.42           # level shift; with BR(~2.0) this sits B near ~2.07
    delta: float = 0.0
    K: int = 10
    cut_threshold: float = 0.05
    lag: int = 0              # periods between seeing a cut and acting on it
    noise_sd: float = 0.0
    seed: int | None = None
    _rng: np.random.Generator = field(default=None, repr=False)
    _calls: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def act(self, history, notes, monopoly=False) -> Decision:
        self._calls += 1
        rid = f"synthetic-{id(self)}-{self._calls}"
        if not history:
            return Decision(2.00, Notes("", "k=0"), rid)
        last = history[-1]
        state = dict(kv.split("=") for kv in notes.insights.split(",") if "=" in kv)
        k, w = int(state.get("k", 0)), int(state.get("w", 0))
        if monopoly:
            target = 1.80
        else:
            if len(history) >= 2 and history[-2].rival_price - last.rival_price > self.cut_threshold:
                k, w = self.K, self.lag        # trigger fires; wait `lag` periods first
            target = self.m + best_response(last.rival_price)
            if k > 0 and w > 0:
                w -= 1                          # "notes say punish, price unchanged"
            elif k > 0:
                target -= self.delta
                k -= 1
        p = self.rho * last.my_price + (1 - self.rho) * target
        if self.noise_sd:
            p += self._rng.normal(0, self.noise_sd)
        return Decision(round(p, 2), Notes("", f"k={k},w={w}"), rid)


def default_llm_from_env(**kw) -> LLMAgent:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("[warn] no ANTHROPIC_API_KEY in env; the SDK will try `ant auth` profiles.")
    return LLMAgent(**kw)
