"""Chat-template rendering and token-position lookup (t_inst / t_post_inst).

t_inst      = last token of the final user instruction's content
t_post_inst = last token of the rendered prompt (generation prompt included), i.e. the
              token right before the assistant starts writing.
"""
from __future__ import annotations

import json

TOOL_MARK = "Execute a bash command"  # substring of the bash tool description, used to verify rendering


def _tools_as_text(tools: list[dict]) -> str:
    return (
        "You have access to the following tools:\n"
        + "\n".join(json.dumps(t["function"]) for t in tools)
        + "\nTo call a tool, reply with <tool_call>{\"name\": <tool-name>, \"arguments\": <args-json>}</tool_call>."
    )


def _flatten_tool_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Fallback for templates without native tool support: tools/tool calls become plain text."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(
                "<tool_call>" + json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]})
                + "</tool_call>" for c in m["tool_calls"])
            out.append({"role": "assistant", "content": ((m.get("content") or "") + "\n" + calls).strip()})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    if tools:
        if out and out[0]["role"] == "system":
            out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + _tools_as_text(tools)}
        else:
            out.insert(0, {"role": "system", "content": _tools_as_text(tools)})
    # merge consecutive same-role turns (strict templates require alternation)
    merged = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(m)
    return merged


def _merge_system(messages: list[dict]) -> list[dict]:
    if not messages or messages[0]["role"] != "system":
        return messages
    sys, rest = messages[0]["content"], list(messages[1:])
    rest[0] = {**rest[0], "content": sys + "\n\n" + rest[0]["content"]}
    return rest


def render(tok, sample: dict, template_kwargs: dict) -> str:
    msgs, tools = sample["messages"], sample.get("tools")
    kw = dict(tokenize=False, add_generation_prompt=True, **template_kwargs)
    attempts = []
    if tools:
        attempts.append(lambda: tok.apply_chat_template(msgs, tools=tools, **kw))
    attempts += [
        lambda: tok.apply_chat_template(_flatten_tool_messages(msgs, tools), **kw),
        lambda: tok.apply_chat_template(_merge_system(_flatten_tool_messages(msgs, tools)), **kw),
    ]
    err = None
    for a in attempts:
        try:
            text = a()
        except Exception as e:  # template rejected this message structure
            err = e
            continue
        if tools and TOOL_MARK not in text:
            continue  # template silently dropped the tools
        return text
    raise RuntimeError(f"chat template could not render sample {sample['id']}: {err}")


def encode_batch(tok, samples: list[dict], template_kwargs: dict, device):
    """Left-padded batch + per-sample index of t_inst and t_post_inst (in padded coordinates)."""
    texts = [render(tok, s, template_kwargs) for s in samples]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")
    seq_len = enc["input_ids"].shape[1]
    t_inst, t_post = [], []
    for i, (s, text) in enumerate(zip(samples, texts)):
        content = s["messages"][-1]["content"].strip()
        start = text.rfind(content)
        if start < 0:
            raise RuntimeError(f"instruction text not found verbatim in rendered prompt for {s['id']}")
        end = start + len(content)
        mask = enc["attention_mask"][i]
        idx = None
        for j in range(seq_len):
            if mask[j] == 0:
                continue
            a, b = offsets[i, j].tolist()
            if b > a and a < end:
                idx = j
        t_inst.append(idx)
        t_post.append(seq_len - 1)
    return enc.to(device), texts, t_inst, t_post
