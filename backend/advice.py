"""
LLM-generated plant-care guidance (What it is / Treat it / Prevent it).

Generated once per (plant, disease) with OpenAI, then cached in Supabase
(`recommendations`) so it becomes a growing knowledge base that experts can later
edit. Degrades safely: if no OPENAI_API_KEY is set, advice is simply omitted.
"""

import os

import db  # reuses the same .env loader + Supabase REST helpers

KEY = (db._env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")).strip()
ENABLED = bool(KEY and not KEY.startswith("PASTE_"))
MODEL = (db._env.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

_SCHEMA = {
    "type": "object",
    "properties": {
        "what_it_is": {"type": "string"},
        "treat": {"type": "string"},
        "prevent": {"type": "string"},
    },
    "required": ["what_it_is", "treat", "prevent"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a practical plant pathologist writing for farmers and gardeners. "
    "Given a plant and a disease (or pest), return concise field guidance. "
    "Each field is 1-2 plain sentences, no markdown, no lists. "
    "'what_it_is': the cause and how it spreads. "
    "'treat': concrete first actions to take now. "
    "'prevent': how to avoid it next season. "
    "Be safe and non-prescriptive about chemicals — say to follow local label rates "
    "and consult an agronomist for products. If the exact pairing is unfamiliar, give "
    "best practice for that disease type rather than inventing specifics."
)


def generate(plant: str, disease: str) -> dict:
    """Call OpenAI to produce care guidance. Raises on API error."""
    import json
    from openai import OpenAI

    client = OpenAI(api_key=KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Plant: {plant}\nDisease: {disease}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "care_guide", "strict": True, "schema": _SCHEMA},
        },
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "what_it_is": data["what_it_is"],
        "treat": data["treat"],
        "prevent": data["prevent"],
        "model": MODEL,
    }


_CHAT_SYSTEM = (
    "You are Sage, AgroCure's intelligent plant-care assistant — friendly, knowledgeable, "
    "and trustworthy. You're chatting with a farmer or gardener about ONE specific leaf they "
    "just had diagnosed. Stay focused on this case and general plant-health practice. Be warm "
    "and concise — 2 to 4 short sentences, plain language, no markdown, no headings, no bullet "
    "lists. Give concrete, practical guidance. For any chemical or pesticide, say to follow the "
    "local product label rates and consult an agronomist rather than naming exact doses. If the "
    "diagnosis was uncertain and handed to an expert, be honest that a human is confirming it. "
    "Never say you are an AI language model — you are Sage. If asked something unrelated to "
    "plants, gently steer back to the leaf."
)


def _context_blurb(ctx: dict) -> str:
    plant = ctx.get("plant") or "Unknown plant"
    disease = ctx.get("disease")
    healthy = ctx.get("healthy")
    conf = ctx.get("confidence")
    uncertain = ctx.get("uncertain")
    parts = [f"Plant: {plant}."]
    if healthy:
        parts.append("The leaf was assessed as healthy — no disease detected.")
    elif disease:
        parts.append(f"Diagnosis: {disease}.")
    if conf is not None:
        try:
            parts.append(f"Model confidence: {round(float(conf) * 100)}%.")
        except (TypeError, ValueError):
            pass
    if uncertain:
        parts.append("This case was uncertain and has been sent to a human expert to confirm.")
    return " ".join(parts)


def chat(ctx: dict, messages: list) -> str:
    """Free-form follow-up Q&A grounded in the user's diagnosis. Raises on API error."""
    from openai import OpenAI

    client = OpenAI(api_key=KEY)
    convo = [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "system", "content": "This conversation is about: " + _context_blurb(ctx or {})},
    ]
    for m in (messages or [])[-12:]:
        role = "user" if (m.get("role") == "user") else "assistant"
        content = str(m.get("content", "")).strip()[:1500]
        if content:
            convo.append({"role": role, "content": content})
    # Mirror the (working) Care Guide call. Some newer models reject custom
    # temperature / max_tokens — fall back to a bare call so chat works wherever
    # the Care Guide does.
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=convo, temperature=0.4, max_tokens=320,
        )
    except Exception:
        resp = client.chat.completions.create(model=MODEL, messages=convo)
    return (resp.choices[0].message.content or "").strip()


def get_or_create(plant: str, disease: str) -> dict | None:
    """Return cached guidance, or generate + cache it. None if unavailable."""
    if not plant or not disease or disease.lower() == "healthy":
        return None
    cached = db.get_recommendation(plant, disease)
    if cached:
        return cached
    if not ENABLED:
        return None
    gen = generate(plant, disease)
    if db.ENABLED:
        try:
            db.insert("recommendations", {"plant": plant, "disease": disease,
                                          "source": "llm", **gen})
        except Exception:
            pass  # serve it even if caching failed
    return gen
