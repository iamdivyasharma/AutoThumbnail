# orchestrator_stdio.py
# Orchestrator that calls downstream MCP servers over **stdio** using fastmcp.Client

from __future__ import annotations
import os, json, asyncio, hashlib
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
from time import time

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# pip install fastmcp
from fastmcp import Client as MCPClient

# Optional: AWS Bedrock (Claude) for routing/composition
import boto3
from botocore.config import Config

# ==================== Config ====================
ORCH_PORT = int(os.getenv("ORCH_PORT", "9000"))

# Commands to spawn each stdio MCP server
# e.g., "python hotel_server_stdio.py"
HOTEL_CMD = os.getenv("HOTEL_CMD", "python hotel_server_stdio.py")
AIR_CMD   = os.getenv("AIR_CMD",   "python air_server_stdio.py")
TDNA_CMD  = os.getenv("TDNA_CMD",  "python tdna_server_stdio.py")

# Logical services → stdio cmd + tool name + input field
SERVICES: Dict[str, Dict[str, str]] = {
    "hotel": {"cmd": HOTEL_CMD, "tool": "hotel_query_tool", "field": os.getenv("HOTEL_FIELD", "qquery")},
    "air":   {"cmd": AIR_CMD,   "tool": "air_query_tool",   "field": os.getenv("AIR_FIELD",   "qquery")},
    "tdna":  {"cmd": TDNA_CMD,  "tool": "tdna_query_tool",  "field": os.getenv("TDNA_FIELD",  "qquery")},
}

# ==================== Bedrock LLM (optional) ====================
BEDROCK_REGION    = os.getenv("BEDROCK_REGION", "us-east-1")
CLAUDE_MODEL_ID   = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
ANTHROPIC_VERSION = "bedrock-2023-05-31"

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2, "mode": "standard"}),
)

def _invoke_bedrock(body: dict) -> dict:
    resp = bedrock.invoke_model(modelId=CLAUDE_MODEL_ID, body=json.dumps(body))
    return json.loads(resp["body"].read().decode("utf-8"))

def claude_json(system: str, user_obj: dict, max_tokens=256, temperature=0):
    payload = _invoke_bedrock({
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role":"user","content":[{"type":"text","text": json.dumps(user_obj)}]}],
    })
    return json.loads(payload["content"][0]["text"])

def claude_text(system: str, user_obj: dict, max_tokens=512, temperature=0) -> str:
    payload = _invoke_bedrock({
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role":"user","content":[{"type":"text","text": json.dumps(user_obj)}]}],
    })
    return payload["content"][0]["text"].strip()

# ==================== Prompts ====================
ROUTER_SYSTEM = (
    "You are a strict router for a multi-tool system. Tools: air, hotel, tdna. "
    'Return ONLY JSON like {"targets":["air"|"hotel"|"tdna", ...]}'
)
COMPOSER_SYSTEM = (
    "You are a response composer. Combine multiple tool answers (air/hotel/tdna) "
    "into one concise reply. Be actionable; mention conflicts briefly if any."
)

# ==================== Memory & Cache ====================
MEMORY: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"history": deque(maxlen=20), "created": time()})
def mem_append(sid: str, role: str, content: Any): MEMORY[sid]["history"].append({"role": role, "content": content, "ts": time()})
def mem_get(sid: str, k: int = 10) -> List[Dict[str, Any]]:
    hist = list(MEMORY[sid]["history"]); return hist[-k:] if k else hist
def mem_clear(sid: str): MEMORY.pop(sid, None)

CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
_CACHE: Dict[str, Tuple[float, Any]] = {}
def _h(obj: Any) -> str: return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
def cache_get(k: str):
    rec = _CACHE.get(k)
    if not rec: return None
    exp, val = rec
    if exp < time(): _CACHE.pop(k, None); return None
    return val
def cache_set(k: str, v: Any, ttl: int = CACHE_TTL_SEC): _CACHE[k] = (time()+ttl, v)

# ==================== Core helpers ====================
def compose_final(user_query: str, answers_by_target: Dict[str, Any]) -> str:
    if not answers_by_target: return ""
    ckey = f"compose:{_h({'q':user_query,'a':answers_by_target})}"
    hit = cache_get(ckey)
    if hit is not None: return hit
    try:
        txt = claude_text(COMPOSER_SYSTEM, {"user_query": user_query, "answers_by_target": answers_by_target}, max_tokens=512)
        cache_set(ckey, txt); return txt
    except Exception:
        parts = [f"{k}: {v}" for k, v in answers_by_target.items()]
        txt = " | ".join(map(str, parts))
        cache_set(ckey, txt); return txt

def llm_decide_targets(query: str) -> List[str]:
    try:
        out = claude_json(ROUTER_SYSTEM, {"query": query}, max_tokens=128)
        picks = [t for t in out.get("targets", []) if t in SERVICES]
        return picks or ["hotel"]
    except Exception:
        return ["hotel"]

# ==================== Downstream MCP calls (stdio) ====================
async def call_backend(target: str, query: str, extra: Optional[Dict[str, Any]]):
    svc = SERVICES[target]
    payload = {svc["field"]: query}
    if isinstance(extra, dict): payload.update(extra)

    ckey = f"bot:{target}:{_h({'q':query,'x':extra})}"
    hit = cache_get(ckey)
    if hit is not None: return hit

    # Spawn the stdio MCP server process and call its tool
    cmd = svc["cmd"]
    try:
        async with MCPClient(cmd) as client:
            result = await client.call_tool(svc["tool"], payload)
        # Normalize typical shapes
        if isinstance(result, dict):
            val = result.get("answer") or result.get("result") or result.get("output") or result.get("content") or result
        else:
            val = getattr(result, "json", None) or getattr(result, "data", None) or getattr(result, "text", None) or result
        cache_set(ckey, val)
        return val
    except Exception as e:
        return f"{type(e).__name__}: {e}"

# ==================== FastAPI surface (unchanged for clients) ====================
app = FastAPI(title="Orchestrator (MCP over stdio)")

class RouteIn(BaseModel):
    query: str
    extra: Optional[Dict[str, Any]] = None
    target: Optional[str] = None
    targets: Optional[List[str]] = None

class RouteOut(BaseModel):
    tool: str = "route_query_tool"
    targets: List[str]
    routed_by: str
    answer: Any
    answers_by_target: Optional[Dict[str, Any]] = None
    errors: Optional[Dict[str, str]] = None

@app.post("/route_query", operation_id="route_query_tool", response_model=RouteOut)
async def route_query(inp: RouteIn = Body(...)):
    q, extra = inp.query, inp.extra
    if inp.targets:
        targets, routed_by = [t for t in inp.targets if t in SERVICES], "hint:list"
    elif inp.target in SERVICES:
        targets, routed_by = [inp.target], "hint:single"
    elif inp.target == "both":
        targets, routed_by = ["hotel", "air"], "hint:both"
    else:
        targets, routed_by = llm_decide_targets(q), "llm"

    results, errors = {}, {}
    if len(targets) <= 1:
        tgt = targets[0] if targets else "hotel"
        results[tgt] = await call_backend(tgt, q, extra)
    else:
        coros = [call_backend(t, q, extra) for t in targets]
        outs = await asyncio.gather(*coros, return_exceptions=True)
        for t, out in zip(targets, outs):
            if isinstance(out, Exception):
                errors[t] = f"{type(out).__name__}: {out}"
            else:
                results[t] = out

    answer = compose_final(q, results) if len(results) > 1 else next(iter(results.values()), "")
    return RouteOut(
        targets=targets, routed_by=routed_by, answer=answer,
        answers_by_target=(results if len(results) > 1 else None),
        errors=(errors or None)
    )

# -------- Optional: tiny agent loop (kept intact) --------
class AgentIn(BaseModel):
    query: str
    extra: Optional[Dict[str, Any]] = None
    session_id: str = "default"
    max_iters: int = 2
    parallel_cap: int = 2
    autonomy: bool = True

class AgentOut(BaseModel):
    tool: str = "agent_query_tool"
    session_id: str
    iterations: int
    final_answer: Any
    observations: List[Dict[str, Any]]
    autonomy: bool

@app.post("/agent_query", operation_id="agent_query_tool", response_model=AgentOut)
async def agent_query(inp: AgentIn = Body(...)):
    q, extra, sid = inp.query, inp.extra, inp.session_id
    mem_append(sid, "user", q)
    observations: List[Dict[str, Any]] = []
    final_answer = None

    async def execute(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        steps = [s for s in steps if s.get("target") in SERVICES][:inp.parallel_cap]
        if not steps: return []
        outs = await asyncio.gather(
            *[call_backend(s["target"], s.get("input", q), extra) for s in steps],
            return_exceptions=True
        )
        batch: List[Dict[str, Any]] = []
        for s, out in zip(steps, outs):
            if isinstance(out, Exception):
                batch.append({"step": s, "error": f"{type(out).__name__}: {out}"})
            else:
                batch.append({"step": s, "result": out})
        return batch

    # one or two simple passes
    for _ in range(max(1, min(inp.max_iters, 6))):
        try:
            plan = claude_json(
                "You plan tool calls. Return ONLY JSON: {\"steps\":[{\"target\":\"air|hotel|tdna\",\"input\":\"...\"}],\"finish\":false,\"final_answer\":\"\"}",
                {"user_query": q, "memory": mem_get(sid, 6), "observations": observations[-2:]},
                max_tokens=256
            )
        except Exception:
            plan = {"steps":[{"target": "hotel", "input": q}], "finish": False, "final_answer": ""}

        if plan.get("finish") and plan.get("final_answer"):
            final_answer = plan["final_answer"]; break

        steps = plan.get("steps", [])
        if not steps: break

        batch = await execute(steps)
        observations.extend(batch)

        try:
            fin = claude_json(
                'If observations+memory satisfy the goal, return {"finish": true, "final_answer": "..."} else {"finish": false}.',
                {"user_query": q, "memory": mem_get(sid, 6), "observations": batch}, max_tokens=196
            )
        except Exception:
            fin = {"finish": False}
        if fin.get("finish"):
            final_answer = fin.get("final_answer"); break

    if final_answer is None:
        latest = {}
        for ob in reversed(observations):
            tgt = ob.get("step",{}).get("target")
            if tgt and "result" in ob and tgt not in latest:
                latest[tgt] = ob["result"]
        final_answer = compose_final(q, latest) if latest else "No result."

    mem_append(sid, "assistant", final_answer)
    return AgentOut(
        session_id=sid,
        iterations=min(len(observations), inp.max_iters),
        final_answer=final_answer,
        observations=observations[-6:],
        autonomy=inp.autonomy
    )

# -------- Memory utils --------
@app.post("/memory/clear")
def memory_clear(payload: Dict[str, Any] = Body(...)):
    sid = payload.get("session_id","default"); mem_clear(sid)
    return JSONResponse({"ok": True, "session_id": sid})

@app.get("/memory/show")
def memory_show(session_id: str = "default"):
    return JSONResponse({"session_id": session_id, "history": mem_get(session_id, 0)})

# ===== Main =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ORCH_PORT)
