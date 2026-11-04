import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import networkx as nx
import numpy as np

# ---------------------------------------------------------
# Global Seed Initialization (SEED=42 reproducibility)
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Chapter 6 Intelligent Agent API",
    description="Lightweight serverless-ready knowledge graph and RAG service without external commercial LLM APIs",
    version="1.0.0",
)

# ---------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Knowledge Base Loader Helpers
# ---------------------------------------------------------
def get_data_dir() -> Path:
    """Resolve data directory path robustly across local and Vercel environments."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    # Fallback to default expected path
    return Path(__file__).resolve().parent.parent / "data"


DATA_DIR = get_data_dir()


def load_knowledge_graph(filename: str) -> Tuple[Dict[str, int], nx.DiGraph, Dict[str, Any]]:
    """Loads a Knowledge Graph JSON file into NetworkX DiGraph and extracts node/edge stats."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return {"nodes": 0, "edges": 0}, nx.DiGraph(), {}

    with open(filepath, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    graph = nx.DiGraph()

    for node in kg_data.get("nodes", []):
        node_id = node.get("id") if isinstance(node, dict) else str(node)
        graph.add_node(node_id, **(node if isinstance(node, dict) else {"id": node}))

    edges = kg_data.get("edges", kg_data.get("triples", []))
    for edge in edges:
        source = edge.get("source", edge.get("subject"))
        target = edge.get("target", edge.get("object"))
        relation = edge.get("relation", edge.get("predicate", ""))
        if source and target:
            graph.add_edge(source, target, relation=relation)

    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }
    return stats, graph, kg_data


def load_rag_chunks(filename: str) -> Tuple[int, List[Dict[str, Any]]]:
    """Loads a RAG chunks JSON file and returns chunk count and raw documents."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return 0, []

    with open(filepath, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    chunks = (
        rag_data
        if isinstance(rag_data, list)
        else rag_data.get("chunks", rag_data.get("documents", []))
    )
    return len(chunks), chunks


# ---------------------------------------------------------
# Load Knowledge Bases on Startup
# ---------------------------------------------------------
diabetes_kg_stats, diabetes_kg_graph, raw_diabetes_kg = load_knowledge_graph("diabetes_kg.json")
stock_kg_stats, stock_kg_graph, raw_stock_kg = load_knowledge_graph("stock_kg.json")

diabetes_rag_count, raw_diabetes_rag = load_rag_chunks("diabetes_rag.json")
stock_rag_count, raw_stock_rag = load_rag_chunks("stock_rag.json")


# ---------------------------------------------------------
# Health Endpoints (Accessible via both /api/health and /health)
# ---------------------------------------------------------
def create_health_payload() -> Dict[str, Any]:
    return {
        "status": "OK",
        "seed": SEED,
        "knowledge_graphs": {
            "diabetes": {
                "nodes": diabetes_kg_stats["nodes"],
                "edges": diabetes_kg_stats["edges"],
            },
            "stock": {
                "nodes": stock_kg_stats["nodes"],
                "edges": stock_kg_stats["edges"],
            },
        },
        "rag_chunks": {
            "diabetes": diabetes_rag_count,
            "stock": stock_rag_count,
        },
        "domains": {
            "diabetes": {
                "kg_nodes": diabetes_kg_stats["nodes"],
                "kg_edges": diabetes_kg_stats["edges"],
                "rag_chunks": diabetes_rag_count,
            },
            "stock": {
                "kg_nodes": stock_kg_stats["nodes"],
                "kg_edges": stock_kg_stats["edges"],
                "rag_chunks": stock_rag_count,
            },
        },
    }


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    """Primary health check endpoint required by the Chapter 6 agent specification."""
    return create_health_payload()


@app.get("/health")
def health() -> Dict[str, Any]:
    """Fallback health route for serverless rewrites stripping /api prefix."""
    return create_health_payload()


# ---------------------------------------------------------
# API Info Route
# ---------------------------------------------------------
@app.get("/api")
def root_info() -> Dict[str, Any]:
    return {
        "message": "Chapter 6 (Intelligent Agent) Foundation API is active",
        "docs_url": "/docs",
        "health_check": "/api/health",
        "seed": SEED,
    }


# ---------------------------------------------------------
# Data Inspection Endpoints
# ---------------------------------------------------------
@app.get("/api/kg/{domain}")
@app.get("/kg/{domain}")
def get_kg(domain: str) -> Dict[str, Any]:
    if domain.lower() == "diabetes":
        return raw_diabetes_kg
    elif domain.lower() == "stock":
        return raw_stock_kg
    return {"error": f"Unknown domain '{domain}'. Supported domains: 'diabetes', 'stock'."}


@app.get("/api/rag/{domain}")
@app.get("/rag/{domain}")
def get_rag(domain: str) -> List[Dict[str, Any]]:
    if domain.lower() == "diabetes":
        return raw_diabetes_rag
    elif domain.lower() == "stock":
        return raw_stock_rag
    return []


from api.agent import IntelligentAgent
from api.benchmark import run_benchmark
from api.memory import AgentMemory, shared_memory
from api.tools import TOOL_REGISTRY

default_agent = IntelligentAgent(memory=shared_memory, tools=TOOL_REGISTRY, max_steps=5)

# ---------------------------------------------------------
# Tool & Memory API Endpoints
# ---------------------------------------------------------
@app.get("/api/tools")
@app.get("/tools")
def get_tools() -> Dict[str, Any]:
    return {"status": "SUCCESS", "tools": list(TOOL_REGISTRY.keys())}


@app.post("/api/tools/{tool_name}")
@app.post("/tools/{tool_name}")
def run_tool(tool_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        return {
            "status": "FAILED",
            "code": "TOOL_NOT_FOUND",
            "result": None,
            "message": f"Tool '{tool_name}' not found in registry.",
        }
    tool_fn = TOOL_REGISTRY[tool_name]
    kwargs = payload or {}
    return tool_fn(**kwargs)


# ---------------------------------------------------------
# Static Files & Frontend SPA Resolution
# ---------------------------------------------------------
def get_public_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "public",
        Path(__file__).resolve().parent / "public",
        Path.cwd() / "public",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return Path(__file__).resolve().parent.parent / "public"


PUBLIC_DIR = get_public_dir()

# ---------------------------------------------------------
# Memory Endpoints
# ---------------------------------------------------------
@app.get("/api/memory")
@app.get("/memory")
def get_memory_state() -> Dict[str, Any]:
    return {
        "status": "SUCCESS",
        "context": shared_memory.get_context(),
        "user_profile": shared_memory.user_profile,
    }


@app.post("/api/memory/reset")
@app.post("/memory/reset")
def reset_memory() -> Dict[str, Any]:
    shared_memory.reset()
    return {
        "status": "SUCCESS",
        "message": "Agent memory reset successfully.",
        "context": shared_memory.get_context(),
    }


@app.post("/api/memory/clear_entity")
@app.post("/memory/clear_entity")
def clear_entity() -> Dict[str, Any]:
    shared_memory.clear_entity()
    return {
        "status": "SUCCESS",
        "message": "Active entity cleared.",
        "context": shared_memory.get_context(),
    }


# ---------------------------------------------------------
# Core Chat & Query Execution Endpoints
# ---------------------------------------------------------
@app.post("/api/query")
@app.post("/query")
@app.post("/api/chat")
@app.post("/chat")
@app.post("/api/agent/run")
@app.post("/agent/run")
def agent_chat(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = (payload or {}).get("query", "")
    if not query:
        return {
            "status": "FAILED",
            "code": "EMPTY_QUERY",
            "result": None,
            "message": "Field 'query' is required.",
        }
    return default_agent.run(query)


@app.get("/api/query")
def agent_query_get(q: str = Query("", description="Natural language user query")) -> Dict[str, Any]:
    if not q:
        return {
            "status": "FAILED",
            "code": "EMPTY_QUERY",
            "result": None,
            "message": "Query parameter 'q' is required.",
        }
    return default_agent.run(q)


# ---------------------------------------------------------
# Benchmark Endpoint
# ---------------------------------------------------------
@app.get("/api/benchmark")
@app.get("/benchmark")
def get_benchmark() -> Dict[str, Any]:
    """Executes Chapter 6 benchmark evaluating 3 system configurations across 10 multi-step test queries."""
    return run_benchmark()


# ---------------------------------------------------------
# Frontend SPA & Static Asset Handlers
# ---------------------------------------------------------
@app.get("/")
def serve_index() -> Any:
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Chapter 6 (Intelligent Agent) Foundation API is active",
        "docs_url": "/docs",
        "health_check": "/api/health",
        "benchmark": "/api/benchmark",
    }


@app.get("/{filename:path}")
def serve_static_or_spa(filename: str) -> Any:
    # Do not intercept API routes
    if filename.startswith("api/") or filename.startswith("docs") or filename.startswith("openapi"):
        return {"error": "API route not found."}
    target = PUBLIC_DIR / filename
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"error": f"File '{filename}' not found."}


# Entrypoint for standalone execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
