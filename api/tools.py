import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from api.memory import AgentMemory, shared_memory
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Global Seed & Path Resolution
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_data_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return Path(__file__).resolve().parent.parent / "data"


DATA_DIR = get_data_dir()

# Cache for Knowledge Graphs and RAG corpora
_KG_CACHE: Dict[str, nx.DiGraph] = {}
_RAG_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# ---------------------------------------------------------
# Trained Logistic Regression for Medical Risk Scoring
# ---------------------------------------------------------
def _train_synthetic_diabetes_model() -> LogisticRegression:
    """Pre-trains a reproducible scikit-learn LogisticRegression model for diabetes risk."""
    rng = np.random.RandomState(SEED)
    n_samples = 400

    # Synthetic realistic feature distributions:
    # [glucose (mg/dL), bmi, hba1c (%), age]
    glucose = rng.normal(110, 25, n_samples).clip(70, 250)
    bmi = rng.normal(27, 4, n_samples).clip(18, 45)
    hba1c = rng.normal(5.8, 1.1, n_samples).clip(4.5, 12.0)
    age = rng.normal(48, 12, n_samples).clip(20, 80)

    X = np.column_stack([glucose, bmi, hba1c, age])

    # Diagnostic rule-of-thumb: high risk if glucose >= 126 or hba1c >= 6.5 with noise
    logits = (
        -4.5
        + 0.032 * (glucose - 100)
        + 0.065 * (bmi - 25)
        + 1.85 * (hba1c - 5.7)
        + 0.028 * (age - 45)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    model = LogisticRegression(random_state=SEED, max_iter=200)
    model.fit(X, y)
    return model


_DIABETES_ML_MODEL = _train_synthetic_diabetes_model()


# ---------------------------------------------------------
# Helper: Domain & KG Loading
# ---------------------------------------------------------
def _resolve_domain_file(domain: str, category: str) -> Path:
    dom = (domain or "").strip().lower()
    if dom in ["medical", "diabetes", "health"]:
        filename = f"diabetes_{category}.json"
    elif dom in ["stock", "stocks", "finance", "vn_stock"]:
        filename = f"stock_{category}.json"
    else:
        filename = f"{dom}_{category}.json"
    return DATA_DIR / filename


def _get_or_load_kg(domain: str) -> nx.DiGraph:
    dom_key = (domain or "").strip().lower()
    if dom_key in _KG_CACHE:
        return _KG_CACHE[dom_key]

    filepath = _resolve_domain_file(domain, "kg")
    graph = nx.DiGraph()

    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            nid = node.get("id") if isinstance(node, dict) else str(node)
            graph.add_node(nid, **(node if isinstance(node, dict) else {"id": nid}))

        edges = data.get("edges", data.get("triples", []))
        for edge in edges:
            src = edge.get("source", edge.get("subject"))
            tgt = edge.get("target", edge.get("object"))
            rel = edge.get("relation", edge.get("predicate", ""))
            if src and tgt:
                graph.add_edge(src, tgt, relation=rel)

    _KG_CACHE[dom_key] = graph
    return graph


def _get_or_load_rag(domain: str) -> List[Dict[str, Any]]:
    dom_key = (domain or "").strip().lower()
    if dom_key in _RAG_CACHE:
        return _RAG_CACHE[dom_key]

    filepath = _resolve_domain_file(domain, "rag")
    chunks: List[Dict[str, Any]] = []

    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
            chunks = raw if isinstance(raw, list) else raw.get("chunks", raw.get("documents", []))

    _RAG_CACHE[dom_key] = chunks
    return chunks


# =========================================================
# TOOL 1: safety_check
# =========================================================
def safety_check(query: str = "", domain: str = "", **kwargs: Any) -> Dict[str, Any]:
    """Evaluates query safety guardrails. Blocks unverified prescriptions/dosages in medical

    and guaranteed return/insider tipping in financial queries.
    """
    q_norm = (query or "").lower().strip()
    dom = (domain or "").lower().strip()

    medical_refusal_patterns = [
        "kê đơn",
        "đơn thuốc",
        "liều dùng",
        "uống thuốc",
        "prescription",
        "prescribe",
        "dosage",
    ]

    stock_refusal_patterns = [
        "chắc chắn lãi",
        "cam kết",
        "tất tay",
        "phím hàng",
        "chắc chắn thắng",
        "đội lái",
        "all-in",
    ]

    # Check medical safety
    for pattern in medical_refusal_patterns:
        if pattern in q_norm:
            return {
                "status": "BLOCKED",
                "code": "REFUSED_SAFETY",
                "result": None,
                "message": (
                    f"Kích hoạt Chốt chặn An toàn: Câu hỏi y tế yêu cầu kê đơn hoặc xác định liều lượng thuốc "
                    f"('{pattern}'). Hệ thống tự động bắt buộc từ chối cung cấp đơn thuốc khi chưa có chỉ định lâm sàng trực tiếp từ bác sĩ."
                ),
            }

    # Check financial safety
    for pattern in stock_refusal_patterns:
        if pattern in q_norm:
            return {
                "status": "BLOCKED",
                "code": "REFUSED_SAFETY",
                "result": None,
                "message": (
                    f"Kích hoạt Chốt chặn An toàn: Câu hỏi tài chính yêu cầu cam kết lợi nhuận, phím hàng nội gián hoặc lời khuyên tất tay "
                    f"('{pattern}'). Quy định pháp luật nghiêm cấm mọi hình thức cam kết lợi nhuận chắc chắn trên thị trường chứng khoán."
                ),
            }

    return {
        "status": "SUCCESS",
        "code": "PASSED",
        "result": {"passed": True, "query": query, "domain": domain},
        "message": "Câu hỏi đã vượt qua kiểm tra an toàn thành công.",
    }


# =========================================================
# TOOL 2: kg_query
# =========================================================
def kg_query(
    entity: str = "", domain: str = "", max_hops: int = 1, **kwargs: Any
) -> Dict[str, Any]:
    """Queries domain Knowledge Graph for structural 1-hop or 2-hop triples around an entity."""
    if not entity:
        return {
            "status": "FAILED",
            "code": "ENTITY_NOT_FOUND",
            "result": [],
            "message": "Entity parameter cannot be empty.",
        }

    graph = _get_or_load_kg(domain)
    if graph.number_of_nodes() == 0:
        return {
            "status": "FAILED",
            "code": "ENTITY_NOT_FOUND",
            "result": [],
            "message": f"Knowledge Graph for domain '{domain}' is empty or unavailable.",
        }

    # Fuzzy/normalized node matching
    target_node = None
    ent_norm = entity.strip().lower()
    ent_alt = ent_norm.replace(" ", "_")

    for node in graph.nodes:
        n_str = str(node).lower()
        if ent_norm == n_str or ent_alt == n_str:
            target_node = node
            break
        # Also check label attribute
        lbl = str(graph.nodes[node].get("label", "")).lower()
        if ent_norm == lbl or ent_norm in lbl:
            target_node = node
            break

    if target_node is None:
        return {
            "status": "FAILED",
            "code": "ENTITY_NOT_FOUND",
            "result": [],
            "message": f"Không tìm thấy thực thể '{entity}' trong đồ thị tri thức miền '{domain}'.",
        }

    # Extract 1-hop or 2-hop triples
    visited_nodes: Set[Any] = {target_node}
    current_frontier: Set[Any] = {target_node}
    collected_triples: List[Dict[str, Any]] = []
    seen_triples: Set[Tuple[str, str, str]] = set()

    for _ in range(max(1, min(max_hops, 2))):
        next_frontier: Set[Any] = set()
        for curr in current_frontier:
            # Outgoing edges
            for _, tgt, data in graph.out_edges(curr, data=True):
                rel = data.get("relation", "RELATED_TO")
                trip_key = (str(curr), str(rel), str(tgt))
                if trip_key not in seen_triples:
                    seen_triples.add(trip_key)
                    collected_triples.append(
                        {
                            "subject": str(curr),
                            "predicate": str(rel),
                            "object": str(tgt),
                            "source": str(curr),
                            "relation": str(rel),
                            "target": str(tgt),
                        }
                    )
                if tgt not in visited_nodes:
                    visited_nodes.add(tgt)
                    next_frontier.add(tgt)

            # Incoming edges
            for src, _, data in graph.in_edges(curr, data=True):
                rel = data.get("relation", "RELATED_TO")
                trip_key = (str(src), str(rel), str(curr))
                if trip_key not in seen_triples:
                    seen_triples.add(trip_key)
                    collected_triples.append(
                        {
                            "subject": str(src),
                            "predicate": str(rel),
                            "object": str(curr),
                            "source": str(src),
                            "relation": str(rel),
                            "target": str(curr),
                        }
                    )
                if src not in visited_nodes:
                    visited_nodes.add(src)
                    next_frontier.add(src)

        current_frontier = next_frontier

    return {
        "status": "SUCCESS",
        "code": "PASSED",
        "result": collected_triples,
        "message": f"Đã truy xuất {len(collected_triples)} bộ ba tri thức cho thực thể '{target_node}' (max_hops={max_hops}).",
    }


# =========================================================
# TOOL 3: rag_retrieve
# =========================================================
def rag_retrieve(
    query: str = "", domain: str = "", top_k: int = 2, **kwargs: Any
) -> Dict[str, Any]:
    """Retrieves relevant RAG chunks using TF-IDF cosine similarity. Enforces tau=0.12 knowledge gap threshold."""
    if not query or not query.strip():
        return {
            "status": "FAILED",
            "code": "ABSTAIN_KNOWLEDGE_GAP",
            "result": [],
            "message": "Tham số truy vấn không được để trống.",
        }

    chunks = _get_or_load_rag(domain)
    if not chunks:
        return {
            "status": "FAILED",
            "code": "ABSTAIN_KNOWLEDGE_GAP",
            "result": [],
            "message": f"Không có tài liệu quy chuẩn RAG cho miền '{domain}'.",
        }

    corpus = [chunk.get("text", chunk.get("content", "")) for chunk in chunks]

    vn_stopwords = [
        "và", "của", "là", "có", "các", "những", "cho", "với", "trong", "được",
        "này", "đó", "ở", "đến", "từ", "theo", "như", "về", "đạt", "khi", "trên",
        "nào", "gì", "sao", "bao", "nhiêu", "bị", "do", "đã", "sẽ", "đang", "ra",
        "vào", "lại", "thì", "mà", "tỷ", "lệ", "báo", "cáo",
    ]

    try:
        vectorizer = TfidfVectorizer(stop_words=vn_stopwords)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    except Exception as e:
        return {
            "status": "FAILED",
            "code": "ABSTAIN_KNOWLEDGE_GAP",
            "result": [],
            "message": f"Lỗi trong quá trình vector hóa TF-IDF: {str(e)}",
        }

    max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0
    TAU_THRESHOLD = 0.12

    # Chapter 6 Knowledge Gap Refusal
    if max_sim < TAU_THRESHOLD:
        return {
            "status": "FAILED",
            "code": "ABSTAIN_KNOWLEDGE_GAP",
            "result": [],
            "message": (
                f"Phát hiện Khoảng trống Tri thức: Độ tương đồng cosine TF-IDF cao nhất ({max_sim:.4f}) nằm dưới "
                f"ngưỡng kiềm chế tau={TAU_THRESHOLD}. Từ chối suy đoán theo nguyên tắc kiểm soát ảo giác."
            ),
        }

    ranked_indices = np.argsort(similarities)[::-1]
    top_chunks: List[Dict[str, Any]] = []

    for idx in ranked_indices[: max(1, top_k)]:
        score = float(similarities[idx])
        chunk_entry = dict(chunks[idx])
        chunk_entry["similarity_score"] = round(score, 4)
        top_chunks.append(chunk_entry)

    return {
        "status": "SUCCESS",
        "code": "PASSED",
        "result": top_chunks,
        "message": f"Truy xuất thành công {len(top_chunks)} đoạn tài liệu quy chuẩn (độ tương đồng cao nhất: {max_sim:.4f}).",
    }


# =========================================================
# TOOL 4: ml_predict
# =========================================================
def ml_predict(
    domain: str = "", features: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Dict[str, Any]:
    """Executes local machine learning surrogate models without external LLMs.

    - Medical: Logistic risk scoring based on glucose, bmi, hba1c, age.
    - Stock: Momentum-based surrogate forecasting next closing price for ticker.
    """
    dom = (domain or "").strip().lower()
    feats = features if features is not None else kwargs

    # Medical / Diabetes Domain
    if dom in ["medical", "diabetes", "health"]:
        glucose = float(feats.get("glucose") or feats.get("blood_glucose") or 100.0)
        bmi = float(feats.get("bmi") or shared_memory.get("default_bmi", 26.5))
        hba1c = float(feats.get("hba1c") or feats.get("hba1c_level") or 5.4)
        age = float(feats.get("age") or shared_memory.get("default_age", 45))

        # Evaluate via pre-trained Logistic Regression
        input_vector = np.array([[glucose, bmi, hba1c, age]])
        prob = float(_DIABETES_ML_MODEL.predict_proba(input_vector)[0][1])

        if prob < 0.35:
            risk_tier = "Nguy cơ Thấp (Low Risk)"
        elif prob < 0.65:
            risk_tier = "Nguy cơ Trung bình (Moderate Risk)"
        else:
            risk_tier = "Nguy cơ Cao (High Risk)"

        result_payload = {
            "domain": "diabetes",
            "diabetes_risk_score": round(prob, 4),
            "risk_percentage": f"{prob * 100:.1f}%",
            "risk_level": risk_tier,
            "predicted_class": 1 if prob >= 0.5 else 0,
            "features_evaluated": {
                "glucose": glucose,
                "bmi": bmi,
                "hba1c": hba1c,
                "age": age,
            },
            "clinical_thresholds": {
                "fpg_threshold": ">= 126 mg/dL",
                "hba1c_threshold": ">= 6.5%",
            },
        }
        return {
            "status": "SUCCESS",
            "code": "PASSED",
            "result": result_payload,
            "message": f"Tính toán rủi ro y khoa: {risk_tier} (xác suất {prob * 100:.1f}%).",
        }

    # Stock / Finance Domain
    elif dom in ["stock", "stocks", "finance", "vn_stock"]:
        ticker = str(
            feats.get("ticker")
            or feats.get("symbol")
            or shared_memory.get("active_entity")
            or "FPT"
        ).upper()

        current_price = float(
            feats.get("current_price") or feats.get("close_price") or 135000.0
        )
        momentum = float(feats.get("momentum") or feats.get("momentum_5d") or 0.015)
        roe = float(feats.get("roe") or 0.261)
        de_ratio = float(feats.get("de_ratio") or 1.01)

        # Momentum surrogate forecasting model bounded by HOSE +/- 7% limit
        raw_delta = 0.60 * momentum + 0.25 * (roe - 0.15) - 0.15 * (de_ratio - 1.0)
        # Clamped strictly inside HOSE price fluctuation band
        clamped_delta = max(-0.07, min(0.07, raw_delta))

        # Tick-size rounding (HOSE standard: 100 VND)
        predicted_close = round(current_price * (1.0 + clamped_delta), -2)
        hose_ceiling = round(current_price * 1.07, -2)
        hose_floor = round(current_price * 0.93, -2)

        if clamped_delta > 0.005:
            trend = "TĂNG GIÁ (BULLISH)"
        elif clamped_delta < -0.005:
            trend = "GIẢM GIÁ (BEARISH)"
        else:
            trend = "TRUNG LẬP (NEUTRAL)"

        stock_payload = {
            "domain": "stock",
            "ticker": ticker,
            "current_price": current_price,
            "predicted_close": predicted_close,
            "expected_return_pct": f"{clamped_delta * 100:+.2f}%",
            "trend": trend,
            "hose_ceiling_price": hose_ceiling,
            "hose_floor_price": hose_floor,
            "features_used": {
                "momentum": momentum,
                "roe": roe,
                "de_ratio": de_ratio,
            },
        }
        return {
            "status": "SUCCESS",
            "code": "PASSED",
            "result": stock_payload,
            "message": f"Dự báo xu hướng cổ phiếu {ticker}: {trend} ({clamped_delta * 100:+.2f}% đạt {predicted_close:,.0f} VND).",
        }

    return {
        "status": "FAILED",
        "code": "UNSUPPORTED_DOMAIN",
        "result": None,
        "message": f"ML prediction domain '{domain}' is unsupported. Available domains: 'diabetes', 'stock'.",
    }


# =========================================================
# TOOL 5: memory_tool
# =========================================================
def memory_tool(
    action: str = "get", key: str = "", value: Any = None, **kwargs: Any
) -> Dict[str, Any]:
    """Executes stateful operations (get or set) on the shared AgentMemory."""
    act = (action or "").strip().lower()

    if act == "get":
        if not key:
            return {
                "status": "FAILED",
                "code": "MISSING_KEY",
                "result": None,
                "message": "Key parameter is required for memory get.",
            }
        val = shared_memory.get(key)
        return {
            "status": "SUCCESS",
            "code": "PASSED",
            "result": val,
            "message": f"Truy xuất thành công khóa bộ nhớ '{key}'.",
        }

    elif act == "set":
        if not key:
            return {
                "status": "FAILED",
                "code": "MISSING_KEY",
                "result": None,
                "message": "Cần cung cấp tham số khóa (key) cho thao tác ghi nhớ.",
            }
        shared_memory.set(key, value)
        return {
            "status": "SUCCESS",
            "code": "PASSED",
            "result": value,
            "message": f"Cập nhật thành công khóa bộ nhớ '{key}'.",
        }

    elif act == "context":
        ctx = shared_memory.get_context()
        return {
            "status": "SUCCESS",
            "code": "PASSED",
            "result": ctx,
            "message": "Truy xuất thành công ngữ cảnh bộ nhớ.",
        }

    return {
        "status": "FAILED",
        "code": "INVALID_ACTION",
        "result": None,
        "message": f"Unsupported memory action '{action}'. Permitted actions: 'get', 'set', 'context'.",
    }


# =========================================================
# Unified Tool Registry Export
# =========================================================
TOOL_REGISTRY: Dict[str, Any] = {
    "safety_check": safety_check,
    "kg_query": kg_query,
    "rag_retrieve": rag_retrieve,
    "ml_predict": ml_predict,
    "memory_tool": memory_tool,
}
