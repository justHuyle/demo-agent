import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from api.memory import AgentMemory, shared_memory
from api.tools import TOOL_REGISTRY


class IntelligentAgent:
    """Chapter 6 Intelligent Agent architecture with state equation logging,

    adaptive planning, guardrails, and grounded synthesis without external commercial LLMs.
    """

    def __init__(
        self,
        memory: Optional[AgentMemory] = None,
        tools: Optional[Dict[str, Any]] = None,
        max_steps: int = 5,
    ) -> None:
        self.memory = memory if memory is not None else shared_memory
        self.tools = tools if tools is not None else TOOL_REGISTRY
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Domain & Entity Recognition + Pronoun & Topic Shift Resolution
    # ------------------------------------------------------------------
    def _detect_domain(self, text: str) -> Optional[str]:
        """Detects domain ('medical' or 'stock') from query keyword matching."""
        t_low = text.lower()

        medical_keywords = [
            "hba1c", "tiểu đường", "đái tháo đường", "glucose", "đường huyết",
            "insulin", "metformin", "huyết áp", "retinopathy", "nephropathy",
            "neuropathy", "bmi", "fpg", "ogtt", "sglt2", "nguy cơ", "sàng lọc",
            "kháng insulin", "béo phì", "dsmcs", "mnt", "kê đơn", "đơn thuốc",
            "liều dùng", "uống thuốc"
        ]

        stock_keywords = [
            "fpt", "vnm", "vinamilk", "hpg", "hòa phát", "gas", "cổ phiếu",
            "doanh thu", "lợi nhuận", "p/e", "roe", "d/e", "vốn hóa", "hose",
            "vn30", "chứng khoán", "thị trường", "giá đóng cửa", "thanh khoản",
            "giao dịch", "dự báo giá", "biên độ", "phím hàng", "tất tay",
            "chắc chắn lãi", "mã", "đầu tư"
        ]

        med_score = sum(1 for kw in medical_keywords if kw in t_low)
        stk_score = sum(1 for kw in stock_keywords if kw in t_low)

        if med_score > stk_score and med_score > 0:
            return "medical"
        elif stk_score > med_score and stk_score > 0:
            return "stock"
        return None

    def _detect_explicit_entity(self, text: str, domain: Optional[str]) -> Optional[str]:
        """Detects explicitly mentioned entity from domain knowledge entities."""
        t_low = text.lower()

        # Stock Entities
        stock_entities = {
            "FPT": ["fpt"],
            "VNM": ["vnm", "vinamilk"],
            "HPG": ["hpg", "hòa phát", "hoa phat"],
            "GAS": ["gas", "pv gas", "petrovietnam gas"],
            "HOSE_Exchange": ["sàn hose", "sở giao dịch chứng khoán tp.hcm", "hose"],
            "VN30_Index": ["vn30", "chỉ số vn30"],
            "ROE_LTM": ["roe", "roe ltm"],
            "DE_Ratio": ["d/e", "đòn bẩy", "nợ trên vốn"],
            "PE_Ratio": ["p/e", "hệ số p/e"],
            "Market_Cap": ["vốn hóa"],
            "IT_Sector": ["ngành it", "ngành công nghệ thông tin", "cntt"],
        }

        # Medical Entities
        medical_entities = {
            "HbA1c_Level": ["hba1c", "a1c"],
            "Blood_Glucose_Level": ["đường huyết", "blood glucose"],
            "Fasting_Plasma_Glucose": ["fpg", "glucose đói", "đường huyết lúc đói"],
            "Type_2_Diabetes": ["tiểu đường", "đái tháo đường", "type 2 diabetes", "t2d"],
            "Metformin": ["metformin"],
            "SGLT2_Inhibitor": ["sglt2", "ức chế sglt2"],
            "Hypertension": ["huyết áp", "tăng huyết áp", "hypertension"],
            "Retinopathy": ["võng mạc", "retinopathy"],
            "Nephropathy": ["thận", "nephropathy"],
            "Neuropathy": ["thần kinh", "neuropathy"],
            "Cardiovascular_Disease": ["tim mạch", "cardiovascular"],
            "Obesity": ["béo phì", "obesity"],
            "Insulin_Resistance": ["kháng insulin"],
            "Lifestyle_Modification": ["lối sống", "vận động", "tập luyện"],
        }

        search_dict = {}
        if domain == "stock":
            search_dict.update(stock_entities)
        elif domain == "medical":
            search_dict.update(medical_entities)
        else:
            search_dict.update(stock_entities)
            search_dict.update(medical_entities)

        for ent_id, triggers in search_dict.items():
            for trig in triggers:
                # Word boundary check for short tickers like FPT, GAS
                pattern = r"\b" + re.escape(trig) + r"\b" if len(trig) <= 4 else re.escape(trig)
                if re.search(pattern, t_low):
                    return ent_id

        return None

    def _resolve_pronouns_and_topic_shift(
        self, query: str
    ) -> Tuple[str, str, Optional[str], bool]:
        """Resolves ellipsis/pronouns or clears active entity upon topic shift (Failure Mode 4).

        Returns: (resolved_query, domain, entity, topic_shifted)
        """
        domain_prev = self.memory.short_term.get("active_domain")
        entity_prev = self.memory.short_term.get("active_entity")

        query_domain = self._detect_domain(query)
        effective_domain = query_domain or domain_prev or "stock"

        explicit_entity = self._detect_explicit_entity(query, effective_domain)

        pronouns = [
            "nó", "chỉ số đó", "chỉ số này", "doanh nghiệp này", "doanh nghiệp đó",
            "công ty này", "công ty đó", "mã này", "mã đó", "bệnh này", "căn bệnh này"
        ]
        has_pronoun = any(p in query.lower() for p in pronouns)

        topic_shifted = False

        # Condition 1: Cross-domain topic shift
        if query_domain and domain_prev and query_domain != domain_prev:
            topic_shifted = True

        # Condition 2: Explicit entity switch without pronouns
        if explicit_entity and entity_prev and explicit_entity != entity_prev and not has_pronoun:
            topic_shifted = True

        # Condition 3: Neither explicit entity nor pronoun, moving to a new topic without referencing previous entity
        if not explicit_entity and not has_pronoun and entity_prev:
            topic_shifted = True

        if topic_shifted:
            self.memory.clear_entity()
            active_entity = explicit_entity
        else:
            if has_pronoun and entity_prev:
                active_entity = entity_prev
            elif explicit_entity:
                active_entity = explicit_entity
            else:
                active_entity = None

        # Ellipsis substitution for text query resolution
        resolved_query = query
        if has_pronoun and active_entity:
            for p in pronouns:
                if p in resolved_query.lower():
                    # Replace primary pronoun with active entity
                    resolved_query = re.sub(re.escape(p), active_entity, resolved_query, count=1, flags=re.IGNORECASE)
                    break

        return resolved_query, effective_domain, active_entity, topic_shifted

    # ------------------------------------------------------------------
    # Feature Extraction for ML Prediction
    # ------------------------------------------------------------------
    def _extract_ml_features(self, query: str, domain: str, entity: Optional[str]) -> Dict[str, Any]:
        """Extracts numerical features from query text or user profile defaults."""
        user_prof = self.memory.long_term.get("user_profile", {})

        if domain == "medical":
            # Extract glucose
            glucose_m = re.search(r"(?:glucose|đường huyết|fpg)\s*(?:là|:|=)?\s*(\d+(?:\.\d+)?)", query, re.I)
            glucose = float(glucose_m.group(1)) if glucose_m else 105.0

            # Extract bmi
            bmi_m = re.search(r"(?:bmi)\s*(?:là|:|=)?\s*(\d+(?:\.\d+)?)", query, re.I)
            bmi = float(bmi_m.group(1)) if bmi_m else float(user_prof.get("default_bmi", 26.5))

            # Extract hba1c
            hba1c_m = re.search(r"(?:hba1c|a1c)\s*(?:là|:|=)?\s*(\d+(?:\.\d+)?)", query, re.I)
            hba1c = float(hba1c_m.group(1)) if hba1c_m else 5.6

            # Extract age
            age_m = re.search(r"(?:tuổi|age)\s*(?:là|:|=)?\s*(\d+)", query, re.I)
            age = float(age_m.group(1)) if age_m else float(user_prof.get("default_age", 45))

            return {"glucose": glucose, "bmi": bmi, "hba1c": hba1c, "age": age}

        elif domain == "stock":
            ticker = entity if entity in ["FPT", "VNM", "HPG", "GAS"] else "FPT"
            price_m = re.search(r"(?:giá|price)\s*(?:là|:|=)?\s*(\d+[\d\.,]*)", query, re.I)
            current_price = 135000.0
            if price_m:
                raw_p = price_m.group(1).replace(".", "").replace(",", "")
                if raw_p.isdigit() and float(raw_p) > 1000:
                    current_price = float(raw_p)

            return {
                "ticker": ticker,
                "current_price": current_price,
                "momentum": 0.015,
                "roe": 0.261,
                "de_ratio": 1.01,
            }

        return {}

    # ------------------------------------------------------------------
    # Grounded Answer Synthesis
    # ------------------------------------------------------------------
    def _synthesize_grounded_answer(
        self,
        query: str,
        domain: str,
        entity: Optional[str],
        kg_facts: List[Dict[str, Any]],
        rag_chunks: List[Dict[str, Any]],
        ml_output: Optional[Dict[str, Any]],
    ) -> str:
        """Assembles a grounded factual response citing ML outputs, KG triples, and RAG sources."""
        sections = []

        if domain == "medical":
            sections.append("### Kết quả Phân tích Tri thức Y khoa (ADA 2026)")
        else:
            sections.append("### Kết quả Phân tích Đầu tư Chứng khoán (HOSE)")

        # 1. ML Output Citation
        if ml_output:
            if domain == "medical":
                sections.append(
                    f"**1. Mô hình Dự báo Rủi ro Sàng lọc (ML Logistic):**\n"
                    f"- Xác suất nguy cơ: **{ml_output.get('risk_percentage')}** ({ml_output.get('risk_level')}).\n"
                    f"- Yếu tố đánh giá: Nồng độ Glucose={ml_output['features_evaluated']['glucose']} mg/dL, "
                    f"HbA1c={ml_output['features_evaluated']['hba1c']}%, "
                    f"Chỉ số BMI={ml_output['features_evaluated']['bmi']}, "
                    f"Tuổi={ml_output['features_evaluated']['age']}."
                )
            else:
                sections.append(
                    f"**1. Mô hình Dự báo Xu hướng Giá (ML Momentum Surrogate):**\n"
                    f"- Mã cổ phiếu: **{ml_output.get('ticker')}** | Tín hiệu: **{ml_output.get('trend')}**\n"
                    f"- Dự báo giá đóng cửa phiên kế tiếp: **{ml_output.get('predicted_close'):,.0f} VND** "
                    f"({ml_output.get('expected_return_pct')}).\n"
                    f"- Biên độ dao động HOSE (+/- 7%): Giá Sàn {ml_output.get('hose_floor_price'):,.0f} VND - "
                    f"Giá Trần {ml_output.get('hose_ceiling_price'):,.0f} VND."
                )

        # 2. Knowledge Graph Triples Citation
        if kg_facts:
            RELATION_VI = {
                "DIAGNOSES": "chẩn đoán xác định",
                "RISK_FACTOR_FOR": "là yếu tố nguy cơ của",
                "COMPLICATION_OF": "là biến chứng của",
                "TREATS": "điều trị",
                "FIRST_LINE_DRUG_FOR": "là thuốc chỉ định đầu tay cho",
                "COMORBIDITY_WITH": "đồng mắc với",
                "INTERVENTION_FOR": "can thiệp hỗ trợ cho",
                "BIOMARKER_FOR": "dấu ấn sinh học của",
                "LISTED_ON": "được niêm yết tại",
                "BELONGS_TO_SECTOR": "thuộc nhóm ngành",
                "TRACKED_BY_METRIC": "được theo dõi qua chỉ số",
                "INCLUDED_IN_INDEX": "nằm trong rổ chỉ số",
                "REGULATED_BY": "chịu sự quản lý của",
                "METRIC_FOR": "chỉ số đánh giá của",
                "COMPETITOR_OF": "đối thủ cạnh tranh với",
            }
            ENTITY_VI = {
                "HbA1c_Level": "Chỉ số HbA1c",
                "Blood_Glucose_Level": "Nồng độ Đường huyết",
                "Fasting_Plasma_Glucose": "Đường huyết lúc đói (FPG)",
                "Type_2_Diabetes": "Bệnh Đái tháo đường Type 2",
                "Metformin": "Thuốc Metformin",
                "SGLT2_Inhibitor": "Thuốc ức chế SGLT2",
                "Hypertension": "Tăng huyết áp",
                "Retinopathy": "Bệnh võng mạc tiểu đường",
                "Nephropathy": "Bệnh thận tiểu đường",
                "Neuropathy": "Bệnh thần kinh tiểu đường",
                "Cardiovascular_Disease": "Bệnh tim mạch (CVD)",
                "Obesity": "Béo phì",
                "Physical_Inactivity": "Lối sống ít vận động",
                "Insulin_Resistance": "Đề kháng Insulin",
                "Lifestyle_Modification": "Can thiệp lối sống",
                "FPT": "Cổ phiếu FPT",
                "VNM": "Cổ phiếu Vinamilk (VNM)",
                "HPG": "Cổ phiếu Hòa Phát (HPG)",
                "GAS": "Cổ phiếu PV Gas (GAS)",
                "HOSE_Exchange": "Sở Giao dịch Chứng khoán TP.HCM (HOSE)",
                "VN30_Index": "Chỉ số VN30",
                "IT_Sector": "Ngành Công nghệ Thông tin",
                "ROE_LTM": "Chỉ số ROE LTM (26.1%)",
                "DE_Ratio": "Tỷ lệ Nợ trên Vốn D/E (1.01x)",
                "PE_Ratio": "Hệ số Định giá P/E",
                "Market_Cap": "Vốn hóa Thị trường",
                "State_Securities_Commission": "Ủy ban Chứng khoán Nhà nước",
                "VSDC": "Tổng công ty Lưu ký và Bù trừ Chứng khoán",
            }
            triple_lines = []
            for t in kg_facts[:5]:
                sub_vi = ENTITY_VI.get(t['subject'], t['subject'])
                rel_vi = RELATION_VI.get(t['predicate'], t['predicate'])
                obj_vi = ENTITY_VI.get(t['object'], t['object'])
                triple_lines.append(f"  - **{sub_vi}** (`{t['subject']}`) --[_{rel_vi}_]--> **{obj_vi}** (`{t['object']}`)")
            sections.append(
                f"**2. Tri thức Cấu trúc Định danh (Knowledge Graph):**\n"
                + "\n".join(triple_lines)
            )

        # 3. RAG Documents Citation
        if rag_chunks:
            rag_lines = []
            for c in rag_chunks[:2]:
                title = c.get("title", "Tài liệu quy chuẩn")
                src = c.get("source", "Cơ sở dữ liệu ADA/HOSE")
                text = c.get("text", c.get("content", ""))
                sim = c.get("similarity_score", 0.0)
                rag_lines.append(f"- **{title}** (Nguồn: _{src}_ | Độ khớp TF-IDF: {sim:.4f}):\n  > {text}")
            sections.append(
                f"**3. Trích xuất Tài liệu Quy chuẩn (RAG Retrieval):**\n"
                + "\n\n".join(rag_lines)
            )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Core Loop: Observe -> Reason -> Plan -> Act -> Observe
    # ------------------------------------------------------------------
    def run(self, query: str) -> Dict[str, Any]:
        """Executes the Chapter 6 Intelligent Agent core loop with state equation logging."""
        resolved_query, domain, active_entity, topic_shifted = self._resolve_pronouns_and_topic_shift(query)

        # Set active context in memory
        self.memory.short_term["active_domain"] = domain
        self.memory.short_term["active_entity"] = active_entity

        trace: Dict[str, Any] = {
            "query": query,
            "resolved_query": resolved_query,
            "domain": domain,
            "active_entity": active_entity,
            "topic_shifted": topic_shifted,
            "status": "RUNNING",
            "steps": [],
            "context": {
                "kg_facts": [],
                "rag_chunks": [],
                "ml_output": None,
            },
            "total_steps": 0,
            "final_answer": "",
        }

        accumulated_kg: List[Dict[str, Any]] = []
        accumulated_rag: List[Dict[str, Any]] = []
        accumulated_ml: Optional[Dict[str, Any]] = None

        seen_actions: Set[Tuple[str, str]] = set()

        kg_done = False
        rag_done = False
        ml_done = False
        knowledge_gap_detected = False

        step_idx = 1
        current_status = "RUNNING"
        current_observation: Any = {"raw_query": query, "resolved_query": resolved_query}

        while step_idx <= self.max_steps:
            remaining_steps = self.max_steps - step_idx + 1

            # s(t): Current internal state
            s_t = {
                "step": step_idx,
                "active_domain": domain,
                "active_entity": active_entity,
                "remaining_steps": remaining_steps,
                "status": current_status,
            }

            # x(t): Current observation
            x_t = current_observation

            # C(t): Accumulated context
            C_t = {
                "kg_facts": copy.deepcopy(accumulated_kg),
                "rag_chunks": copy.deepcopy(accumulated_rag),
                "ml_output": copy.deepcopy(accumulated_ml),
            }

            # ---------------------------------------------------------
            # Action Selection Policy pi(s(t), x(t), C(t))
            # ---------------------------------------------------------
            tool_name = ""
            action_kwargs: Dict[str, Any] = {}

            # STEP 1: Mandatory Guardrail Check
            if step_idx == 1:
                tool_name = "safety_check"
                action_kwargs = {"query": query, "domain": domain}

            # SUBSEQUENT STEPS: Adaptive Planning
            else:
                pred_keywords = ["dự báo", "nguy cơ", "sàng lọc", "dự đoán", "tiên lượng", "predict", "forecast", "risk"]
                has_pred = any(kw in query.lower() for kw in pred_keywords)

                if has_pred and not ml_done:
                    tool_name = "ml_predict"
                    feats = self._extract_ml_features(query, domain, active_entity)
                    action_kwargs = {"domain": domain, "features": feats}
                    ml_done = True

                elif active_entity and not kg_done:
                    tool_name = "kg_query"
                    action_kwargs = {"entity": active_entity, "domain": domain, "max_hops": 1}
                    kg_done = True

                elif not rag_done:
                    tool_name = "rag_retrieve"
                    action_kwargs = {"query": resolved_query, "domain": domain, "top_k": 2}
                    rag_done = True

                else:
                    # All necessary knowledge gathering completed
                    break

            # ---------------------------------------------------------
            # Duplicate Action Check
            # ---------------------------------------------------------
            action_key = (tool_name, str(sorted(action_kwargs.items())))
            if action_key in seen_actions:
                # Disallow duplicate execution
                break
            seen_actions.add(action_key)

            # a(t): Selected action
            a_t = {
                "tool_name": tool_name,
                "kwargs": action_kwargs,
            }

            # ---------------------------------------------------------
            # Action Execution: a(t) -> action_result
            # ---------------------------------------------------------
            if tool_name in self.tools:
                action_result = self.tools[tool_name](**action_kwargs)
            else:
                action_result = {
                    "status": "FAILED",
                    "code": "TOOL_NOT_FOUND",
                    "result": None,
                    "message": f"Tool '{tool_name}' not available in tool registry.",
                }

            # ---------------------------------------------------------
            # State Transition Equation: s(t+1) = F(s(t), x(t), C(t), a(t))
            # ---------------------------------------------------------
            next_status = current_status
            next_domain = domain
            next_entity = active_entity

            # Update accumulated context based on action_result
            if tool_name == "safety_check":
                if action_result.get("status") == "BLOCKED":
                    next_status = "REFUSED_SAFETY"

            elif tool_name == "kg_query":
                if action_result.get("status") == "SUCCESS":
                    new_triples = action_result.get("result", [])
                    accumulated_kg.extend(new_triples)

            elif tool_name == "rag_retrieve":
                if action_result.get("status") == "SUCCESS":
                    accumulated_rag.extend(action_result.get("result", []))
                elif action_result.get("code") == "ABSTAIN_KNOWLEDGE_GAP":
                    knowledge_gap_detected = True

            elif tool_name == "ml_predict":
                if action_result.get("status") == "SUCCESS":
                    accumulated_ml = action_result.get("result")

            s_next = {
                "step": step_idx + 1,
                "active_domain": next_domain,
                "active_entity": next_entity,
                "remaining_steps": max(0, remaining_steps - 1),
                "status": next_status,
            }

            # Generate human-readable thought for ReAct timeline
            if tool_name == "safety_check":
                thought = f"Bước 1: Thực hiện kiểm tra an toàn bắt buộc cho câu hỏi miền '{domain}' nhằm đảm bảo ranh giới đạo đức y khoa/tài chính."
            elif tool_name == "ml_predict":
                thought = f"Phát hiện ý định dự báo định lượng; kích hoạt mô hình máy học cục bộ để phân tích rủi ro/xu hướng cho miền {domain}."
            elif tool_name == "kg_query":
                thought = f"Xác định thực thể kích hoạt '{active_entity}'; truy vấn Đồ thị Tri thức để trích xuất các bộ ba quan hệ cấu trúc 1-hop."
            elif tool_name == "rag_retrieve":
                thought = f"Truy xuất tài liệu quy chuẩn và báo cáo chính thức cho '{resolved_query}' với ngưỡng tương đồng tau=0.12."
            else:
                thought = f"Thực thi công cụ '{tool_name}' theo kế hoạch với tham số {action_kwargs}."

            # Record step trace
            step_record = {
                "step": step_idx,
                "thought": thought,
                "action": tool_name,
                "action_input": action_kwargs,
                "observation": action_result,
                "s_t": s_t,
                "x_t": x_t,
                "C_t": C_t,
                "a_t": a_t,
                "action_result": action_result,
                "s_next": s_next,
            }
            trace["steps"].append(step_record)

            # Check if safety violation halted execution immediately
            if next_status == "REFUSED_SAFETY":
                trace["status"] = "REFUSED_SAFETY"
                trace["final_answer"] = action_result.get("message", "Request refused by safety guardrails.")
                self.memory.update_interaction(
                    query=query,
                    actions=["safety_check"],
                    answer=trace["final_answer"],
                    entity=active_entity,
                    domain=domain,
                )
                trace["total_steps"] = len(trace["steps"])
                trace["context"] = {
                    "kg_facts": accumulated_kg,
                    "rag_chunks": accumulated_rag,
                    "ml_output": accumulated_ml,
                }
                trace["memory_state"] = {
                    "active_entity": active_entity,
                    "active_domain": domain,
                    "turn_count": len(self.memory.conversation_history),
                    "user_profile": self.memory.user_profile,
                }
                return trace

            # Prepare observation for next step
            current_observation = action_result
            current_status = next_status
            step_idx += 1

        # -------------------------------------------------------------
        # Post-loop Termination & Grounded Synthesis
        # -------------------------------------------------------------
        # Knowledge Gap Check (No RAG and No KG facts)
        if knowledge_gap_detected and len(accumulated_kg) == 0 and accumulated_ml is None:
            trace["status"] = "ABSTAIN_KNOWLEDGE_GAP"
            trace["final_answer"] = (
                "Cảnh báo Khoảng trống Tri thức (Knowledge Gap): Hệ thống không tìm thấy đủ tri thức tin cậy "
                "trong cơ sở tri thức (Knowledge Graph và tài liệu quy chuẩn) để giải đáp câu hỏi này "
                "(độ tương đồng dưới ngưỡng an toàn tau=0.12). Theo nguyên tắc kiểm soát ảo giác Chapter 6, "
                "hệ thống từ chối suy đoán."
            )
        else:
            trace["status"] = "SUCCESS"
            trace["final_answer"] = self._synthesize_grounded_answer(
                query=query,
                domain=domain,
                entity=active_entity,
                kg_facts=accumulated_kg,
                rag_chunks=accumulated_rag,
                ml_output=accumulated_ml,
            )

        # Update persistent interaction history
        action_names = [st["a_t"]["tool_name"] for st in trace["steps"]]
        self.memory.update_interaction(
            query=query,
            actions=action_names,
            answer=trace["final_answer"],
            entity=active_entity,
            domain=domain,
        )

        trace["total_steps"] = len(trace["steps"])
        trace["context"] = {
            "kg_facts": accumulated_kg,
            "rag_chunks": accumulated_rag,
            "ml_output": accumulated_ml,
        }
        trace["memory_state"] = {
            "active_entity": active_entity,
            "active_domain": domain,
            "turn_count": len(self.memory.conversation_history),
            "user_profile": self.memory.user_profile,
        }
        return trace

    # Ergonomic aliases
    def chat(self, query: str) -> Dict[str, Any]:
        return self.run(query)

    def execute(self, query: str) -> Dict[str, Any]:
        return self.run(query)
