import copy
from typing import Any, Dict, List, Optional


class AgentMemory:
    """Two-tier memory system for Antigravity Intelligent Agent (Chapter 6).

    Short-term memory manages conversational state, active entity, and domain context.
    Long-term memory persists user profile defaults and historical attributes.
    """

    def __init__(self) -> None:
        self.short_term: Dict[str, Any] = {
            "conversation_history": [],
            "active_entity": None,
            "active_domain": None,
        }
        self.long_term: Dict[str, Any] = {
            "user_profile": {
                "default_age": 45,
                "default_bmi": 26.5,
                "portfolio": ["FPT", "HPG"],
            }
        }

    # ------------------------------------------------------------------
    # Properties for direct ergonomic access
    # ------------------------------------------------------------------
    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        return self.short_term["conversation_history"]

    @property
    def active_entity(self) -> Optional[str]:
        return self.short_term.get("active_entity")

    @active_entity.setter
    def active_entity(self, value: Optional[str]) -> None:
        self.short_term["active_entity"] = value

    @property
    def active_domain(self) -> Optional[str]:
        return self.short_term.get("active_domain")

    @active_domain.setter
    def active_domain(self, value: Optional[str]) -> None:
        self.short_term["active_domain"] = value

    @property
    def user_profile(self) -> Dict[str, Any]:
        return self.long_term["user_profile"]

    # ------------------------------------------------------------------
    # Core Methods specified in Chapter 6
    # ------------------------------------------------------------------
    def update_interaction(
        self,
        query: str,
        actions: Any,
        answer: str,
        entity: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Appends interaction turn and updates active entity and domain state."""
        turn = {
            "turn_index": len(self.short_term["conversation_history"]) + 1,
            "query": query,
            "actions": actions,
            "answer": answer,
            "entity": entity,
            "domain": domain,
        }
        self.short_term["conversation_history"].append(turn)

        if entity is not None:
            self.short_term["active_entity"] = entity
        if domain is not None:
            self.short_term["active_domain"] = domain

        return turn

    def get_context(self) -> Dict[str, Any]:
        """Returns current active_entity, active_domain, and conversation turn count."""
        return {
            "active_entity": self.short_term.get("active_entity"),
            "active_domain": self.short_term.get("active_domain"),
            "turn_count": len(self.short_term.get("conversation_history", [])),
        }

    def clear_entity(self) -> None:
        """Resets active_entity to None upon detected topic shifts."""
        self.short_term["active_entity"] = None

    # ------------------------------------------------------------------
    # Key-Value Access for memory_tool
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves value by key from short_term, user_profile, or long_term."""
        if key in self.short_term:
            return copy.deepcopy(self.short_term[key])
        user_prof = self.long_term.get("user_profile", {})
        if key in user_prof:
            return copy.deepcopy(user_prof[key])
        if key in self.long_term:
            return copy.deepcopy(self.long_term[key])

        # Support dotted paths: e.g. 'short_term.active_entity', 'user_profile.default_age'
        if "." in key:
            parts = key.split(".")
            curr: Any = {"short_term": self.short_term, "long_term": self.long_term, "user_profile": user_prof}
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return default
            return copy.deepcopy(curr)

        return default

    def set(self, key: str, value: Any) -> None:
        """Sets value by key across short_term or user_profile."""
        if key == "active_entity":
            self.short_term["active_entity"] = value
        elif key == "active_domain":
            self.short_term["active_domain"] = value
        elif key in self.short_term:
            self.short_term[key] = value
        elif key in self.long_term.get("user_profile", {}):
            self.long_term["user_profile"][key] = value
        elif key == "user_profile":
            self.long_term["user_profile"] = value
        else:
            self.short_term[key] = value

    def reset(self) -> None:
        """Clears short_term history and states back to initial conditions."""
        self.short_term["conversation_history"] = []
        self.short_term["active_entity"] = None
        self.short_term["active_domain"] = None


# Shared singleton memory instance for default tool execution
shared_memory = AgentMemory()
