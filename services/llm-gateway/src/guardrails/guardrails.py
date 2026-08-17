from typing import Dict, Any
import re
from ..config.settings import settings


class InputGuardrail:
    @staticmethod
    async def validate(prompt: str) -> bool:
        """Basic check: no obvious injection, length limits."""
        if not prompt or len(prompt) > 10000:
            return False
        # Simple injection detection (naive)
        if re.search(r"(?i)(system\s*:|role\s*:|instruction\s*:)", prompt):
            # Allow if it's part of normal text? For now, block.
            return False
        return True


class OutputGuardrail:
    @staticmethod
    async def validate(content: str) -> bool:
        """Check for toxicity, PII, etc. (placeholder)."""
        if not content:
            return False
        if len(content) > 50000:
            return False
        # Simple check: no obvious private info (phone, email)
        # Could be extended with regex.
        return True


async def apply_input_guardrails(prompt: str) -> bool:
    if not settings.enable_input_guardrails:
        return True
    return await InputGuardrail.validate(prompt)


async def apply_output_guardrails(content: str) -> bool:
    if not settings.enable_output_guardrails:
        return True
    return await OutputGuardrail.validate(content)