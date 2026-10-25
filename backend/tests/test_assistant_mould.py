"""Mould questions get a throw-it-away answer, from the model and without it."""

from __future__ import annotations

from app.ai.assistant import _SYSTEM, answer_from_knowledge
from app.schemas import AssistantRequest


def test_the_fallback_answers_mould_not_storage() -> None:
    """It once matched "bread" and replied with how to store bread."""
    reply = answer_from_knowledge(
        AssistantRequest(message="Is my mouldy bread safe if I cut off the mould?")
    ).reply.lower()
    assert "throw it away" in reply
    assert "sealed bag" not in reply


def test_the_model_is_told_discarding_is_not_a_safety_judgement() -> None:
    """"Never judge safety" made the model hedge, then suggest freezing it."""
    assert "THROW FOOD AWAY is not a safety judgement" in _SYSTEM
    assert "do not assume or invent items" in _SYSTEM
