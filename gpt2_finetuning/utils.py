"""Utility functions for GPT-2 fine-tuning."""
from typing import List


def prepare_text_data(texts: List[str], separator: str = "\n\n") -> str:
    """Prepare texts for training."""
    return separator.join(texts)


class PromptTemplate:
    """Template for generating prompts."""

    @staticmethod
    def story(setting: str, character: str, conflict: str) -> str:
        """Story prompt template."""
        return f"Write a story about {character} in a {setting} facing {conflict}:"

    @staticmethod
    def conversation(context: str) -> str:
        """Conversation prompt template."""
        return f"Continue this conversation: {context}"

    @staticmethod
    def completion(text: str) -> str:
        """Completion prompt template."""
        return f"{text}"
