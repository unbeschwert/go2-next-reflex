"""409 'please confirm' answers: {"detail": ..., "confirm": "<question for the user>"} (see docs/API.md)."""
from fastapi import HTTPException


class Confirm(HTTPException):
    def __init__(self, question: str, detail: str = "confirmation required"):
        super().__init__(409, detail)
        self.question = question


def need(confirm: bool, question: str, detail: str = "confirmation required") -> None:
    """Raise a 409 confirm request unless the client already confirmed."""
    if not confirm:
        raise Confirm(question, detail)
