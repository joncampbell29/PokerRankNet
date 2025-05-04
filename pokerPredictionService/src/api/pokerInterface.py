from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(
    prefix="/poker", 
    tags=["poker"]
)

@router.get("/hand_types")
def get_hand_types():
    return {
        "hand_types": [
            "High Card",
            "One Pair",
            "Two Pair",
            "Three of a Kind",
            "Straight",
            "Flush",
            "Full House",
            "Four of a Kind",
            "Straight Flush",
            "Royal Flush"
        ]
    }