import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import send_prospect_email


class Runtime:
    config = {"metadata": {"user_id": "REP-10001"}}


def test_disqualified_prospect_is_blocked_without_message_id():
    result = send_prospect_email.func(
        {
            "prospect_id": "LEAD-50001",
            "name": "Priya Nair",
            "email": "priya.nair@brightwaveapps.com",
            "disqualified": False,
        },
        "Checking in",
        "Hello",
        Runtime(),
    )

    assert result == {
        "status": "blocked",
        "reason": "Prospect is marked disqualified; send suppressed.",
        "prospect_id": "LEAD-50001",
    }


def test_non_disqualified_prospect_sends_normally():
    result = send_prospect_email.func(
        {
            "prospect_id": "LEAD-10001",
            "name": "Omar Okafor",
            "email": "omar.okafor@lakesideanalytics.com",
        },
        "Checking in",
        "Hello",
        Runtime(),
    )

    assert result["status"] == "sent"
    assert result["message_id"].startswith("msg-")
