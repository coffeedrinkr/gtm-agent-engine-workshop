import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from gtm_agent import data_service
from gtm_agent import gtm_agent


PROSPECT_ID = "LEAD-12853"
TECHNOLOGY = "Snowflake"


@pytest.fixture(autouse=True)
def restore_prospect_state():
    record = data_service.PROSPECTS[PROSPECT_ID]
    original_stack = list(record["tech_stack"])
    original_profile = data_service._PROFILES.pop(PROSPECT_ID, None)
    yield
    record["tech_stack"] = original_stack
    if original_profile is None:
        data_service._PROFILES.pop(PROSPECT_ID, None)
    else:
        data_service._PROFILES[PROSPECT_ID] = original_profile


def test_update_persists_technology_to_source_record():
    data_service.update_prospect_info(PROSPECT_ID, TECHNOLOGY)

    assert TECHNOLOGY in data_service.fetch_tech_stack(PROSPECT_ID)


def test_update_invalidates_cached_profile():
    cached = gtm_agent.build_prospect_profile.func(PROSPECT_ID)

    data_service.update_prospect_info(PROSPECT_ID, TECHNOLOGY)
    refreshed = gtm_agent.build_prospect_profile.func(PROSPECT_ID)

    assert TECHNOLOGY not in cached["prospect_profile"]["tech_stack"]
    assert TECHNOLOGY in refreshed["prospect_profile"]["tech_stack"]


def test_score_reads_updated_technology_from_source_record(monkeypatch):
    profile = gtm_agent.build_prospect_profile.func(PROSPECT_ID)["prospect_profile"]
    offering = data_service.OFFERINGS["OFFER-10001"]
    captured = {}

    class FakeResult:
        def model_dump(self):
            return captured

    class FakeScoringLLM:
        def invoke(self, messages):
            captured["prospect_profile"] = messages[1]["content"]
            return FakeResult()

    monkeypatch.setattr(gtm_agent, "_scoring_llm", FakeScoringLLM())
    data_service.update_prospect_info(PROSPECT_ID, TECHNOLOGY)

    gtm_agent.score_prospect.func(profile, offering)

    assert f'"{TECHNOLOGY}"' in captured["prospect_profile"]
