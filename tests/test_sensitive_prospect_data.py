import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


class SensitiveProspectDataTest(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def test_sensitive_fields_do_not_leave_tools_or_reach_scoring_model(self):
        prospect_id = "LEAD-18993"
        sensitive_values = ("tax_id", "date_of_birth", "card_on_file", "credit_check_ref")

        prospect_result = get_prospect.invoke({"prospect_id": prospect_id})
        profile_result = build_prospect_profile.invoke({"prospect_id": prospect_id})

        scoring_llm = Mock()
        scoring_llm.invoke.return_value = Mock(model_dump=lambda: {"score": 80})
        with patch("gtm_agent.gtm_agent._scoring_llm", scoring_llm):
            score_prospect.invoke({
                "prospect_profile": {
                    "prospect_id": prospect_id,
                    "name": "Sven Smith",
                    "annual_revenue": 70000000,
                    "billing_qualification": {
                        field: field for field in sensitive_values
                    },
                },
                "offering": {
                    "required_tech_stack": ["AWS"],
                    "min_annual_revenue": 1,
                    "description": "A valid offering",
                },
            })

        for result in (prospect_result, profile_result):
            serialized = json.dumps(result)
            self.assertNotIn("billing_qualification", serialized)
            for field in sensitive_values[:3]:
                self.assertNotIn(field, serialized)

        scoring_messages = scoring_llm.invoke.call_args.args[0]
        scoring_payload = scoring_messages[1]["content"]
        self.assertNotIn("billing_qualification", scoring_payload)
        for field in sensitive_values[:3]:
            self.assertNotIn(field, scoring_payload)


if __name__ == "__main__":
    unittest.main()
