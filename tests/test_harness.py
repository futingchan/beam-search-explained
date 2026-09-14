"""Tests for the benchmark harness — no GPU or model weights required.

Run:  python -m unittest discover tests      (stdlib only)
      python -m pytest tests/                (if pytest installed)

Covers the three things the blog's numbers rest on: the from-scratch
beam implementation, the JSON validity check, and the canonical-task
scorers. Model-loading paths are deliberately untested here.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import beam_search_scratch as bss
import experiment_canonical as ec
from demo_hf import try_parse


class TestBeamSearchScratch(unittest.TestCase):
    def test_beam_beats_greedy_on_rigged_lm(self):
        # The toy tree is rigged so greedy commits to "Visit" (-0.4) and
        # lands on a -3.9 branch, while beam finds "Book ..." at -1.1.
        best = bss.beam_search(bss.toy_lm, beam_width=2, verbose=False)[0]
        self.assertEqual(best[0][0], "Book")
        self.assertGreater(best[1], -1.2)

    def test_width1_is_greedy(self):
        best = bss.beam_search(bss.toy_lm, beam_width=1, verbose=False)[0]
        self.assertEqual(best[0], ["Visit", "museum", "2026-10-03"])

    def test_results_sorted_best_first(self):
        res = bss.beam_search(bss.toy_lm, beam_width=4, verbose=False)
        scores = [s for _, s in res]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestTryParse(unittest.TestCase):
    VALID = '{"recommendations": [{"action": "Book", "target": "hotel"}]}'

    def test_valid(self):
        self.assertEqual(try_parse(self.VALID)[0]["action"], "Book")

    def test_malformed_json(self):
        self.assertIsNone(try_parse('{"recommendations": [{'))

    def test_off_vocab_action_rejected(self):
        self.assertIsNone(try_parse(
            '{"recommendations": [{"action": "Fly", "target": "x"}]}'))

    def test_missing_key(self):
        self.assertIsNone(try_parse('{"other": []}'))


class TestExtractionScorer(unittest.TestCase):
    ITEM = {"gold": {"vendor": "Skyline Airlines", "date": "2026-03-14",
                     "amount": "412.50", "currency": "USD",
                     "confirmation_code": "SKX-8841"}}

    def test_perfect(self):
        self.assertEqual(ec.score_extraction(
            '{"vendor":"Skyline Airlines","date":"2026-03-14",'
            '"amount":"412.50","currency":"USD",'
            '"confirmation_code":"SKX-8841"}', self.ITEM), 1.0)

    def test_case_insensitive_and_amount_tolerance(self):
        self.assertEqual(ec.score_extraction(
            '{"vendor":"skyline airlines","date":"2026-03-14",'
            '"amount":"412.5","currency":"usd",'
            '"confirmation_code":"skx-8841"}', self.ITEM), 1.0)

    def test_partial(self):
        self.assertEqual(ec.score_extraction(
            '{"vendor":"Skyline Airlines","date":"WRONG","amount":"1",'
            '"currency":"EUR","confirmation_code":"X"}', self.ITEM), 0.2)

    def test_garbage_scores_zero(self):
        self.assertEqual(ec.score_extraction("not json at all", self.ITEM), 0.0)

    def test_markdown_fence_stripped(self):
        self.assertEqual(ec.score_extraction(
            '```json\n{"vendor":"Skyline Airlines","date":"2026-03-14",'
            '"amount":"412.50","currency":"USD",'
            '"confirmation_code":"SKX-8841"}\n```', self.ITEM), 1.0)


class TestNl2sqlScorer(unittest.TestCase):
    def test_correct_query(self):
        item = next(i for i in self._items() if "COUNT" in i["sql"].upper())
        self.assertEqual(ec.score_nl2sql(item["sql"], item), 1.0)

    def test_wrong_result(self):
        item = self._items()[0]
        self.assertEqual(ec.score_nl2sql("SELECT 1;", item), 0.0)

    def test_invalid_sql(self):
        item = self._items()[0]
        self.assertEqual(ec.score_nl2sql("SELEC * FRM trips;", item), 0.0)

    @staticmethod
    def _items():
        import json
        with open(ec.ROOT / "data" / "nl2sql.json") as f:
            return json.load(f)["items"]


class TestTranslationScorer(unittest.TestCase):
    def test_identical_scores_high(self):
        try:
            import sacrebleu  # noqa: F401
        except ImportError:
            self.skipTest("sacrebleu not installed")
        score = ec.score_translation("Das Wetter ist schön.",
                                     {"ref": "Das Wetter ist schön."})
        self.assertGreater(score, 90.0)


if __name__ == "__main__":
    unittest.main()
