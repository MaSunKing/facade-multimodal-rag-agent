"""Test timer semantics without importing the API or loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch


class PlannerDeadlineTests(unittest.TestCase):
    def criterion(self, max_seconds, budget):
        path = Path(__file__).resolve().parents[3] / 'backend/app.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'local_generation_stopping_criteria')
        self.clock = SimpleNamespace(monotonic=lambda: 100.0)
        if budget is not None: budget.cancelled = threading.Event()
        namespace = {'time': self.clock, 'current_budget': SimpleNamespace(get=lambda: budget)}
        module = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
        with patch.dict('sys.modules', {'transformers': module}):
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
            return namespace['local_generation_stopping_criteria'](max_seconds)[0]

    def test_planner_has_no_local_time_cutoff(self):
        criterion = self.criterion(None, SimpleNamespace(expired=lambda: False))
        self.clock.monotonic = lambda: 100000.0
        self.assertFalse(criterion(None, None))

    def test_legacy_expired_flag_does_not_stop_without_cancel(self):
        criterion = self.criterion(None, SimpleNamespace(expired=lambda: True))
        self.assertFalse(criterion(None, None))

    def test_answer_node_local_deadline_is_removed(self):
        criterion = self.criterion(15, None)
        self.clock.monotonic = lambda: 116.0
        self.assertFalse(criterion(None, None))


if __name__ == '__main__':
    unittest.main()
