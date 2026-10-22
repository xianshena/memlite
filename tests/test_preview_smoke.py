import unittest
from examples.cpu_quickstart import demo
class PreviewSmoke(unittest.TestCase):
    def test_observed_history_flip_and_clear(self):
        result=demo(3)
        self.assertEqual(result['history_swaps_flip_choice'],3)
        self.assertEqual(result['cleared_memory_reports_unknown'],3)
if __name__=='__main__':unittest.main()
