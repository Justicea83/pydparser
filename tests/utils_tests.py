import unittest
from parser import utils


class UtilsTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_doc_upload_works(self):
        doc = utils.extract_text_from_doc('../files/doc/man.docx')
        self.assertTrue(len(doc) > 0)


if __name__ == '__main__':
    unittest.main()
