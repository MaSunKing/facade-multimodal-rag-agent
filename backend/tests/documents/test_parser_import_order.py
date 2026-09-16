"""Fresh-process import checks catch cycles hidden by app-first tests."""
from pathlib import Path
import subprocess
import sys
import unittest


class ParserImportOrderTests(unittest.TestCase):
    def test_parser_first_import_and_lazy_public_contract(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run([sys.executable, "-c",
            "from backend.document_parsing.file_ingestion import ingest_uploaded_file; "
            "from backend.documents import EvidenceDocumentV2, convert_intermediate_to_v2; "
            "assert ingest_uploaded_file and EvidenceDocumentV2 and convert_intermediate_to_v2"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
