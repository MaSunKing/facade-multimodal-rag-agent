import unittest
from unittest.mock import patch
from pathlib import Path, PurePosixPath
from backend.sales.retriever import LocalRagRetriever


class WslBackendTests(unittest.TestCase):
    def test_windows_asset_path_maps_to_same_drive_not_new_data_copy(self):
        with patch('backend.sales.retriever.os.name', 'posix'), \
             patch('backend.sales.retriever.Path') as path:
            path.side_effect = lambda value: PurePosixPath(value)
            with patch('pathlib.PurePosixPath.is_file', create=True, return_value=True):
                result = LocalRagRetriever._resolve_visual_path('C:\\Projects\\资料\\sample.png')
        self.assertEqual(str(result), '/mnt/c/Projects/资料/sample.png')

    def test_existing_linux_path_is_unchanged(self):
        with patch.object(Path, 'is_file', return_value=True):
            result = LocalRagRetriever._resolve_visual_path('/mnt/c/Projects/sample.png')
        self.assertEqual(str(result).replace('\\', '/'), '/mnt/c/Projects/sample.png')
