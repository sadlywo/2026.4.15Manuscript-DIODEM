import tempfile
import unittest
from pathlib import Path

from project.main_cleanup_workspace import _is_within_directory, _move_if_exists


class TestCleanupWorkspace(unittest.TestCase):
    def test_is_within_directory_rejects_sibling_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            root.mkdir()
            inside = root / "outputs" / "table.csv"
            sibling = Path(tmpdir) / "repo-other" / "table.csv"
            self.assertTrue(_is_within_directory(inside, root))
            self.assertFalse(_is_within_directory(sibling, root))

    def test_move_if_exists_moves_file_and_creates_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "mnt" / "data" / "figure.png"
            target = root / "docs" / "figures" / "figure.png"
            source.parent.mkdir(parents=True)
            source.write_text("figure", encoding="utf-8")
            moved = _move_if_exists(source, target, root=root, dry_run=False)
            self.assertTrue(moved)
            self.assertFalse(source.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "figure")


if __name__ == "__main__":
    unittest.main()
