"""
Unit tests for file_utils module.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from src.drawing_intelligence.utils import file_utils


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "test_dir"
        assert not new_dir.exists()

        file_utils.ensure_directory(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_create_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"

        file_utils.ensure_directory(str(nested_dir))

        assert nested_dir.exists()

    def test_existing_directory_no_error(self, tmp_path):
        """Test that existing directory doesn't raise error."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        # Should not raise
        file_utils.ensure_directory(str(existing_dir))


class TestGetFileHash:
    """Tests for get_file_hash function."""

    def test_md5_hash(self, tmp_path):
        """Test MD5 hash calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = file_utils.get_file_hash(str(test_file), "md5")

        # Known MD5 hash for "Hello, World!"
        expected = "65a8e27d8879283831b664bd8b7f0ad4"
        assert hash_result == expected

    def test_sha256_hash(self, tmp_path):
        """Test SHA256 hash calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = file_utils.get_file_hash(str(test_file), "sha256")

        assert len(hash_result) == 64  # SHA256 produces 64 character hex

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            file_utils.get_file_hash("nonexistent_file.txt")

    def test_invalid_algorithm_raises_error(self, tmp_path):
        """Test that invalid algorithm raises ValueError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError):
            file_utils.get_file_hash(str(test_file), "invalid_algo")


class TestGetFileSize:
    """Tests for get_file_size_mb function."""

    def test_small_file_size(self, tmp_path):
        """Test file size calculation for small file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 1024)  # 1 KB

        size_mb = file_utils.get_file_size_mb(str(test_file))

        # 1 KB = 0.00097... MB
        assert 0.0009 < size_mb < 0.001

    def test_exact_1mb_file(self, tmp_path):
        """Test file size for exactly 1 MB file."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"A" * (1024 * 1024))

        size_mb = file_utils.get_file_size_mb(str(test_file))

        assert size_mb == 1.0


class TestListFiles:
    """Tests for list_files function."""

    def test_list_all_files(self, tmp_path):
        """Test listing all files in directory."""
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.pdf").write_text("test")
        (tmp_path / "file3.doc").write_text("test")

        files = file_utils.list_files(str(tmp_path))

        assert len(files) == 3
        assert all(os.path.isabs(f) for f in files)

    def test_filter_by_extension(self, tmp_path):
        """Test filtering files by extension."""
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.pdf").write_text("test")
        (tmp_path / "file3.pdf").write_text("test")

        pdf_files = file_utils.list_files(str(tmp_path), extension=".pdf")

        assert len(pdf_files) == 2
        assert all(f.endswith(".pdf") for f in pdf_files)

    def test_recursive_listing(self, tmp_path):
        """Test recursive file listing."""
        (tmp_path / "file1.txt").write_text("test")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("test")

        files = file_utils.list_files(str(tmp_path), recursive=True)

        assert len(files) == 2

    def test_not_directory_raises_error(self, tmp_path):
        """Test that non-directory path raises error."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(NotADirectoryError):
            file_utils.list_files(str(file_path))


class TestSafeFilename:
    """Tests for safe_filename function."""

    def test_remove_unsafe_characters(self):
        """Test removal of unsafe characters."""
        unsafe_name = "file<name>:test?.txt"
        safe_name = file_utils.safe_filename(unsafe_name)

        assert "<" not in safe_name
        assert ">" not in safe_name
        assert ":" not in safe_name
        assert "?" not in safe_name

    def test_strip_spaces_and_dots(self):
        """Test stripping of leading/trailing spaces and dots."""
        name = "  ..filename.txt..  "
        safe_name = file_utils.safe_filename(name)

        assert not safe_name.startswith(" ")
        assert not safe_name.startswith(".")
        assert not safe_name.endswith(" ")
        assert not safe_name.endswith(".")

    def test_empty_filename_becomes_unnamed(self):
        """Test that empty filename becomes 'unnamed'."""
        safe_name = file_utils.safe_filename("")
        assert safe_name == "unnamed"

    def test_truncate_long_filename(self):
        """Test truncation of very long filename."""
        long_name = "a" * 300 + ".txt"
        safe_name = file_utils.safe_filename(long_name, max_length=255)

        assert len(safe_name) <= 255
        assert safe_name.endswith(".txt")


class TestGenerateUniqueId:
    """Tests for generate_unique_id function."""

    def test_without_prefix(self):
        """Test ID generation without prefix."""
        uid = file_utils.generate_unique_id()

        parts = uid.split("-")
        assert len(parts) == 3  # YYYYMMDD-HHMMSS-HASH
        assert len(parts[0]) == 8  # Date
        assert len(parts[1]) == 6  # Time
        assert len(parts[2]) == 8  # Hash

    def test_with_prefix(self):
        """Test ID generation with prefix."""
        uid = file_utils.generate_unique_id("DWG")

        assert uid.startswith("DWG-")
        parts = uid.split("-")
        assert len(parts) == 4  # PREFIX-YYYYMMDD-HHMMSS-HASH

    def test_uniqueness(self):
        """Test that consecutive IDs are unique."""
        uid1 = file_utils.generate_unique_id()
        uid2 = file_utils.generate_unique_id()

        assert uid1 != uid2


class TestCopyFileWithBackup:
    """Tests for copy_file_with_backup function."""

    def test_copy_to_new_location(self, tmp_path):
        """Test copying to new location."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "dest.txt"

        file_utils.copy_file_with_backup(str(source), str(dest))

        assert dest.exists()
        assert dest.read_text() == "test content"

    def test_backup_existing_destination(self, tmp_path):
        """Test that existing destination is backed up."""
        source = tmp_path / "source.txt"
        source.write_text("new content")
        dest = tmp_path / "dest.txt"
        dest.write_text("old content")

        file_utils.copy_file_with_backup(str(source), str(dest))

        backup = tmp_path / "dest.txt.bak"
        assert backup.exists()
        assert backup.read_text() == "old content"
        assert dest.read_text() == "new content"

    def test_nonexistent_source_raises_error(self, tmp_path):
        """Test that nonexistent source raises error."""
        dest = tmp_path / "dest.txt"

        with pytest.raises(FileNotFoundError):
            file_utils.copy_file_with_backup("nonexistent.txt", str(dest))


class TestAtomicWrite:
    """Tests for atomic_write function."""

    def test_atomic_write_new_file(self, tmp_path):
        """Test atomic write to new file."""
        file_path = tmp_path / "test.txt"
        content = "test content"

        file_utils.atomic_write(str(file_path), content)

        assert file_path.exists()
        assert file_path.read_text() == content

    def test_atomic_write_overwrites_existing(self, tmp_path):
        """Test atomic write overwrites existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("old content")

        file_utils.atomic_write(str(file_path), "new content")

        assert file_path.read_text() == "new content"

    def test_no_partial_write_on_error(self, tmp_path):
        """Test that partial writes don't occur on error."""
        # This test is conceptual - actual implementation would need
        # more sophisticated error injection
        pass


class TestGetAvailableDiskSpace:
    """Tests for get_available_disk_space function."""

    def test_returns_positive_number(self, tmp_path):
        """Test that function returns positive GB value."""
        space_gb = file_utils.get_available_disk_space(str(tmp_path))

        assert space_gb > 0
        assert isinstance(space_gb, float)

    def test_nonexistent_path_raises_error(self):
        """Test that nonexistent path raises error."""
        with pytest.raises(OSError):
            file_utils.get_available_disk_space("/nonexistent/path")


# Test fixtures
@pytest.fixture
def tmp_path(tmpdir):
    """Provide temporary directory for tests."""
    return Path(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
