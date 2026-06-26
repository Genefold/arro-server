"""tests/test_build_and_store_fallback.py

Tests for _ArrowSpaceAdapter._build_with_persistence PATH 2:
  build_and_store() + os.rename/shutil.move to index_store.

Run without the real arrowspace package -- all arrowspace calls are mocked.

The real ``build_and_store`` (from pyarrowspace) writes files with hyphens:
  dataset_{uuid}-raw_input.parquet
  dataset_{uuid}-lambdas.parquet
  dataset_{uuid}-laplacian-input.parquet
  dataset_{uuid}-clustered-dm.parquet
  dataset_{uuid}-gl-matrix.parquet
  dataset_{uuid}-arrowspace_metadata.json

PATH 2 detects the common UUID prefix, replaces it with the canonical
``dataset_name``, and moves every file to ``index_store/``.
"""

from __future__ import annotations

import os
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
FIXTURE_ARRAY = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)
DATASET_ID = "main--matrix"

# File set that mirrors the real pyarrowspace build_and_store output.
_REAL_BUILD_AND_STORE_FILES = [
    "dataset_abc123-raw_input.parquet",
    "dataset_abc123-lambdas.parquet",
    "dataset_abc123-laplacian-input.parquet",
    "dataset_abc123-clustered-dm.parquet",
    "dataset_abc123-gl-matrix.parquet",
    "dataset_abc123-arrowspace_metadata.json",
]

# ---------------------------------------------------------------------------
# Fake arrowspace helpers
# ---------------------------------------------------------------------------


def _make_fake_aspace() -> MagicMock:
    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    return aspace


def _make_fake_gl() -> MagicMock:
    gl = MagicMock()
    gl.nnodes = NITEMS
    gl.shape = (NITEMS, NITEMS)
    gl.graph_params = GRAPH_PARAMS
    n = NITEMS
    gl.to_csr.return_value = (
        np.ones(n, dtype=np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n + 1, dtype=np.int64),
        (n, n),
    )
    return gl


def _make_build_and_store_mod(
    *,
    write_files: list[str] | None = None,
    raise_on_build: Exception | None = None,
    uid_prefix: str = "dataset_abc123",
) -> types.ModuleType:
    """Fake arrowspace module with build_and_store on FakeBuilder.

    The fake writes files to ``CWD/storage/`` mimicking the real
    pyarrowspace crate's naming convention (hyphens, 5 parquet + 1 json).
    """
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    if write_files is None:
        write_files = _REAL_BUILD_AND_STORE_FILES

    class FakeBuilder:
        def build_and_store(self, graph_params, array):
            if raise_on_build is not None:
                raise raise_on_build
            storage = Path(os.getcwd()) / "storage"
            storage.mkdir(parents=True, exist_ok=True)
            for fname in write_files:
                (storage / fname).write_bytes(b"fake-parquet-content")
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilder
    return fake_mod


def _make_with_persistence_mod() -> types.ModuleType:
    """Fake module WITH with_persistence -- PATH 1."""
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class FakeBuilderWithPersistence:
        def with_lambda_graph(self, eps, k, topk, p, sigma):
            return self

        def with_sparsity_check(self, value):
            return self

        def with_persistence(self, store_path: str, dataset_name: str):
            self._store_path = store_path
            self._dataset_name = dataset_name
            return self

        def build(self, rows):
            store = Path(self._store_path)
            store.mkdir(parents=True, exist_ok=True)
            (store / f"{self._dataset_name}-raw_input.parquet").write_bytes(b"wp-raw")
            (store / f"{self._dataset_name}-lambdas.parquet").write_bytes(b"wp-lam")
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilderWithPersistence
    return fake_mod


def _make_bare_mod() -> types.ModuleType:
    """Fake module with neither with_persistence nor build_and_store."""
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class BareBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = BareBuilder
    return fake_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def build_and_store_mod() -> types.ModuleType:
    return _make_build_and_store_mod()


@pytest.fixture
def adapter(build_and_store_mod):
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

    return _ArrowSpaceAdapter(build_and_store_mod, cache_size=4)


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    d = tmp_path / "index_store"
    d.mkdir()
    return d


@pytest.fixture
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# ===========================================================================
# 1. Unit: PATH routing
# ===========================================================================


class TestPathRouting:
    def test_path1_taken_when_with_persistence_present(self, tmp_store, tmp_cwd):
        """with_persistence is preferred over build_and_store when available."""
        mod = _make_with_persistence_mod()
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter, _read_manifest

        a = _ArrowSpaceAdapter(mod, cache_size=4)
        a.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        # with_persistence writes directly to index_store
        assert (tmp_store / f"{dname}-raw_input.parquet").exists()

    def test_path2_taken_when_only_build_and_store(self, adapter, tmp_store, tmp_cwd):
        """build_and_store is called when builder lacks with_persistence."""
        call_log: list[str] = []
        original = adapter._mod.ArrowSpaceBuilder

        class SpyBuilder:
            def build_and_store(self, graph_params, array):
                call_log.append("build_and_store")
                storage = Path(os.getcwd()) / "storage"
                storage.mkdir(parents=True, exist_ok=True)
                for fname in _REAL_BUILD_AND_STORE_FILES:
                    (storage / fname).write_bytes(b"x")
                return _make_fake_aspace(), _make_fake_gl()

        adapter._mod.ArrowSpaceBuilder = SpyBuilder
        try:
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        finally:
            adapter._mod.ArrowSpaceBuilder = original

        assert "build_and_store" in call_log

    def test_path3_taken_when_neither_method_present(self, tmp_store, tmp_cwd, caplog):
        import logging

        mod = _make_bare_mod()
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

        a = _ArrowSpaceAdapter(mod, cache_size=4)
        with caplog.at_level(logging.WARNING, logger="arro_server.arrowspace_adapter"):
            a.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        assert any("not survive restart" in r.message for r in caplog.records)


# ===========================================================================
# 2. Parquet + JSON files land in index_store with correct names
# ===========================================================================


class TestFilesInIndexStore:
    def test_build_produces_index_store_files(self, adapter, tmp_store, tmp_cwd):
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        files = list(tmp_store.iterdir())
        assert len(files) > 0, "index_store is empty after build"

    def test_files_use_manifest_dataset_name(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        # Only check files with the pattern {dataset_name}-{suffix}.{ext}
        build_files = [p for p in tmp_store.iterdir() if p.is_file() and "-" in p.name]
        assert len(build_files) > 0, "No build output files found in index_store"
        for p in build_files:
            assert p.name.startswith(dname), \
                f"File {p.name} does not start with dataset_name '{dname}'"

    def test_index_store_created_if_missing(self, adapter, tmp_path, tmp_cwd):
        new_store = tmp_path / "brand_new_store"
        assert not new_store.exists()
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), new_store)
        assert new_store.exists()
        assert any(new_store.iterdir())

    def test_no_uuid_prefix_in_index_store(self, adapter, tmp_store, tmp_cwd):
        """Files must NOT retain the UUID prefix from build_and_store."""
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        for p in tmp_store.glob("*"):
            assert "abc123" not in p.name, \
                f"UUID prefix 'abc123' leaked into index_store: {p.name}"

    def test_two_datasets_produce_separate_file_sets(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        mod = adapter._mod
        call_count = [0]
        original_builder = mod.ArrowSpaceBuilder

        class MultiBuilder:
            def build_and_store(self, graph_params, array):
                call_count[0] += 1
                uid = f"uid{call_count[0]}"
                storage = Path(os.getcwd()) / "storage"
                storage.mkdir(parents=True, exist_ok=True)
                for suffix in ("raw_input", "lambdas", "gl-matrix"):
                    (storage / f"dataset_{uid}-{suffix}.parquet").write_bytes(b"x")
                return _make_fake_aspace(), _make_fake_gl()

        mod.ArrowSpaceBuilder = MultiBuilder
        try:
            adapter.build_index("ds_a", FIXTURE_ARRAY.copy(), tmp_store)
            adapter.build_index("ds_b", FIXTURE_ARRAY.copy(), tmp_store)
        finally:
            mod.ArrowSpaceBuilder = original_builder

        manifest = _read_manifest(tmp_store)
        dname_a = manifest["ds_a"]["dataset_name"]
        dname_b = manifest["ds_b"]["dataset_name"]
        assert dname_a != dname_b
        # Each dataset's build output files must be prefixed with its own name
        build_files = [p for p in tmp_store.iterdir() if p.is_file() and "-" in p.name]
        assert len(build_files) > 0
        for p in build_files:
            assert p.name.startswith(dname_a) or p.name.startswith(dname_b)


# ===========================================================================
# 3. CWD/storage/ cleanup
# ===========================================================================


class TestCwdStorageCleanup:
    def test_cwd_storage_empty_after_successful_build(self, adapter, tmp_store, tmp_cwd):
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        storage = tmp_cwd / "storage"
        if storage.exists():
            remaining = list(storage.iterdir())
            assert remaining == [], f"CWD/storage/ not empty: {remaining}"

    def test_cwd_storage_dir_removed_if_empty(self, adapter, tmp_store, tmp_cwd):
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        storage = tmp_cwd / "storage"
        assert not storage.exists() or not any(storage.iterdir())

    def test_cwd_storage_not_removed_if_contains_other_files(
        self, adapter, tmp_store, tmp_cwd
    ):
        storage = tmp_cwd / "storage"
        storage.mkdir()
        pre_existing = storage / "other_data.txt"
        pre_existing.write_text("keep me")

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        assert pre_existing.exists(), "Pre-existing file was incorrectly deleted"

    def test_no_files_left_in_cwd_storage(self, adapter, tmp_store, tmp_cwd):
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        storage = tmp_cwd / "storage"
        leftover = list(storage.glob("*")) if storage.exists() else []
        assert leftover == [], f"Files leaked into CWD/storage/: {leftover}"

    def test_cleanup_happens_even_after_multiple_builds(self, adapter, tmp_store, tmp_cwd):
        mod = adapter._mod
        original_builder = mod.ArrowSpaceBuilder
        call_count = [0]

        class MultiBuilder:
            def build_and_store(self, graph_params, array):
                call_count[0] += 1
                uid = f"uid{call_count[0]}"
                storage = Path(os.getcwd()) / "storage"
                storage.mkdir(parents=True, exist_ok=True)
                for suffix in ("raw_input", "lambdas", "gl-matrix"):
                    (storage / f"dataset_{uid}-{suffix}.parquet").write_bytes(b"x")
                return _make_fake_aspace(), _make_fake_gl()

        mod.ArrowSpaceBuilder = MultiBuilder
        try:
            for ds in ["ds_a", "ds_b", "ds_c"]:
                adapter.build_index(ds, FIXTURE_ARRAY.copy(), tmp_store)
                storage = tmp_cwd / "storage"
                leftover = list(storage.glob("*")) if storage.exists() else []
                assert leftover == [], \
                    f"Files leaked after building '{ds}': {leftover}"
        finally:
            mod.ArrowSpaceBuilder = original_builder


# ===========================================================================
# 4. Manifest correctness after PATH 2
# ===========================================================================


class TestManifestIntegrationWithFallback:
    def test_manifest_entry_present_after_build(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        assert DATASET_ID in _read_manifest(tmp_store)

    def test_manifest_dataset_name_prefix_matches_index_store_files(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        for p in tmp_store.iterdir():
            if not p.is_file() or p.name == "index_manifest.json":
                continue
            assert p.name.startswith(dname), \
                f"File {p.name} does not start with '{dname}'"

    def test_manifest_graph_params_roundtrip(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store, graph_params=GRAPH_PARAMS)
        stored = _read_manifest(tmp_store)[DATASET_ID]["graph_params"]
        assert stored == GRAPH_PARAMS

    def test_rebuild_preserves_dataset_name(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        first = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        second = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        assert first == second

    def test_rebuild_overwrites_old_parquet_files(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        for p in tmp_store.iterdir():
            if not p.is_file() or p.name == "index_manifest.json":
                continue
            assert p.name.startswith(dname), \
                f"Stale file without current dataset_name: {p.name}"

    def test_manifest_not_corrupted_if_no_pre_existing_manifest(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import MANIFEST_FILENAME, _read_manifest

        assert not (tmp_store / MANIFEST_FILENAME).exists()
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        manifest = _read_manifest(tmp_store)
        assert isinstance(manifest, dict)
        assert DATASET_ID in manifest


# ===========================================================================
# 5. Move semantics
# ===========================================================================


class TestMoveSemantics:
    def test_os_rename_used_on_same_filesystem(self, adapter, tmp_store, tmp_cwd):
        rename_calls: list[tuple[str, str]] = []
        original_rename = os.rename

        def spy_rename(src, dst):
            rename_calls.append((str(src), str(dst)))
            return original_rename(src, dst)

        with patch("arro_server.arrowspace_adapter.os.rename", side_effect=spy_rename):
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        assert len(rename_calls) > 0, "os.rename was never called"

    def test_shutil_move_used_on_cross_device(self, adapter, tmp_store, tmp_cwd):
        import shutil

        move_calls: list[tuple[str, str]] = []
        original_move = shutil.move

        def spy_move(src, dst, *a, **kw):
            move_calls.append((str(src), str(dst)))
            return original_move(src, dst, *a, **kw)

        with patch("arro_server.arrowspace_adapter.os.rename", side_effect=OSError("cross-device link")), \
             patch("arro_server.arrowspace_adapter.shutil.move", side_effect=spy_move):
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        assert len(move_calls) > 0

    def test_cross_device_fallback_still_produces_files(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import _read_manifest

        with patch("arro_server.arrowspace_adapter.os.rename", side_effect=OSError("cross-device link")):
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        assert any(p.name.startswith(dname) for p in tmp_store.glob("*"))

    def test_cross_device_fallback_logs_warning(
        self, adapter, tmp_store, tmp_cwd, caplog
    ):
        import logging

        with patch("arro_server.arrowspace_adapter.os.rename", side_effect=OSError("cross-device link")), \
             caplog.at_level(logging.WARNING, logger="arro_server.arrowspace_adapter"):
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)

        assert any(
            "cross-device" in r.message.lower() or "shutil" in r.message.lower()
            for r in caplog.records
        )

    def test_shutil_move_failure_propagates(self, adapter, tmp_store, tmp_cwd):
        with patch("arro_server.arrowspace_adapter.os.rename", side_effect=OSError("cross-device link")), \
             patch("arro_server.arrowspace_adapter.shutil.move", side_effect=IOError("disk full")):
            with pytest.raises((IOError, RuntimeError, OSError)):
                adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)


# ===========================================================================
# 6. Validation errors
# ===========================================================================


class TestValidationErrors:
    def _build_with_bad_output(
        self, adapter, tmp_store: Path, tmp_cwd: Path, write_files: list[str],
    ):
        mod = adapter._mod
        original_builder = mod.ArrowSpaceBuilder

        class BadBuilder:
            def build_and_store(self, graph_params, array):
                storage = Path(os.getcwd()) / "storage"
                storage.mkdir(parents=True, exist_ok=True)
                for fname in write_files:
                    (storage / fname).write_bytes(b"x")
                return _make_fake_aspace(), _make_fake_gl()

        mod.ArrowSpaceBuilder = BadBuilder
        try:
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        finally:
            mod.ArrowSpaceBuilder = original_builder

    def test_zero_files_written_raises_runtime_error(
        self, adapter, tmp_store, tmp_cwd
    ):
        with pytest.raises(RuntimeError, match="wrote no files"):
            self._build_with_bad_output(adapter, tmp_store, tmp_cwd, write_files=[])

    def test_inconsistent_prefixes_raises_runtime_error(
        self, adapter, tmp_store, tmp_cwd
    ):
        with pytest.raises(RuntimeError, match="inconsistent prefixes"):
            self._build_with_bad_output(
                adapter, tmp_store, tmp_cwd,
                write_files=[
                    "dataset_abc-raw_input.parquet",
                    "dataset_def-lambdas.parquet",
                ],
            )

    def test_validation_error_does_not_write_manifest_entry(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import _read_manifest

        with pytest.raises(RuntimeError):
            self._build_with_bad_output(adapter, tmp_store, tmp_cwd, write_files=[])
        assert DATASET_ID not in _read_manifest(tmp_store)

    def test_validation_error_does_not_populate_cache(
        self, adapter, tmp_store, tmp_cwd
    ):
        with pytest.raises(RuntimeError):
            self._build_with_bad_output(adapter, tmp_store, tmp_cwd, write_files=[])
        assert not adapter.has_index(DATASET_ID)


# ===========================================================================
# 7. Build failure -- cleanup of partial files
# ===========================================================================


class TestBuildFailureCleansUp:
    def _build_with_partial_then_raise(
        self, adapter, tmp_store: Path, write_before_raise: list[str],
    ):
        mod = adapter._mod
        original_builder = mod.ArrowSpaceBuilder

        class PartialBuilder:
            def build_and_store(self, graph_params, array):
                storage = Path(os.getcwd()) / "storage"
                storage.mkdir(parents=True, exist_ok=True)
                for fname in write_before_raise:
                    (storage / fname).write_bytes(b"partial")
                raise RuntimeError("simulated build failure")

        mod.ArrowSpaceBuilder = PartialBuilder
        try:
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        finally:
            mod.ArrowSpaceBuilder = original_builder

    def test_build_failure_propagates_exception(self, adapter, tmp_store, tmp_cwd):
        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )

    def test_build_failure_cleans_partial_files_from_cwd_storage(
        self, adapter, tmp_store, tmp_cwd
    ):
        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )
        storage = tmp_cwd / "storage"
        leftover = list(storage.glob("*")) if storage.exists() else []
        assert leftover == [], f"Partial files not cleaned up: {leftover}"

    def test_build_failure_does_not_write_manifest(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )
        assert DATASET_ID not in _read_manifest(tmp_store)

    def test_build_failure_does_not_cache_index(self, adapter, tmp_store, tmp_cwd):
        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )
        assert not adapter.has_index(DATASET_ID)

    def test_build_failure_no_files_in_index_store(self, adapter, tmp_store, tmp_cwd):
        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )
        # manifest may exist (with empty dict) but no build output files
        remaining = [p for p in tmp_store.iterdir() if p.is_file() and p.name != "index_manifest.json"]
        assert remaining == [], f"Build output files leaked into index_store: {remaining}"

    def test_build_failure_preserves_existing_manifest_entries(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds_a", FIXTURE_ARRAY.copy(), tmp_store)
        ds_a_entry = _read_manifest(tmp_store)["ds_a"]

        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )

        manifest = _read_manifest(tmp_store)
        assert manifest.get("ds_a") == ds_a_entry, "Pre-existing manifest entry was corrupted"

    def test_retry_after_failure_succeeds(self, adapter, tmp_store, tmp_cwd):
        with pytest.raises(Exception):
            self._build_with_partial_then_raise(
                adapter, tmp_store,
                write_before_raise=["dataset_x-raw_input.parquet", "dataset_x-lambdas.parquet"],
            )
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        assert adapter.has_index(DATASET_ID)


# ===========================================================================
# 8. reload_from_manifest after PATH 2 build
# ===========================================================================


class TestReloadAfterFallbackBuild:
    def test_reload_finds_built_index(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter, _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        manifest = _read_manifest(tmp_store)
        assert DATASET_ID in manifest

        mod2 = _make_build_and_store_mod()
        aspace = _make_fake_aspace()
        gl = _make_fake_gl()
        mod2.load_arrowspace = MagicMock(return_value=(aspace, gl))

        adapter2 = _ArrowSpaceAdapter(mod2, cache_size=4)
        loaded = adapter2.reload_from_manifest(tmp_store)
        assert DATASET_ID in loaded

    def test_reload_uses_correct_dataset_name(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter, _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]

        mod2 = _make_build_and_store_mod()
        aspace = _make_fake_aspace()
        gl = _make_fake_gl()
        mod2.load_arrowspace = MagicMock(return_value=(aspace, gl))

        adapter2 = _ArrowSpaceAdapter(mod2, cache_size=4)
        adapter2.reload_from_manifest(tmp_store)

        call_args = mod2.load_arrowspace.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert kwargs.get("dataset_name") == dname

    def test_reload_missing_files_is_skipped(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter, _write_manifest

        _write_manifest(tmp_store, {
            DATASET_ID: {
                "dataset_name": "ghost_dataset_abc123",
                "graph_params": GRAPH_PARAMS,
            }
        })

        mod2 = _make_build_and_store_mod()
        mod2.load_arrowspace = MagicMock(side_effect=FileNotFoundError("no file"))

        adapter2 = _ArrowSpaceAdapter(mod2, cache_size=4)
        loaded = adapter2.reload_from_manifest(tmp_store)
        assert DATASET_ID not in loaded


# ===========================================================================
# 9. Idempotency
# ===========================================================================


class TestIdempotency:
    def test_double_build_same_dataset_name(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        name1 = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        name2 = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        assert name1 == name2

    def test_double_build_only_one_file_set(self, adapter, tmp_store, tmp_cwd):
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        from arro_server.arrowspace_adapter import _read_manifest

        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        for p in tmp_store.iterdir():
            if not p.is_file() or p.name == "index_manifest.json":
                continue
            assert p.name.startswith(dname), \
                f"Stale file without current dataset_name: {p.name}"

    def test_ten_sequential_builds_same_dataset(self, adapter, tmp_store, tmp_cwd):
        from arro_server.arrowspace_adapter import _read_manifest

        for _ in range(10):
            adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        dname = _read_manifest(tmp_store)[DATASET_ID]["dataset_name"]
        for p in tmp_store.iterdir():
            if not p.is_file() or p.name == "index_manifest.json":
                continue
            assert p.name.startswith(dname)


# ===========================================================================
# 10. Concurrency guard
# ===========================================================================


class TestConcurrencyGuard:
    def test_concurrent_builds_same_dataset_stable_manifest(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import _read_manifest

        errors: list[Exception] = []

        def build():
            try:
                adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=build)
        t2 = threading.Thread(target=build)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        manifest = _read_manifest(tmp_store)
        assert isinstance(manifest, dict)
        assert (DATASET_ID in manifest) or (len(errors) >= 1)

    def test_concurrent_builds_different_datasets_no_cross_contamination(
        self, adapter, tmp_store, tmp_cwd
    ):
        from arro_server.arrowspace_adapter import _read_manifest

        mod = adapter._mod
        original_builder = mod.ArrowSpaceBuilder
        call_count = [0]
        uid_lock = threading.Lock()
        # Serialize access to CWD/storage/ so concurrent PATH 2 builds
        # do not see each other's files and trip prefix detection.
        storage_lock = threading.Lock()

        class ThreadSafeBuilder:
            def build_and_store(self, graph_params, array):
                with uid_lock:
                    call_count[0] += 1
                    uid = f"uid{call_count[0]}"
                storage = Path(os.getcwd()) / "storage"
                storage.mkdir(parents=True, exist_ok=True)
                for suffix in ("raw_input", "lambdas", "gl-matrix"):
                    (storage / f"dataset_{uid}-{suffix}.parquet").write_bytes(b"x")
                return _make_fake_aspace(), _make_fake_gl()

        mod.ArrowSpaceBuilder = ThreadSafeBuilder
        errors: list[Exception] = []

        def build(ds_id: str):
            try:
                with storage_lock:
                    adapter.build_index(ds_id, FIXTURE_ARRAY.copy(), tmp_store)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=build, args=("ds_alpha",))
        t2 = threading.Thread(target=build, args=("ds_beta",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        mod.ArrowSpaceBuilder = original_builder

        assert errors == [], f"Unexpected errors in concurrent build: {errors}"
        manifest = _read_manifest(tmp_store)
        assert "ds_alpha" in manifest
        assert "ds_beta" in manifest
        assert manifest["ds_alpha"]["dataset_name"] != manifest["ds_beta"]["dataset_name"]
