"""Unit tests for the configuration loader."""

from __future__ import annotations

from football_ai.config import settings


class TestSettings:
    def test_project_name_loaded(self) -> None:
        assert settings.project_name == "Football AI Platform"

    def test_version_format(self) -> None:
        parts = settings.version.split(".")
        assert len(parts) == 3

    def test_pitch_dimensions(self) -> None:
        assert settings.pitch.length == 120.0
        assert settings.pitch.width == 80.0

    def test_zones_positive(self) -> None:
        assert settings.pitch.zones_x > 0
        assert settings.pitch.zones_y > 0

    def test_possession_chain_bounds(self) -> None:
        assert settings.possession.min_chain_length >= 2
        assert settings.possession.max_chain_length >= settings.possession.min_chain_length

    def test_vaep_k_positive(self) -> None:
        assert settings.vaep.k_actions > 0

    def test_root_dir_exists(self) -> None:
        assert settings.root_dir.exists()

    def test_abs_path_helper(self) -> None:
        p = settings.abs(settings.paths.data_raw)
        # Should return a Path object
        from pathlib import Path
        assert isinstance(p, Path)
