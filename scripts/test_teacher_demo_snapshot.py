"""CARLA-free snapshot publishing tests for Windows file sharing."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.data_layer import DashboardDataLayer

def main():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "dashboard_snapshot.json"
        DashboardDataLayer.write_snapshot_json({"frame_id": 1}, path)
        assert json.loads(path.read_text())["frame_id"] == 1
        count = [0]
        original = os.replace
        def transient(source, destination):
            count[0] += 1
            if count[0] < 3:
                raise PermissionError(13, "File busy")
            return original(source, destination)
        with patch("dashboard.data_layer.os.replace", side_effect=transient):
            with patch("dashboard.data_layer.time.sleep"):
                DashboardDataLayer.write_snapshot_json({"frame_id": 2}, path)
        assert count[0] == 3
        assert json.loads(path.read_text())["frame_id"] == 2
        assert not list(Path(folder).glob("*.tmp"))
        with patch("dashboard.data_layer.os.replace", side_effect=PermissionError(13, "File busy")):
            with patch("dashboard.data_layer.time.sleep"):
                try:
                    DashboardDataLayer.write_snapshot_json({"frame_id": 3}, path)
                    assert False, "Unreported persistent file lock"
                except PermissionError:
                    pass
        assert json.loads(path.read_text())["frame_id"] == 2
        assert not list(Path(folder).glob("*.tmp"))
    print("SNAPSHOT WINDOWS LOCK TEST PASSED")

if __name__ == "__main__":
    main()
