from thinking_buildings.display import Display, COLOR_RED, COLOR_ORANGE, COLOR_GREEN
from thinking_buildings.config import DisplayConfig


def _display() -> Display:
    return Display(DisplayConfig(show_window=False, box_color=(0, 255, 0)))


class TestBoxColor:
    def test_no_identity_returns_default(self, make_detection):
        d = _display()
        det = make_detection(identity=None)
        assert d._box_color(det) == (0, 255, 0)

    def test_occluded_returns_red(self, make_detection):
        d = _display()
        det = make_detection(identity="occluded_face", occlusion=True)
        assert d._box_color(det) == COLOR_RED

    def test_unknown_returns_orange(self, make_detection):
        d = _display()
        det = make_detection(identity="unknown_person")
        assert d._box_color(det) == COLOR_ORANGE

    def test_known_returns_green(self, make_detection):
        d = _display()
        det = make_detection(identity="alice")
        assert d._box_color(det) == COLOR_GREEN


class TestBuildLabel:
    def test_known_person(self, make_detection):
        d = _display()
        det = make_detection(identity="alice", confidence=0.91)
        label = d._build_label(det)
        assert "alice" in label

    def test_unknown_person(self, make_detection):
        d = _display()
        det = make_detection(identity="unknown_person", confidence=0.8)
        label = d._build_label(det)
        assert "unknown" in label

    def test_occluded(self, make_detection):
        d = _display()
        det = make_detection(identity="occluded_face", occlusion=True, confidence=0.75)
        label = d._build_label(det)
        assert "OCCLUDED" in label

    def test_non_person(self, make_detection):
        d = _display()
        det = make_detection(label="cat", identity=None, confidence=0.88)
        label = d._build_label(det)
        assert "cat" in label
