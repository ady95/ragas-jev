import pytest

from helpers import noul_decision
from ragas_jev.lang import detect_lang
from ragas_jev.scoring.calibration import Calibrator, IsotonicMap, fit_isotonic


def test_isotonic_fit_is_monotone_and_matches_bin_rates():
    ps = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9, 0.95]
    ys = [0, 0, 1, 0, 1, 1, 1]
    m = fit_isotonic(ps, ys, clip=0.0)
    assert list(m.ys) == sorted(m.ys)
    assert m(0.05) == 0.0 and m(0.99) == 1.0
    # the violating pair (0.3 -> 1, 0.6 -> 0) is pooled to 0.5
    assert m(0.45) == pytest.approx(0.5)


def test_clip_keeps_calibrated_values_off_the_extremes():
    m = fit_isotonic([0.1, 0.9], [0, 1], clip=0.01)
    assert m(0.0) == 0.01 and m(1.0) == 0.99


def test_interpolation_between_breakpoints():
    m = IsotonicMap((0.2, 0.8), (0.1, 0.7))
    assert m(0.5) == pytest.approx(0.4)


def test_calibrator_rewrites_p_and_confidence_but_keeps_jev_p(tmp_path):
    # JEV says 0.6 for chunks that are relevant only 10% of the time
    cal = Calibrator({"test.v1:ko": IsotonicMap((0.0, 0.6, 1.0), (0.01, 0.1, 0.9))}, source="unit-test")
    d = cal.apply(noul_decision("chunk_0", 0.6), "ko")
    assert d.p == pytest.approx(0.1)
    assert d.confidence == pytest.approx(0.9)
    assert d.jev_p == 0.6 and d.calibrated
    # no map for this language and no wildcard: unchanged
    assert cal.apply(noul_decision("chunk_0", 0.6), "en").p == 0.6
    path = tmp_path / "cal.json"
    cal.save(path)
    assert Calibrator.load(path).lookup("test.v1", "ko")(0.6) == pytest.approx(0.1)


def test_wildcard_language_fallback():
    cal = Calibrator({"test.v1:*": IsotonicMap((0.0, 1.0), (0.2, 0.8))})
    assert cal.apply(noul_decision("c", 0.5), "en").p == pytest.approx(0.5)


@pytest.mark.parametrize(("text", "lang"), [("그리스의 수도는 어디인가요?", "ko"), ("What is the capital?", "en"), ("", "en"), ("COVID-19 백신", "ko")])
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang
