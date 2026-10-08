from src.hardware import PeakMemory, host_summary


def test_host_and_peak_rss():
    info = host_summary()
    assert info["ram_gb"] > 0
    assert info["python"]
    assert info["gpu"] is None
    peak = PeakMemory(interval_s=0.05)
    peak.start()
    _ = [0] * 1000
    gb = peak.stop()
    assert gb > 0
