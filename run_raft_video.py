"""PyCharm'da sağ tık -> Run:  RAFT hedefiyle (modelin öğrenmeye çalıştığı dx, dy, alpha) stabilize edilmiş
referans videoları üretir. RAFT önbelleği yoksa o klip için bir kez hesaplanır."""
import pipeline.device  # noqa: F401  (MPS ayarı)
from pipeline.raft_video import render_raft_reference

# None = config.TEST_CLIPS içindeki tüm test klipleri
CLIPS = ["video_1783343528738_part02"]

if __name__ == "__main__":
    render_raft_reference(clip_ids=CLIPS)
