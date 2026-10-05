"""
Tüm klipler için IMU özelliklerini ve RAFT ground truth'unu önbelleğe alır (data/cache/).
`train` ve `test` bunu otomatik çağırır; ayrıca çalıştırmak zorunlu değildir.
"""
import time

from pipeline import dataset


def prepare_clips(clips, need_gt=True, device=None, bf16=False, force=False):
    """Eksik önbellekleri üretir. RAFT modeli yalnızca gerçekten gerekiyorsa yüklenir."""
    todo = [c for c in clips if force or (need_gt and not dataset.has_gt_cache(c))]
    raft = None
    if need_gt and todo:
        from pipeline.raft_flow import RaftGlobalMotion
        raft = RaftGlobalMotion(device=device, bf16=bf16)
        print("RAFT cihazı: {}{}  | hesaplanacak klip: {}".format(
            raft.device, " (bf16)" if raft.bf16 else "", len(todo)))

    out = {}
    for i, clip in enumerate(clips):
        cached = not (force or (need_gt and not dataset.has_gt_cache(clip)))
        t0 = time.time()
        if not cached:
            print("  [{}/{}] {} -> RAFT ground truth hesaplanıyor...".format(i + 1, len(clips), clip["clip_id"]))
        out[clip["clip_id"]] = dataset.prepare_clip(clip, raft=raft, need_gt=need_gt, force=force)
        if not cached:
            print("      tamam ({:.1f} sn)".format(time.time() - t0))
    return out
