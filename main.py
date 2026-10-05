"""
ACCFlow — komut satırı giriş noktası / command-line entry point

    python main.py train      # IDA_split/train + Araba/Vapur/Yürüyüş (test klipleri hariç) ile modeli eğitir
    python main.py test       # görülmemiş klipler (config.TEST_CLIPS: 8 IDA + 1 araba + 1 vapur) x 5 durum
    python main.py plot       # kayıtlı test sonuçlarından TR + EN grafikleri yeniden üretir
    python main.py raft-video # RAFT hedefiyle stabilize edilmiş referans videolar (eğitim verisini gözle kontrol)
    python main.py prepare    # (opsiyonel) tüm kliplerin RAFT ground truth önbelleğini önceden hesaplar

PyCharm kısayolları: run_train.py, run_test.py, run_raft_video.py
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pipeline.device  # noqa: F401,E402  (MPS fallback ortam değişkenini torch'tan önce ayarlar)


def _add_common(p):
    p.add_argument("--device", default=None, help="cuda | mps | cpu (varsayılan: otomatik)")
    p.add_argument("--raft-bf16", action="store_true",
                   help="RAFT'ı CPU'da bfloat16 ile çalıştır (~2x hızlı, yalnızca CPU)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACCFlow IMU-Alpha-Net: multi-video train / test")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Modeli IDA_split/train üzerinde eğit")
    _add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)

    p_test = sub.add_parser("test", help="Görülmemiş test klipleri üzerinde 4 durumu değerlendir")
    _add_common(p_test)
    p_test.add_argument("--clips", nargs="*", default=None,
                        help="clip_id listesi (örn. video_1778671955586). Varsayılan: config.TEST_CLIPS")
    p_test.add_argument("--all-test", action="store_true", help="test klasöründeki tüm klipleri kullan")
    p_test.add_argument("--model", default=None, help="model ağırlık dosyası (varsayılan: config.MODEL_PATH)")
    p_test.add_argument("--raft-reference", choices=["cache", "off", "compute"], default="compute",
                        help="RAFT yalnızca karşılaştırma içindir; stabilizasyon her zaman yalnızca IMU ile. "
                             "compute (varsayılan): önbelleği olmayan klipte bir kez hesapla | cache: yalnızca "
                             "önbellek | off")
    p_test.add_argument("--out", default=None,
                        help="sonuç klasörü (varsayılan: config.TEST_RESULTS_DIR)")

    p_plot = sub.add_parser("plot", help="Kayıtlı test sonuçlarından TR/EN grafikleri üret")
    p_plot.add_argument("--dir", default=None, help="test sonuç klasörü (varsayılan: config.TEST_RESULTS_DIR)")

    p_raft = sub.add_parser("raft-video", help="RAFT hedefiyle stabilize edilmiş referans videoları üret")
    _add_common(p_raft)
    p_raft.add_argument("--split", choices=["train", "test", "all"], default="test",
                        help="test = config.TEST_CLIPS (varsayılan), train/all = o bölümdeki tüm klipler")
    p_raft.add_argument("--clips", nargs="*", default=None, help="belirli clip_id listesi")

    p_prep = sub.add_parser("prepare", help="RAFT ground truth önbelleğini önceden hesapla")
    _add_common(p_prep)
    p_prep.add_argument("--split", choices=["train", "test", "all"], default="all")
    p_prep.add_argument("--force", action="store_true", help="önbelleği yeniden hesapla")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from pipeline.train import train
        train(device=args.device, raft_bf16=args.raft_bf16, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr)
    elif args.cmd == "test":
        from pipeline.test import run_test
        run_test(clip_ids=args.clips, all_test=args.all_test, model_path=args.model,
                 device=args.device, raft_bf16=args.raft_bf16, raft_reference=args.raft_reference,
                 out_dir=args.out)
    elif args.cmd == "plot":
        from pipeline.plots import make_all_plots
        make_all_plots(args.dir)
    elif args.cmd == "raft-video":
        from pipeline.raft_video import render_raft_reference
        render_raft_reference(clip_ids=args.clips, split=args.split, device=args.device, raft_bf16=args.raft_bf16)
    elif args.cmd == "prepare":
        from pipeline import dataset
        from pipeline.prepare import prepare_clips
        from pipeline.dataset import select_test_clips
        if args.split == "train":
            clips = dataset.load_clips("train")
        elif args.split == "test":
            clips = select_test_clips()
        else:
            clips = select_test_clips() + dataset.load_clips("train")
        prepare_clips(clips, need_gt=True, device=args.device, bf16=args.raft_bf16, force=args.force)


if __name__ == "__main__":
    main()
