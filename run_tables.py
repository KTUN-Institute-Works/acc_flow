"""PyCharm'da sağ tık -> Run: kayıtlı test/eğitim sonuçlarından tez tablolarını (TR + EN) yeniden üretir.
Eğitim, test veya RAFT çalıştırmaz; birkaç saniye sürer.
Çıktı: results/multidomain/tables/{tr,en}/  (*.md, *.tex, *.csv, all_tables.tex, all_tables.md)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.tables import make_all_tables  # noqa: E402

if __name__ == "__main__":
    make_all_tables()
