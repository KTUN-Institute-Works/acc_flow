"""PyCharm'da sağ tık -> Run ile çalıştırılabilir kısayol:  python main.py train"""
import sys

from main import main

if __name__ == "__main__":
    main(["train"] + sys.argv[1:])
