"""PyCharm'da sağ tık -> Run ile çalıştırılabilir kısayol:  python main.py test"""
import sys

from main import main

if __name__ == "__main__":
    main(["test"] + sys.argv[1:])
