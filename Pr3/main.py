import os
import sys

if getattr(sys, "frozen", False):
    os.environ.setdefault("NUMBA_CACHE_DIR",
                          os.path.join(os.path.expanduser("~"), ".lab3_numba_cache"))

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Лабораторная работа 3 (вариант 4)")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
