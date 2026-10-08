import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        QTimer.singleShot(0, lambda: window.start_scan(sys.argv[1:]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
