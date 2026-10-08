
import sys

from engine.scanner import run_scan


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 1
    print(f"{'Файл':40} {'Формат':12} {'Размер, px':>12} {'DPI':>14} {'Бит':>4}  {'Сжатие':32} Статус")

    def show(batch):
        for m in batch:
            print(f"{m.name[:40]:40} {m.format or '—':12} {m.size_text:>12} {m.dpi_text:>14} "
                  f"{m.bpp_text:>4}  {(m.compression or '—')[:32]:32} {m.status}"
                  + (f"  [{m.message}]" if m.message else ""))

    s = run_scan(argv, on_batch=show)
    print(f"\nФайлов: {s.processed}; OK {s.ok}, предупреждений {s.warnings}, повреждено {s.corrupt}, "
          f"прочее {s.other}; прочитано {s.bytes_read} из {s.bytes_total} байт; {s.elapsed:.2f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
