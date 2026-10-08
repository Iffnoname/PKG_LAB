"""Поиск файлов и многопоточное чтение метаданных (без зависимости от Qt — удобно тестировать)."""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional

from models import (ImageMetadata, STATUS_CORRUPT, STATUS_ERROR, STATUS_NOT_IMAGE, STATUS_OK,
                    STATUS_WARNING)
from parsers import FORMAT_BY_EXT, inspect_file

IMAGE_EXTENSIONS = frozenset(FORMAT_BY_EXT)
MAX_FILES = 100_000
DEFAULT_WORKERS = 4
CHUNK_SIZE = 48                 # файлов в одной задаче пула
HIDDEN_ATTRIBUTE = 0x2          # FILE_ATTRIBUTE_HIDDEN (Windows)


@dataclass
class ScanSummary:
    total_found: int = 0
    processed: int = 0
    ok: int = 0
    warnings: int = 0
    corrupt: int = 0
    other: int = 0
    bytes_read: int = 0
    bytes_total: int = 0
    elapsed: float = 0.0
    cancelled: bool = False
    limit_reached: bool = False
    skipped_dirs: int = 0


class _Walk:
    """Состояние обхода (чтобы вернуть флаги вместе с генератором)."""
    def __init__(self) -> None:
        self.limit_reached = False
        self.skipped_dirs = 0


def _is_hidden(entry: os.DirEntry) -> bool:
    if entry.name.startswith("."):
        return True
    try:
        return bool(getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0) & HIDDEN_ATTRIBUTE)
    except OSError:
        return False


def _is_link(entry: os.DirEntry) -> bool:
    if entry.is_symlink():
        return True
    is_junction = getattr(entry, "is_junction", None)
    return bool(is_junction and is_junction())


def iter_image_files(sources: Iterable[str], cancel: threading.Event, limit: int, walk: _Walk
                     ) -> Iterator[str]:
    """Файлы из списка источников. Папки обходятся рекурсивно; скрытые объекты и ссылки пропускаются."""
    count = 0
    for src in sources:
        if cancel.is_set():
            return
        if os.path.isfile(src):                     # явно выбранный файл берём как есть
            if count >= limit:
                walk.limit_reached = True
                return
            count += 1
            yield src
            continue
        stack = [src]
        while stack:
            if cancel.is_set():
                return
            folder = stack.pop()
            try:
                with os.scandir(folder) as it:
                    entries = sorted(it, key=lambda e: e.name.lower())
            except OSError:
                walk.skipped_dirs += 1
                continue
            subdirs = []
            for e in entries:
                try:
                    if _is_hidden(e) or _is_link(e):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        subdirs.append(e.path)
                    elif e.is_file(follow_symlinks=False) and \
                            os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS:
                        if count >= limit:
                            walk.limit_reached = True
                            return
                        count += 1
                        yield e.path
                except OSError:
                    continue
            stack.extend(reversed(subdirs))


def _process_chunk(paths: List[str], cancel: threading.Event) -> List[ImageMetadata]:
    out = []
    for p in paths:
        if cancel.is_set():
            break
        out.append(inspect_file(p))
    return out


def run_scan(
    sources: Iterable[str],
    cancel: Optional[threading.Event] = None,
    on_found: Optional[Callable[[int], None]] = None,
    on_batch: Optional[Callable[[List[ImageMetadata]], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    workers: int = DEFAULT_WORKERS,
    limit: int = MAX_FILES,
    chunk_size: int = CHUNK_SIZE,
) -> ScanSummary:
    """Блокирующая функция: запускать в фоновом потоке.

    1) быстрый обход каталогов -> список путей (on_found сообщает «найдено N»);
    2) пул из `workers` потоков читает метаданные порциями; результаты отдаются в исходном порядке
       через on_batch, прогресс — через on_progress(done, total).
    """
    cancel = cancel or threading.Event()
    summary = ScanSummary()
    t0 = time.perf_counter()

    walk = _Walk()
    paths: List[str] = []
    for p in iter_image_files(sources, cancel, limit, walk):
        paths.append(p)
        if on_found and len(paths) % 500 == 0:
            on_found(len(paths))
    summary.total_found = len(paths)
    summary.limit_reached = walk.limit_reached
    summary.skipped_dirs = walk.skipped_dirs
    if on_found:
        on_found(len(paths))

    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
    pending: deque = deque()
    next_chunk = 0
    done = 0
    window = max(2, workers * 3)       # ограничение числа задач «в полёте»

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="meta") as pool:
        while True:
            while next_chunk < len(chunks) and len(pending) < window and not cancel.is_set():
                pending.append(pool.submit(_process_chunk, chunks[next_chunk], cancel))
                next_chunk += 1
            if not pending:
                break
            batch = pending.popleft().result()
            for m in batch:
                summary.bytes_read += m.bytes_read
                summary.bytes_total += m.file_size
                if m.status == STATUS_OK:
                    summary.ok += 1
                elif m.status == STATUS_WARNING:
                    summary.warnings += 1
                elif m.status == STATUS_CORRUPT:
                    summary.corrupt += 1
                else:
                    summary.other += 1
            done += len(batch)
            summary.processed = done
            if on_batch and batch:
                on_batch(batch)
            if on_progress:
                on_progress(done, len(paths))

    summary.cancelled = cancel.is_set()
    summary.elapsed = time.perf_counter() - t0
    return summary
