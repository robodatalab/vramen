import abc
from dataclasses import dataclass, field
import gc
import multiprocessing
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event
import queue
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

from servery import log
from servery.utils import (
    gpu_memory_limit,
    gpu_memory_used,
    gpu_tensors,
    process_memory,
)
import torch
from transformers import AutoTokenizer


_log = log.logger(__name__)
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

REQUEST_POLL_S = 0.1
STOP_GRACE_S = 30.0
Model = Any


class ModelKind(abc.ABC):

    def __init__(
        self,
        model_id: str,
        manager: "InferenceModelResourceManager",
        mem_required_gb: float,
    ) -> None:
        self.model_id = model_id
        self.manager = manager
        self.mem_required_gb = mem_required_gb

    def __eq__(self, model: Any) -> bool:
        if not isinstance(model, ModelKind):
            return False
        return model.model_id == self.model_id

    def __getstate__(self) -> dict[str, Any]:
        """What crosses into the serving process: the model, not the manager.

        The manager stays on this side. It holds the locks and the queues of
        every resident, none of which survive being pickled, and none of which
        mean anything to a process that only has to load one model and answer.
        """
        return {**self.__dict__, "manager": None}

    @abc.abstractmethod
    def load(self) -> tuple[Model, AutoTokenizer]:
        pass


class ModelNotAvailable(Exception):
    pass


def _log_memory_usage(kind: ModelKind) -> None:
    _log.info(
            "serving %s: tensors %.1fGB, driver %.1fGB, limit %.1fGB",
            kind.model_id,
            gpu_tensors(),
            gpu_memory_used(),
            gpu_memory_limit(),
        )


@dataclass
class MemoryReading:
    """What the serving processes were holding, the last time they said so."""

    gpu_used: float
    gpu_limit: float
    process: float


@dataclass
class _Resident:
    kind: ModelKind
    requests: Queue
    replies: Queue
    # Readings the serving process volunteers, on their own queue: asking for
    # one would sit behind a generation that runs for minutes.
    readings: Queue
    stop: Event
    process: Process
    # Loading takes minutes and cannot hold the quota table, so a second caller
    # for the same model waits here rather than starting a twin.
    ready: threading.Event = field(default_factory=threading.Event)
    failure: str | None = None
    # The replies carry no request ids, so only one caller may be in flight.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Above zero the model is answering someone, and cannot be evicted.
    leases: int = 0
    last_used: float = 0.0
    last_reading: MemoryReading | None = None


class InferenceModelResourceManager:
    """Serving processes, held side by side under a fixed memory quota.

    Models stay resident for as long as their declared sizes fit the quota
    together; past that, the least recently used of the idle ones give up their
    processes to make room. The quota is spent in declarations — what the
    processes report of themselves is read back out, never counted against it.
    """

    def __init__(self, quota_gb: float) -> None:
        self._quota_gb = quota_gb
        self._table = threading.Lock()
        self._released = threading.Condition(self._table)
        self._residents: dict[str, _Resident] = {}

    @property
    def residents(self) -> list[ModelKind]:
        """The models loaded right now, if any."""
        with self._table:
            return [resident.kind for resident in self._residents.values()]

    def memory(self) -> MemoryReading:
        """What the processes holding the models last reported, added up.

        With no model loaded there are no such processes, so this one answers
        for itself — the reading stays continuous either way, and reads as the
        near nothing the server holds on its own.
        """
        with self._table:
            if not self._residents:
                return _memory_reading()
            readings = [self._reading(r) for r in self._residents.values()]
            gpu_used = sum(reading.gpu_used for reading in readings)
            process = sum(reading.process for reading in readings)
        # A limit belongs to the machine rather than to any one process, so it
        # is the one figure here that is not a sum.
        return MemoryReading(gpu_used, gpu_memory_limit(), process)

    def shutdown(self) -> None:
        """Stop every serving process and hand the whole quota back."""
        with self._table:
            for resident in list(self._residents.values()):
                self._evict(resident)
            self._released.notify_all()

    @contextmanager
    def residency(
        self,
        kind: ModelKind,
    ) -> Generator[tuple[Queue, Queue], None, None]:
        resident = self._admit(kind)
        try:
            with resident.lock:
                yield resident.requests, resident.replies
        finally:
            self._release(resident)

    def _admit(self, kind: ModelKind) -> _Resident:
        """A process serving this model, leased to the caller until it is done."""
        if kind.mem_required_gb > self._quota_gb:
            raise ModelNotAvailable(
                f"{kind.model_id} wants {kind.mem_required_gb:.1f}GB of a "
                f"{self._quota_gb:.1f}GB quota"
            )

        with self._table:
            while True:
                resident = self._residents.get(kind.model_id)
                if resident is not None:
                    resident.leases += 1
                    ours = False
                    break
                if self._make_room(kind):
                    resident = self._reserve(kind)
                    ours = True
                    break
                # Every resident is answering someone. One of them will finish.
                self._released.wait()

        if ours:
            self._load(resident)
        else:
            resident.ready.wait()

        if resident.failure is not None:
            # It left the table on the way down; the lease is on nothing.
            raise ModelNotAvailable(
                f"{kind.model_id} did not load: {resident.failure}"
            )
        return resident

    def _release(self, resident: _Resident) -> None:
        with self._table:
            resident.leases -= 1
            resident.last_used = time.monotonic()
            self._released.notify_all()

    def _free_gb(self) -> float:
        claimed = sum(r.kind.mem_required_gb for r in self._residents.values())
        return self._quota_gb - claimed

    def _make_room(self, kind: ModelKind) -> bool:
        """Clear the model's share of the quota, oldest idle resident first."""
        if self._free_gb() >= kind.mem_required_gb:
            return True
        idle = sorted(
            (r for r in self._residents.values() if r.leases == 0),
            key=lambda r: r.last_used,
        )
        for resident in idle:
            _log.info(
                "evicting %s to make room for %s",
                resident.kind.model_id,
                kind.model_id,
            )
            self._evict(resident)
            if self._free_gb() >= kind.mem_required_gb:
                return True
        return False

    def _reserve(self, kind: ModelKind) -> _Resident:
        """Take the model's share of the quota, before its process exists.

        The claim is entered first so that the load, which runs for minutes, can
        happen with the table free for everybody else.
        """
        requests: Queue = Queue()
        replies: Queue = Queue()
        readings: Queue = Queue()
        stop = multiprocessing.Event()

        process = Process(
            target=_serving_process_main,
            kwargs=dict(
                requests=requests,
                replies=replies,
                readings=readings,
                kind=kind,
                stop=stop,
            ),
            daemon=True,
        )

        resident = _Resident(
            kind, requests, replies, readings, stop, process, leases=1
        )
        self._residents[kind.model_id] = resident
        return resident

    def _load(self, resident: _Resident) -> None:
        resident.process.start()

        error, _ = resident.replies.get()
        if error is not None:
            resident.failure = error
            resident.process.join()
            with self._table:
                self._residents.pop(resident.kind.model_id, None)
                self._released.notify_all()

        resident.ready.set()

    def _evict(self, resident: _Resident) -> None:
        resident.stop.set()
        resident.process.join(timeout=STOP_GRACE_S)

        if resident.process.is_alive():
            # Whatever it is doing, its memory is needed now.
            _log.warning("%s did not stop; killing it", resident.kind.model_id)
            resident.process.terminate()
            resident.process.join()

        _log_memory_usage(resident.kind)

        resident.requests.close()
        resident.replies.close()
        resident.readings.close()
        self._residents.pop(resident.kind.model_id, None)

    def _reading(self, resident: _Resident) -> MemoryReading:
        while True:
            try:
                resident.last_reading = resident.readings.get_nowait()
            except queue.Empty:
                return resident.last_reading or MemoryReading(0.0, 0.0, 0.0)


def _memory_reading() -> MemoryReading:
    return MemoryReading(gpu_memory_used(), gpu_memory_limit(), process_memory())


def _serving_process_main(
    requests: Queue,
    replies: Queue,
    readings: Queue,
    kind: ModelKind,
    stop: Event,
) -> None:
    """Child process: hold the model and answer completions until it is killed."""
    log.setup()

    _log.info("Preparing %s for serving ...", kind.model_id)
    try:
        model, tokenizer = kind.load()
    except Exception as err:
        replies.put((str(err), None))
        return

    _log_memory_usage(kind)
    readings.put(_memory_reading())
    replies.put((None, None))
    _log.info("Serving %s", kind.model_id)

    while not stop.is_set():
        try:
            request = requests.get(timeout=REQUEST_POLL_S)
        except queue.Empty:
            readings.put(_memory_reading())
            continue
        try:
            reply = request(model, tokenizer)
            replies.put((None, reply))
        except Exception as err:
            replies.put((str(err), None))

    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    _log_memory_usage(kind)
    _log.info("Stopped serving %s", kind.model_id)
