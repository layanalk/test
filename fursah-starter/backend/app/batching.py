"""Bounded, disposable search sessions. No persistence or LangGraph checkpoints."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from time import monotonic
from uuid import uuid4

from .infrastructure.errors import BatchExpiredError
from .internal import SearchPlan
from .schemas import DiscoveryRequest, SearchConstraints


def input_fingerprint(request: DiscoveryRequest) -> str:
    data = request.model_dump(mode='json', exclude={'request_id', 'batch_token', 'batch_index'})
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class SearchSession:
    fingerprint: str
    plan: SearchPlan
    constraints: SearchConstraints
    filters: tuple[str, ...]
    ordered: tuple[tuple[str, str, int], ...]


class SearchBatchCache:
    def __init__(self, *, capacity=64, ttl_seconds=600, clock=monotonic):
        if capacity < 1 or ttl_seconds <= 0:
            raise ValueError('Search cache bounds must be positive')
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries = OrderedDict()
        self._lock = Lock()

    def _expire(self):
        now = self.clock()
        for token, (expires, _) in list(self._entries.items()):
            if expires <= now:
                del self._entries[token]

    def create(self, request, plan, constraints, filters, ordered) -> str:
        session = SearchSession(input_fingerprint(request), plan.model_copy(deep=True),
            constraints.model_copy(deep=True), tuple(filters), tuple(ordered))
        with self._lock:
            self._expire()
            while len(self._entries) >= self.capacity:
                self._entries.popitem(last=False)
            token = uuid4().hex
            self._entries[token] = (self.clock() + self.ttl_seconds, session)
            return token

    def get(self, request: DiscoveryRequest) -> SearchSession:
        with self._lock:
            self._expire()
            entry = self._entries.get(request.batch_token)
            if entry is None or entry[1].fingerprint != input_fingerprint(request):
                raise BatchExpiredError()
            return entry[1]
