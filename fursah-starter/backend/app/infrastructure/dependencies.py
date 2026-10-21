from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime
from .config import Settings, get_settings
from ..agent import OpenRouterAgent
from ..repository import OpportunityRepository
from .data_package import DatasetLoader
from .tracing import Tracer


@dataclass
class AppDependencies:
    settings: Settings
    bridge: DatasetLoader
    repository: OpportunityRepository
    agent: OpenRouterAgent
    tracer: Tracer
    evaluation_time: datetime | None = None

    def readiness(self) -> tuple[dict[str, bool], dict[str, str]]:
        from ..workflow import application_is_ready
        checks, details = {}, {}
        for name, check in (('storage', self.repository.is_ready),
                            ('application', lambda: application_is_ready(self))):
            try:
                checks[name] = check() is True
            except Exception:
                checks[name] = False
            if not checks[name]:
                details[name] = ('Prepared opportunity data is unavailable or invalid.' if name == 'storage'
                                 else 'Set OPENROUTER_API_KEY in the backend process environment. Readiness does not contact the provider.')
        return checks, details


@lru_cache(maxsize=1)
def get_app_dependencies() -> AppDependencies:
    settings = get_settings()
    bridge = DatasetLoader(settings)
    return AppDependencies(settings=settings, bridge=bridge,
                           repository=OpportunityRepository(bridge),
                           agent=OpenRouterAgent(settings),
                           tracer=Tracer(settings.trace_jsonl_path))
