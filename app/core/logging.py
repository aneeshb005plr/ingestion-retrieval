"""
Structured logging configuration using structlog.

In DEVELOPMENT: colored, human-readable output with timestamps
In PRODUCTION:  JSON output, one line per event — parseable by log aggregators

Usage anywhere in the codebase:
    import structlog
    log = structlog.get_logger()
    log.info("job.created", job_id="abc", tenant_id="docassist")

To bind context for the lifetime of a request/task:
    log = structlog.get_logger().bind(job_id=job_id, tenant_id=tenant_id)
    log.info("ingestion.started")   # job_id + tenant_id on every line after this
"""

import logging
import sys
import structlog
from app.core.config import settings


# Keys that should never appear in log output — redacted automatically
_SENSITIVE_KEYS = frozenset(
    {
        "client_secret",
        "password",
        "api_key",
        "token",
        "authorization",
        "connection_string",
        "mongo_uri",
        "redis_uri",
    }
)


def _scrub_sensitive_data(logger: object, method: str, event_dict: dict) -> dict:
    """
    Structlog processor that redacts sensitive values before output.
    Runs on every log event — ensures credentials never appear in logs.
    """
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***REDACTED***"
    return event_dict


def setup_logging() -> None:
    """
    Call once at application startup (in lifespan).
    Configures both structlog and stdlib logging to work together.
    """
    is_production = settings.ENVIRONMENT == "production"

    # Shared processors that run on every log event regardless of environment
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # thread-local context
        structlog.stdlib.add_log_level,  # adds "level": "info"
        structlog.stdlib.add_logger_name,  # adds "logger": "app.db.mongo"
        structlog.processors.TimeStamper(fmt="iso"),  # adds "timestamp": "2026-..."
        _scrub_sensitive_data,  # redact secrets
        structlog.processors.StackInfoRenderer(),  # stack traces as structured data
    ]
    if is_production:
        # JSON output — one line per event, parseable by any log aggregator
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,  # exceptions as dicts not strings
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Colored, human-readable output for local development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Make stdlib logging (from FastAPI, uvicorn, pymongo) also go through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO if is_production else logging.DEBUG,
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("arq").setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
