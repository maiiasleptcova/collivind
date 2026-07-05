"""
Core exceptions for Collivind.
"""


class CollivindError(Exception):
    """Base exception for all Collivind errors."""

    pass


class ConfigurationError(CollivindError):
    """Raised when there is an issue with the configuration."""

    pass


class DockerExecutionError(CollivindError):
    """Raised when a Docker command fails to execute properly."""

    pass


class HealthCheckFailedError(CollivindError):
    """Raised when services do not become healthy in time."""

    pass


class StorageUnavailableError(CollivindError):
    """Raised when a storage backend is not reachable."""

    pass
