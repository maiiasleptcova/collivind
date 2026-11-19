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


class BatchPartiallyStoredError(CollivindError):
    """A batch failed in storage after some items were already written (#37).

    Validation runs over the whole batch before any write, so this is only
    raised for backend failures mid-batch. `stored_ids` tells the caller what
    landed, so a retry can skip those instead of duplicating them.
    """

    def __init__(self, stored_ids: list, failed_index: int, cause: Exception):
        super().__init__(
            f"memory[{failed_index}] failed to store: {cause}. "
            f"{len(stored_ids)} earlier item(s) were stored; retry only from index {failed_index}."
        )
        self.stored_ids = list(stored_ids)
        self.failed_index = failed_index
