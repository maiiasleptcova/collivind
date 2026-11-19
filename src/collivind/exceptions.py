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


class GraphStoreUnavailableError(StorageUnavailableError):
    """A write needed the graph store and it did not answer (#39).

    Reads degrade to vector-only search; writes do not, because the graph is
    the record of truth and a vector-only write would be invisible to it.
    """

    def __init__(self, cause: Exception, outcome: str = "the memory was NOT saved"):
        super().__init__(
            f"Graph store is unavailable, so {outcome}: {cause}. "
            "Run `collivind status` to check it (docker mode: is the neo4j container restarting?)."
        )


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
