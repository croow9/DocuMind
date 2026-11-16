from enum import StrEnum


class IngestionTopics(StrEnum):
    REQUESTED = "documents.ingestion.requested"
    DLQ = "documents.ingestion.dlq"
