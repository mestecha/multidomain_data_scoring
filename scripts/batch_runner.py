"""
Async Batch GPT Runner with Semaphore-Based Concurrency Control.

This module provides a modern async architecture for running Azure OpenAI batch jobs
with configurable concurrency limits, proper error handling, and structured logging.

Features:
- Max concurrent batch jobs controlled via asyncio.Semaphore
- Max file size enforcement (200MB per shard by default)
- Process shards from ALL domains concurrently (not domain-by-domain)
- Proper type annotations throughout
- Structured logging with loguru
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import backoff
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncAzureOpenAI, OpenAIError

if TYPE_CHECKING:
    from openai.types import FileObject
    from openai.types.batch import Batch

load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

# Constants
DEFAULT_MAX_CONCURRENT_JOBS: int = 5
DEFAULT_MAX_FILE_SIZE_MB: int = 200
BYTES_PER_MB: int = 1024 * 1024


class JobStatus(str, Enum):
    """Enumeration of possible job states."""

    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FILE_PROCESSING = "file_processing"
    FILE_PROCESSED = "file_processed"
    FILE_FAILED = "file_failed"
    BATCH_CREATED = "batch_created"
    BATCH_RUNNING = "batch_running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    SKIPPED = "skipped"


@dataclass
class ShardJob:
    """
    Tracks the state of a single shard batch job through its lifecycle.

    Attributes:
        file_path: Path to the input JSONL shard file.
        domain: Domain name extracted from the file path.
        status: Current job status.
        file_id: Azure OpenAI file ID after upload.
        batch_id: Azure OpenAI batch job ID.
        batch_response: Final batch response object.
        error_message: Error message if job failed.
        start_time: Timestamp when job processing started.
        end_time: Timestamp when job processing completed.
    """

    file_path: Path
    domain: str
    status: JobStatus = JobStatus.PENDING
    file_id: str | None = None
    batch_id: str | None = None
    batch_response: Batch | None = None
    error_message: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    file_size_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Calculate file size after initialization."""
        if self.file_path.exists():
            self.file_size_bytes = self.file_path.stat().st_size

    @property
    def file_size_mb(self) -> float:
        """Return file size in megabytes."""
        return self.file_size_bytes / BYTES_PER_MB

    @property
    def elapsed_time(self) -> float | None:
        """Return elapsed processing time in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def short_file_id(self) -> str:
        """Return truncated file ID for logging."""
        return self.file_id[:12] if self.file_id else "N/A"

    @property
    def short_batch_id(self) -> str:
        """Return truncated batch ID for logging."""
        return self.batch_id[:12] if self.batch_id else "N/A"

    def is_terminal(self) -> bool:
        """Check if job has reached a terminal state."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELED,
            JobStatus.SKIPPED,
            JobStatus.FILE_FAILED,
        )


@dataclass
class BatchPoolConfig:
    """
    Configuration for the BatchPool.

    Attributes:
        max_concurrent_jobs: Maximum number of concurrent batch jobs.
        max_file_size_mb: Maximum file size in MB per shard.
        file_poll_interval: Seconds between file status polls.
        batch_poll_interval: Seconds between batch status polls.
    """

    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_JOBS
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    file_poll_interval: float = 1.0
    batch_poll_interval: float = 30.0


class BatchPool:
    """
    Manages concurrent batch job execution with semaphore-based concurrency control.

    This class handles the full lifecycle of batch jobs:
    1. File upload to Azure OpenAI
    2. File processing status monitoring
    3. Batch job creation
    4. Batch job status monitoring
    5. Results retrieval

    Attributes:
        config: Pool configuration settings.
        client: Async Azure OpenAI client.
        semaphore: Asyncio semaphore for concurrency control.
        jobs: List of all shard jobs being processed.
    """

    def __init__(self, config: BatchPoolConfig | None = None) -> None:
        """
        Initialize the BatchPool.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or BatchPoolConfig()
        self.client = AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_BATCH_GPT_API_KEY"),
            api_version=os.environ.get("AZURE_BATCH_GPT_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_BATCH_GPT_ENDPOINT"),
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        self.jobs: list[ShardJob] = []

    def discover_shards(self, input_path: Path) -> list[ShardJob]:
        """
        Discover JSONL shard files in the input directory.

        Only searches the immediate directory (non-recursive) to avoid
        picking up output files from subdirectories.

        Args:
            input_path: Path to the input directory.

        Returns:
            List of ShardJob objects for each discovered shard.
        """
        logger.info("Scanning for JSONL files in '{}'...", input_path)

        shard_files: list[Path] = []

        # Use glob on immediate directory only (not recursive) to avoid output/ subdir
        for item in input_path.iterdir():
            if item.is_file() and item.suffix == ".jsonl":
                shard_files.append(item)

        # Sort for deterministic ordering
        shard_files.sort()

        jobs: list[ShardJob] = []
        skipped_count = 0

        for file_path in shard_files:
            # Extract domain from filename (assumes format: domain_shard_N.jsonl)
            domain = (
                file_path.stem.rsplit("_shard_", 1)[0]
                if "_shard_" in file_path.stem
                else file_path.stem
            )

            job = ShardJob(file_path=file_path, domain=domain)

            # Check file size limit
            if job.file_size_mb > self.config.max_file_size_mb:
                logger.warning(
                    "Skipping {} - size {:.2f}MB exceeds limit of {}MB",
                    file_path.name,
                    job.file_size_mb,
                    self.config.max_file_size_mb,
                )
                job.status = JobStatus.SKIPPED
                job.error_message = f"File size {job.file_size_mb:.2f}MB exceeds limit"
                skipped_count += 1

            jobs.append(job)

        valid_jobs = [j for j in jobs if j.status != JobStatus.SKIPPED]
        logger.info(
            "Found {} JSONL files ({} valid, {} skipped due to size)",
            len(jobs),
            len(valid_jobs),
            skipped_count,
        )

        return jobs

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60, max_tries=5)
    async def _upload_file(self, job: ShardJob) -> FileObject:
        """
        Upload a shard file to Azure OpenAI.

        Args:
            job: The shard job to upload.

        Returns:
            FileObject from the API response.
        """
        with open(job.file_path, "rb") as f:
            return await self.client.files.create(file=f, purpose="batch")

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60, max_tries=5)
    async def _get_file_status(self, file_id: str) -> FileObject:
        """
        Retrieve file status from Azure OpenAI.

        Args:
            file_id: The file ID to check.

        Returns:
            FileObject with current status.
        """
        return await self.client.files.retrieve(file_id)

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60, max_tries=5)
    async def _create_batch(self, file_id: str) -> Batch:
        """
        Create a batch job for a processed file.

        Args:
            file_id: The processed file ID.

        Returns:
            Batch object from the API response.
        """
        return await self.client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
        )

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60, max_tries=5)
    async def _get_batch_status(self, batch_id: str) -> Batch:
        """
        Retrieve batch job status from Azure OpenAI.

        Args:
            batch_id: The batch job ID to check.

        Returns:
            Batch object with current status.
        """
        return await self.client.batches.retrieve(batch_id)

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=60, max_tries=5)
    async def _get_file_content(self, file_id: str) -> str:
        """
        Download file content from Azure OpenAI.

        Args:
            file_id: The file ID to download.

        Returns:
            File content as string.
        """
        response = await self.client.files.content(file_id)
        return response.text

    async def _monitor_file_processing(self, job: ShardJob) -> bool:
        """
        Monitor file upload processing until complete.

        Args:
            job: The shard job being monitored.

        Returns:
            True if file processed successfully, False otherwise.
        """
        job.status = JobStatus.FILE_PROCESSING

        while True:
            file_obj = await self._get_file_status(job.file_id)
            status = file_obj.status
            elapsed = int(job.elapsed_time or 0)

            logger.debug(
                "File {} | Status: {} | Elapsed: {}s",
                job.short_file_id,
                status,
                elapsed,
            )

            if status == "processed":
                job.status = JobStatus.FILE_PROCESSED
                logger.info(
                    "File {} processed successfully in {}s",
                    job.short_file_id,
                    elapsed,
                )
                return True

            if status in ("error", "failed"):
                job.status = JobStatus.FILE_FAILED
                job.error_message = f"File processing failed with status: {status}"
                logger.error(
                    "File {} failed processing: {}",
                    job.short_file_id,
                    job.error_message,
                )
                return False

            await asyncio.sleep(self.config.file_poll_interval)

    async def _monitor_batch_job(self, job: ShardJob) -> bool:
        """
        Monitor batch job execution until complete.

        Args:
            job: The shard job being monitored.

        Returns:
            True if batch completed successfully, False otherwise.
        """
        job.status = JobStatus.BATCH_RUNNING

        while True:
            batch_response = await self._get_batch_status(job.batch_id)
            status = batch_response.status
            elapsed = int(job.elapsed_time or 0)

            logger.debug(
                "Batch {} | Status: {} | Elapsed: {}s",
                job.short_batch_id,
                status,
                elapsed,
            )

            if status == "completed":
                job.status = JobStatus.COMPLETED
                job.batch_response = batch_response
                job.end_time = time.time()
                logger.success(
                    "Batch {} completed in {}s",
                    job.short_batch_id,
                    elapsed,
                )
                return True

            if status in ("failed", "canceled", "expired"):
                job.status = (
                    JobStatus.FAILED if status == "failed" else JobStatus.CANCELED
                )
                job.batch_response = batch_response
                job.end_time = time.time()

                if status == "failed" and batch_response.errors:
                    errors = batch_response.errors.data or []
                    error_msgs = [f"{e.code}: {e.message}" for e in errors]
                    job.error_message = "; ".join(error_msgs)
                    for error in errors:
                        logger.error(
                            "Batch {} failed: {} - {}",
                            job.short_batch_id,
                            error.code,
                            error.message,
                        )
                else:
                    job.error_message = f"Batch {status}"
                    logger.error("Batch {} {}", job.short_batch_id, status)

                return False

            await asyncio.sleep(self.config.batch_poll_interval)

    async def _process_job(self, job: ShardJob) -> ShardJob:
        """
        Process a single shard job through its full lifecycle.

        This method acquires a semaphore slot before processing to ensure
        we don't exceed the maximum concurrent jobs limit.

        Args:
            job: The shard job to process.

        Returns:
            The job with updated status and results.
        """
        async with self.semaphore:
            logger.info(
                "Starting job for {} (domain: {}, size: {:.2f}MB)",
                job.file_path.name,
                job.domain,
                job.file_size_mb,
            )

            job.start_time = time.time()

            try:
                # Step 1: Upload file
                job.status = JobStatus.UPLOADING
                file_obj = await self._upload_file(job)
                job.file_id = file_obj.id
                job.status = JobStatus.UPLOADED
                logger.info(
                    "Uploaded {} -> File ID: {}",
                    job.file_path.name,
                    job.short_file_id,
                )

                # Step 2: Wait for file processing
                if not await self._monitor_file_processing(job):
                    return job

                # Step 3: Create batch job
                batch = await self._create_batch(job.file_id)
                job.batch_id = batch.id
                job.status = JobStatus.BATCH_CREATED
                logger.info(
                    "Created batch job for {} -> Batch ID: {}",
                    job.short_file_id,
                    job.short_batch_id,
                )

                # Step 4: Monitor batch job
                await self._monitor_batch_job(job)

            except OpenAIError as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.end_time = time.time()
                logger.error(
                    "Job {} failed with OpenAI error: {}",
                    job.file_path.name,
                    e,
                )

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.end_time = time.time()
                logger.exception(
                    "Job {} failed with unexpected error: {}",
                    job.file_path.name,
                    e,
                )

            return job

    async def run(self, jobs: list[ShardJob]) -> list[ShardJob]:
        """
        Run all shard jobs concurrently with semaphore-based throttling.

        Args:
            jobs: List of shard jobs to process.

        Returns:
            List of jobs with updated statuses and results.
        """
        self.jobs = jobs

        # Filter out already-skipped jobs
        valid_jobs = [j for j in jobs if j.status != JobStatus.SKIPPED]

        if not valid_jobs:
            logger.warning("No valid jobs to process.")
            return jobs

        logger.info(
            "Starting batch pool with {} jobs (max {} concurrent)",
            len(valid_jobs),
            self.config.max_concurrent_jobs,
        )

        # Process all jobs concurrently (semaphore limits actual concurrency)
        tasks = [self._process_job(job) for job in valid_jobs]
        await asyncio.gather(*tasks)

        # Summary statistics
        completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
        failed = sum(
            1 for j in jobs if j.status in (JobStatus.FAILED, JobStatus.FILE_FAILED)
        )
        canceled = sum(1 for j in jobs if j.status == JobStatus.CANCELED)
        skipped = sum(1 for j in jobs if j.status == JobStatus.SKIPPED)

        logger.success(
            "Batch pool finished: {} completed, {} failed, {} canceled, {} skipped",
            completed,
            failed,
            canceled,
            skipped,
        )

        return jobs

    async def save_results(self, jobs: list[ShardJob], output_path: Path) -> None:
        """
        Download and save results for all completed batch jobs.

        Args:
            jobs: List of jobs to save results for.
            output_path: Directory to save output files.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        for job in jobs:
            if job.status != JobStatus.COMPLETED or job.batch_response is None:
                continue

            output_file_id = job.batch_response.output_file_id

            if not output_file_id:
                # Try error file if no output file
                output_file_id = job.batch_response.error_file_id

            if not output_file_id:
                logger.warning(
                    "No output file for batch {} (job: {})",
                    job.short_batch_id,
                    job.file_path.name,
                )
                continue

            try:
                content = await self._get_file_content(output_file_id)
                output_file = output_path / f"{job.batch_response.id}.jsonl"

                with open(output_file, "w") as f:
                    f.write(content)

                logger.info("Saved batch output to {}", output_file)

            except OpenAIError as e:
                logger.error(
                    "Failed to download results for batch {}: {}",
                    job.short_batch_id,
                    e,
                )


async def run_async_batch_gpt(
    input_path: Path,
    output_path: Path,
    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_JOBS,
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
) -> list[ShardJob]:
    """
    Run async batch GPT processing on all shards in input directory.

    Args:
        input_path: Path to input directory containing JSONL shards.
        output_path: Path to output directory for results.
        max_concurrent_jobs: Maximum number of concurrent batch jobs.
        max_file_size_mb: Maximum file size in MB per shard.

    Returns:
        List of processed ShardJob objects.
    """
    config = BatchPoolConfig(
        max_concurrent_jobs=max_concurrent_jobs,
        max_file_size_mb=max_file_size_mb,
    )

    pool = BatchPool(config=config)

    # Discover shards (non-recursive to avoid output/ subdir)
    jobs = pool.discover_shards(input_path)

    if not jobs:
        logger.warning("No JSONL files found in the input directory. Exiting.")
        return []

    # Run all jobs with concurrency control
    jobs = await pool.run(jobs)

    # Save results
    await pool.save_results(jobs, output_path)

    return jobs


def parse_batch_gpt_output_files(output_path: str) -> pd.DataFrame:
    """
    Parse all JSONL output files from batch GPT processing.

    Args:
        output_path: Path to directory containing output JSONL files.

    Returns:
        DataFrame with parsed response data.
    """
    all_data: list[dict] = []
    output_dir = Path(output_path)

    # Get all jsonl files (non-recursive)
    output_files = sorted(output_dir.glob("*.jsonl"))
    logger.info("Found {} output files to process", len(output_files))

    for fin in output_files:
        with open(fin, encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for entry in data:
            parsed_entry = {
                "file": fin.name,
                "custom_id": entry.get("custom_id"),
                "model": entry.get("response", {}).get("body", {}).get("model"),
                "status_code": entry.get("response", {}).get("status_code"),
                "message": None,
                "error": entry.get("error"),
            }

            # Extract response message if present
            choices = entry.get("response", {}).get("body", {}).get("choices", [])
            if choices and "message" in choices[0]:
                parsed_entry["message"] = choices[0]["message"].get("content")

            all_data.append(parsed_entry)

    df = pd.DataFrame(all_data)

    if not df.empty:
        df["status"] = df["status_code"].apply(
            lambda x: "Success" if x == 200 else "Failed"
        )

    return df


def pretty_print_dataframe(df: pd.DataFrame, rows: int = 10, cols: int = 10) -> None:
    """
    Print a glimpse of a DataFrame in the terminal.

    Args:
        df: The DataFrame to print.
        rows: Number of rows to display. Default is 10.
        cols: Number of columns to display. Default is 10.
    """
    rows = min(rows, len(df))
    cols = min(cols, len(df.columns))

    df_glimpse = df.iloc[:rows, :cols]
    headers = ["Index"] + list(df_glimpse.columns)

    row_strings: list[list[str]] = []
    for i, row in df_glimpse.iterrows():
        row_strings.append([str(i)] + [str(row[col]) for col in df_glimpse.columns])

    col_widths = [
        max(len(str(header)), max((len(row[i]) for row in row_strings), default=0))
        for i, header in enumerate(headers)
    ]

    header_row = " | ".join(
        header.ljust(col_widths[i]) for i, header in enumerate(headers)
    )
    print("-" * len(header_row))
    print(header_row)
    print("-" * len(header_row))

    for row in row_strings:
        print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))

    print("-" * len(header_row))


def main() -> None:
    """Main entry point for the batch GPT runner."""
    parser = argparse.ArgumentParser(
        description="Run async batch GPT on shards with concurrency control"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/input/shards",
        help="Path to the input shards directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/input/shards/output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_JOBS,
        help=f"Maximum concurrent batch jobs (default: {DEFAULT_MAX_CONCURRENT_JOBS})",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE_MB,
        help=f"Maximum file size in MB per shard (default: {DEFAULT_MAX_FILE_SIZE_MB})",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        logger.error("Input path does not exist: {}", input_path)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        run_async_batch_gpt(
            input_path=input_path,
            output_path=output_path,
            max_concurrent_jobs=args.max_concurrent,
            max_file_size_mb=args.max_file_size_mb,
        )
    )


if __name__ == "__main__":
    main()
