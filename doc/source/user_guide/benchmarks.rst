Model performance tests
=======================

Open **Performance Tests**, select a running chat model, and start a test. You can also open this page from the performance-test button on a running model. Select history records to compare results, or export a record as JSON.

.. figure:: /_static/benchmarks_setup.png
   :alt: Configure a benchmark for a running chat model.
   :width: 100%
   :align: center

   Configure a benchmark for a running chat model.

Configure the approximate input token target, maximum output tokens, comma-separated concurrency points, requests per point, warmup requests, and random seed. When a separate token-count endpoint is unavailable, calibration uses single-token chat requests and their prompt usage; these probes are excluded from measurements. Actual input and output lengths come from response usage; models may stop before the output limit. Warmup is excluded, and each concurrency point uses independent synthetic prompts. Server caches are not flushed.

Tests run sequentially in background processes on the API host and send requests through the Xinference chat API, including its replica routing. Closing the page does not stop a test. Stop a task before deleting it. Results, logs, and configuration snapshots are saved under ``XINFERENCE_HOME/benchmarks``. Restarting the API server interrupts active tasks and preserves their history.

TTFT includes reasoning output. TPOT measures decode time per output token, excluding the first token. ITL measures gaps between streamed chunks, which may contain multiple tokens. The table shows averages and tail percentiles; unavailable measurements appear as a dash. Compare the workload, actual token lengths, runner metric version, and environment snapshots before comparing performance.

With authentication enabled, these endpoints require both ``models:write`` and ``models:read``. Jobs receive a temporary credential when execution starts, restricted to the target model's chat and token-count endpoints. It stays in memory, travels to the runner over stdin, and is revoked when execution ends. Session expiry does not interrupt queued or running jobs. Disabling the account, resetting its password, or removing model-management permissions revokes job authorization. HTTP 401/403 aborts the current concurrency point without saving its metrics. Tests use live serving capacity.

The built-in ``xinference`` runner reuses the serving benchmark implementation. Runner adapters expose ``run``, ``cancel``, and ``capabilities``; records retain the runner version and raw request measurements. Additional tools can implement the adapter contract without changing task storage or the results page.

Each test displays output throughput, TTFT, and TPOT curves against concurrency. Use Latency statistic to switch the latency charts between mean, P95, and P99. Selected history records are overlaid with matching colors across charts. Points represent measurements; missing values stay blank. A notice identifies different workloads or runner metric versions.

.. figure:: /_static/benchmarks_compare.png
   :alt: Compare saved runs with throughput and P99 latency curves.
   :width: 100%
   :align: center

   Compare saved runs with throughput and P99 latency curves.

API example::

    curl -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"model_uid":"my-model","concurrency":[1,4,16],"requests":100}' \
      http://localhost:9997/v1/benchmarks

    curl -H "Authorization: Bearer $TOKEN" \
      http://localhost:9997/v1/benchmarks
