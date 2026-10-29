Model performance tests
=======================

Open **Performance Tests**, select a running chat model, and start a test. You can also open this page from the performance-test button on a running model. Select history records to compare results, or export a record as JSON.

Configure the approximate input token target, maximum output tokens, comma-separated concurrency points, requests per point, warmup requests, and random seed. The model must support token counting. Actual input and output lengths come from response usage; models may stop before the output limit. Warmup is excluded, and each concurrency point uses independent synthetic prompts. Server caches are not flushed.

Tests run sequentially in background processes on the API host and send requests through the Xinference chat API, including its replica routing. Closing the page does not stop a test. Stop a task before deleting it. Results, logs, and configuration snapshots are saved under ``XINFERENCE_HOME/benchmarks``. Restarting the API server interrupts active tasks and preserves their history.

TTFT includes reasoning output. TPOT measures decode time per output token, excluding the first token. ITL measures gaps between streamed chunks, which may contain multiple tokens. The table shows averages and tail percentiles; unavailable measurements appear as a dash. Compare the workload, actual token lengths, runner metric version, and environment snapshots before comparing performance.

With authentication enabled, these endpoints require both ``models:write`` and ``models:read``. Inference uses the initiating credential, which is held in memory and sent to the runner through stdin. It is not saved with results. An expired or revoked credential causes requests to fail. Tests use live serving capacity.

The built-in ``xinference`` runner reuses the serving benchmark implementation. Runner adapters expose ``run``, ``cancel``, and ``capabilities``; records retain the runner version and raw request measurements. Additional tools can implement the adapter contract without changing task storage or the results page.

API example::

    curl -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"model_uid":"my-model","concurrency":[1,4,16],"requests":100}' \
      http://localhost:9997/v1/benchmarks

    curl -H "Authorization: Bearer $TOKEN" \
      http://localhost:9997/v1/benchmarks
