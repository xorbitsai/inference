.. _metrics:

==================
Metrics
==================

There are two types of metrics exporters in Xinferece cluseter:

- Supervisor metrics exporter: <endpoint>/metrics. e.g. http://127.0.0.1:9997/metrics
- Worker metrics exporter at each worker node, the exporter host and port can be set by `--metrics-exporter-host` and `--metrics-exporter-port` options in `xinference-local` or `xinference-worker` command.

Supervisor Metrics
^^^^^^^^^^^^^^^^^^



- **exceptions_total_counter** (counter): Total number of requested which generated an exception

- **requests_total_counter** (counter): Total number of requests received

- **responses_total_counter** (counter): Total number of responses sent

- **status_codes_counter** (counter): Total number of response status codes



Worker Metrics
^^^^^^^^^^^^^^



- **xinference:generate_tokens_per_s** (gauge): Generate throughput in tokens/s.

- **xinference:input_tokens_total_counter** (counter): Total number of input tokens.

- **xinference:output_tokens_total_counter** (counter): Total number of output tokens.

- **xinference:time_to_first_token_ms** (gauge): First token latency in ms.
