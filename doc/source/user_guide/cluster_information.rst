.. _cluster_information:

Cluster information
===================

Open **Cluster Information** to inspect Worker, Supervisor and Router nodes. Search Workers by node address or GPU model, and switch between GPU grid and list views.

The GPU grid shows up to four cards per row and two rows per page on wide screens, or two cards per row on narrow screens. Use the arrows to see additional devices, and select a card to see its model and used, total and available GPU memory. Each card shows GPU utilization and memory usage; progress bars turn red above 90%. Null or non-finite readings are shown as unavailable; backend zero sentinels can still appear as 0% utilization.

Detailed cluster API
--------------------

``GET /v1/cluster/info?detailed=true`` includes a ``gpus`` array for each Worker. Each device reports ``index``, ``name``, ``utilization``, ``memory_used``, ``memory_total`` and ``memory_available``. Utilization is a percentage and memory values are in bytes; missing or unsupported readings can be ``null`` or backend-specific numeric sentinels. NVIDIA may report zero for unavailable memory or unsupported utilization; a client cannot distinguish the latter sentinel from measured 0% utilization. Compact responses keep their existing structure.
