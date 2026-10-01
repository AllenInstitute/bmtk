# Initial-state input recovery diagnostics

`InitStateFromInputModule` retains its existing retry policy and previous-state
fallback. A recoverable interruption now produces a short BMTK-style notice:

```text
[WARNING] Initial-state input generation was interrupted; continuing with the previous initial state. Details: /output/dpointnet-diagnostics-<unique-id>.log.
```

If a retry succeeds, an INFO notice says generation recovered and execution is
continuing normally. The diagnostic file contains the full Python traceback,
TensorFlow exception text, attempt count, and batch size for each interruption,
including iterator-rebuild errors.

Diagnostics are appended to a uniquely named file beside the configured BMTK
file log. Without a file log, they go to the system temporary directory. The
notice gives the absolute path. Failure to write diagnostics is reported
explicitly, including the original error text.

Use `dpointnet.Config` and call `config.build_env()` to configure DPointNet's
file log, console output, level and format before execution. The same logger
handles recovery notices and determines the diagnostic directory.

For eager, initial-state LGN inputs, expected `InvalidArgumentError` and
`UnknownError` exceptions are transferred out of the `tf.data` Python callback
before being re-raised to the existing recovery code. No substitute spikes are
generated; an interrupted partial batch is discarded. Normal seeded batches,
stimulus signatures, and retry behavior are preserved. Ordinary LGN datasets,
graph fetching, and per-device LGN generation retain their existing paths.
Initial-state warmup uses host-reference LGN batches even if the module opts
into per-device generation, so it can run outside the distribution strategy.

TensorFlow's native kernel notices are not globally suppressed or redirected:
they may still appear on stderr. The saved file contains the complete handled
exception and traceback, not a capture of process-wide native stderr. This
avoids hiding unrelated failures or redirecting output from concurrent training.

If no previous state exists, failure still raises the original exception.
Unexpected errors remain fatal rather than being classified as recoverable.
