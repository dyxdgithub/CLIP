# TinyCLIP Harness

This directory is the runtime contract for agent harness tooling.

```text
harness/
├── config/
│   └── environment.json
├── scripts/
│   ├── setup-env.sh
│   ├── start-server.sh
│   └── teardown-env.sh
└── README.md
```

`harness/config/verify.json` is intentionally absent. Verification
configuration is generated at task runtime from `environment.json` and the
specific task context.

This project does not start a long-running HTTP server. The `start-server.sh`
script performs a lightweight version-module smoke check instead.
