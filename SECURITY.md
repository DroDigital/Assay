# Security

## Threat model

assay executes the model you point it at and prints what the model returns.

* **Spec files are code-adjacent.** `type = "python"` imports a module and `type = "command"` runs
  a program. Only run specs you trust, as you would any script.
* **Model output is untrusted.** Reports escape it (terminal escapes, Markdown fences, HTML, XML).
* **Sensitive data.** Sensitive-data checks withhold offending outputs and mask excerpts;
  `redact = true` / `--redact` removes all counterexample payloads and any reason that could quote them. Review reports before sharing
  them if your test cases themselves contain personal data.
* **Network.** HTTP models accept `http(s)` URLs only, never follow redirects (so credentials
  cannot be forwarded to another host), cap response size, and read secrets from environment
  variables.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting (Security tab → "Report a vulnerability") rather
than a public issue. Expect an acknowledgement within a few days.
