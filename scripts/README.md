# scripts/

Helper scripts for local development tooling. These are **not** part of the
Julia package; they run outside the package environment.

## Setup (one-time)

### Runic (code formatter)

Install Runic in the shared `@runic` environment so the `runic` CLI is
available system-wide:

```bash
julia -e 'import Pkg; Pkg.activate("runic"; shared=true); Pkg.add("Runic")'
```

Then put the driver script on your PATH. The canonical location for the
`runic` binary is `~/.local/bin/runic`; see the
[Runic README](https://github.com/fredrikekre/Runic.jl) for the exact
script content.

### ReLint (static analysis)

Install ReLint in the shared `@relint` environment:

```bash
julia -e '
  import Pkg
  Pkg.activate("relint"; shared=true)
  Pkg.add(url="https://github.com/RelationalAI-oss/ReLint.jl")
'
```

Optionally install the generic `relint` CLI to `~/.local/bin/relint`:

```sh
#!/bin/sh
# ~/.local/bin/relint
export JULIA_LOAD_PATH="@relint:@stdlib"
exec julia --startup-file=no -e '
using ReLint
paths = isempty(ARGS) ? ["."] : ARGS
result = ReLint.LintGlobalReport()
for p in paths; ReLint.run_lint(p; result); end
n = result.violations_count + result.recommendations_count + result.fatal_violations_count
exit(n > 0 ? 1 : 0)
' -- "$@"
```

Make it executable: `chmod +x ~/.local/bin/relint`.

## Usage

Run **before committing** (from the package root):

```bash
# 1. Format
runic src/

# 2. Static analysis (project-specific suppressions: @inbounds, return type
#    annotation, in/tin, unsafe-logging)
JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/
```

Both checks are also enforced in CI (`.github/workflows/lint.yml`).
