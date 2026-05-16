# AGENTS.md

Instructions for agents working in this repository.

## Environment

- Host: Fedora 44 / Silverblue on `vega`.
- Interactive user shell: fish. Automation may run under bash unless a task specifically requires fish behavior.
- Project purpose: benchmark Qwen 3.6 35B A3B local agentic serving performance across llama.cpp and vLLM on AMD Strix Halo.

## Sudo Policy

- Always precheck with `sudo -n true` before any sudo command.
- If the precheck passes, use only `sudo -n <command>`.
- If the precheck fails and a real GUI askpass prompt is available, `sudo -A <command>` is allowed.
- If the precheck fails and prompts are not user-visible, ask the user to run `sudo -v` in a separate interactive shell, then continue with `sudo -n <command>`.
- Sudo credentials are globally cached for about 20 minutes (`timestamp_type=global`, `timestamp_timeout=20`).

## Browser Automation

- Before any use of `playwright-cli`, read `/home/sid/.config/agents/AGENTS.playwright-cli.md`.
- Do not rely on skill instructions alone for Playwright; the local instructions are required.

## Repository Workflow

- Start every resumed session by reading `TASKS.md`; it is the canonical project state and work queue.
- Use the `TASKS.md` "Resume Context" section to recover current architecture, runtime ownership rules, and next steps before editing code.
- Keep benchmark documentation and scripts focused on the primary question: how many concurrent local agentic Qwen sessions this machine can support before latency degrades.
- Prefer short, reproducible defaults. Expensive tuning should be opt-in and cached.
- Use `TASKS.md` as the project work queue for known issues and planned improvements.
- When starting work on a listed item, change its marker from `[ ]` to `[-]`.
- When finishing a listed item, change its marker to `[x]` only after code/docs are updated and verified.
- Use `[!]` for blocked tasks and include the blocker in the task text or the final response.
- Add newly discovered follow-up work to `TASKS.md` before or with the related code change.
- Commit meaningful checkpoints periodically when making multi-step changes.
- Do not commit generated benchmark results unless explicitly requested.
- Avoid destructive git operations unless the user explicitly asks for them.

## Benchmark Defaults

- Treat `N*` as the largest primary-ramp concurrency level that keeps TTFT p95 below the configured degradation threshold with 100% request success.
- Keep overload/queueing experiments separate from the primary `N*` answer.
- Record environment details with benchmark output so results can be compared across runtime/container changes.
