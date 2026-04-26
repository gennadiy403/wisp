# Roadmap: Govori

## Overview

Govori's feature set is complete and competitive. This milestone hardens the existing codebase for public release: fix security vulnerabilities, extract modules from the 1900-line monolith, then package and distribute via PyPI. No new features -- ship reliability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Security & Safety** - Fix ship-blocking vulnerabilities and add safety guards for system-level APIs
- [ ] **Phase 2: Architecture & Reliability** - Extract modules from monolith, replace globals with explicit state, add logging and graceful shutdown
- [ ] **Phase 3: Packaging & Distribution** - Package for PyPI, add smoke tests, publish with documentation

## Phase Details

### Phase 1: Security & Safety
**Goal**: Users can run Govori without risk of shell injection, system input freeze, or silent failures
**Depends on**: Nothing (first phase)
**Requirements**: SEC-01, SEC-02, SEC-03, SEC-04, REL-01
**Success Criteria** (what must be TRUE):
  1. Note editing uses subprocess.run -- no shell injection vector exists in the codebase
  2. If Accessibility permission is revoked while running, Govori detects it and warns the user instead of freezing system input
  3. If OpenAI API hangs, transcription times out after 30s with visible feedback to the user
  4. During onboarding, user sees a clear privacy notice stating voice goes to OpenAI and notes go to Anthropic
  5. If no microphone is available or permission is denied, user sees an error message instead of a crash
**Plans:** 3 plans
Plans:
- [x] 01-01-PLAN.md -- Shell injection fix + privacy notice in onboarding
- [x] 01-02-PLAN.md -- HUD error infrastructure + API timeout with retry
- [ ] 01-03-PLAN.md -- CGEventTap health monitoring + microphone error handling

### Phase 2: Architecture & Reliability
**Goal**: Govori runs as a proper Python package with isolated modules, explicit state, structured logging, and clean shutdown
**Depends on**: Phase 1
**Requirements**: ARCH-01, ARCH-02, ARCH-03, REL-02, REL-03, REL-04
**Success Criteria** (what must be TRUE):
  1. govori.py is replaced by a govori/ package with separate modules (config, state, hud, audio, transcribe, notes, macos, predict, cli)
  2. No mutable module-level globals -- application state lives in an AppState dataclass with explicit transitions
  3. All output goes to ~/.config/govori/govori.log via loguru with rotation -- no print() calls remain
  4. Ctrl+C or SIGTERM triggers clean shutdown: audio stream closed, no os._exit(0)
  5. Invalid config YAML produces a human-readable error message, not a silent empty dict
**Plans**: TBD

### Phase 3: Packaging & Distribution
**Goal**: Users can install Govori via `pipx install govori` from PyPI with confidence it works
**Depends on**: Phase 2
**Requirements**: DIST-01, DIST-02, DIST-03, DIST-04, DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. `pipx install govori` installs the tool with all dependencies and `govori` command works
  2. Smoke tests pass in CI: config loading, API error paths, note merge logic, path sanitization
  3. GitHub Actions publishes to PyPI via trusted publishers on tagged release
  4. README shows real GitHub URL, system requirements (macOS >=13, Python >=3.10), and privacy section
  5. govori.io has a privacy policy page documenting what data is sent where
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Security & Safety | 0/3 | Planned | - |
| 2. Architecture & Reliability | 0/TBD | Not started | - |
| 3. Packaging & Distribution | 0/TBD | Not started | - |
