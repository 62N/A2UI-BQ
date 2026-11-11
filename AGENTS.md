# Gemini Enterprise A2UI v0.9 + BigQuery MCP Engineering Guide (`AGENTS.md`)

## Critical A2UI `v0.9` Rules in Gemini Enterprise

1. **Turn 0 Instant Welcome Rendering (`text == ""`)**:
   - On Turn 0, return **only** the `createSurface` and `updateComponents` A2UI `DataPart`s (`mimeType: application/json+a2ui`) with **no** companion Markdown `TextPart`.
   - Emitting any Markdown text on Turn 0 sets `partialDisplayedText` in `ucs-text-streamer.ts` (`forceStreaming = true` when `turns.length === 1`), which blocks the A2UI starter surface from mounting.

2. **`VegaChart` Spec Binding (`updateDataModel`)**:
   - In `gemini_enterprise_composite_catalog.json`, `VegaChart.spec` uses `$ref: common_types.json#/$defs/DynamicValue`, which accepts `DataBinding` (`{"path": "/vegaSpec"}`) rather than an arbitrary inline JSON object.
   - Always emit `{"version": "v0.9", "updateDataModel": {"surfaceId": surface_id, "value": {"vegaSpec": vega_spec}}}` between `createSurface` and `updateComponents`, and set `"spec": {"path": "/vegaSpec"}` on `VegaChart`.

3. **Side-Panel `Canvas` Component Keys**:
   - The browser's strict Zod validator for `Canvas` accepts `id`, `component`, `autoOpen`, `autoFullscreen`, `closePanel`, `cardTitle`, `cardDescription`, `cardIcon`, and `children`. Do **not** pass `cardButtonVariant`.

4. **`GcbpTable` Server-Side Sorting (`fetchData`)**:
   - Clicking a column header with `"sortable": true` in `GcbpTable` dispatches an A2UI action (`{"action": {"name": "fetchData", "sourceComponentId": "...", "context": {"sortString": "<field> <asc|desc>"}}}`).
   - Always intercept `action.name == "fetchData"` in `A2UIRootRouterAgent` to sort the rows and re-render the complete dashboard (`activeTab: 1` on `MaterialTabs`) instead of letting `"User action triggered."` fall through to the LLM.

5. **3-Step Synthetic Thinking Separation**:
   - Emit synthetic thinking steps as distinct `partial=True` `Event`s containing `Part(text=..., thought=True)` separated by `STEP_DELAY_SECONDS = 1.5` so Gemini Enterprise's SSE proxy flushes each step individually.
   - Never combine `adk_thought=True` parts and A2UI `DataPart`s in the same `TaskArtifactUpdateEvent`.
