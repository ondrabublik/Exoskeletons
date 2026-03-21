# WebSocket API and Implementation Notes

This document describes the updated real-time protocol used by `index.js` and `public/index.html`.

## Transport

- Server: Socket.IO on `http://localhost:3000`
- Live data source: UDP packets on port `9999`
- Expected UDP payload: 7 little-endian float32 values (`28` bytes)

## Channel Order

Values are always interpreted in this fixed order:

1. Angle
2. Pitch 1
3. Pitch Rate 1
4. Pitch 2
5. Pitch Rate 2
6. Input
7. Prediction

## Server -> Client Events

### `sensor-data`

Emitted for each valid UDP packet.

Payload:

```json
{
  "timestamp": 1710000000000,
  "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
}
```

### `recording:status`

Broadcast whenever recording state changes and on initial socket connection.

Payload when idle:

```json
{ "isRecording": false }
```

Payload when active:

```json
{
  "isRecording": true,
  "fileName": "2026-03-21T10-11-12-123Z_test.csv",
  "startedAt": "2026-03-21T10:11:12.123Z"
}
```

### `recordings:list`

Broadcast after recording stops and on initial socket connection.

Payload:

```json
[
  {
    "fileName": "2026-03-21T10-11-12-123Z_test.csv",
    "sizeBytes": 53210,
    "createdAt": "2026-03-21T10:11:12.150Z",
    "modifiedAt": "2026-03-21T10:12:03.700Z"
  }
]
```

## Client -> Server Request Events (Ack style)

These events use Socket.IO callback acknowledgements.

## `recording:start`

Starts CSV recording.

Request payload:

```json
{ "name": "optional_custom_name" }
```

Ack success:

```json
{
  "ok": true,
  "status": {
    "isRecording": true,
    "fileName": "2026-03-21T10-11-12-123Z_optional_custom_name.csv",
    "startedAt": "2026-03-21T10:11:12.123Z"
  }
}
```

Ack failure:

```json
{ "ok": false, "error": "Recording is already active" }
```

## `recording:stop`

Stops active recording.

Request payload:

```json
{}
```

Ack success:

```json
{
  "ok": true,
  "status": {
    "isRecording": false,
    "lastFileName": "2026-03-21T10-11-12-123Z_optional_custom_name.csv",
    "sampleCount": 1234
  }
}
```

## `recordings:list`

Fetches all available CSV recordings.

Request payload:

```json
{}
```

Ack success:

```json
{
  "ok": true,
  "recordings": [
    {
      "fileName": "2026-03-21T10-11-12-123Z_optional_custom_name.csv",
      "sizeBytes": 53210,
      "createdAt": "2026-03-21T10:11:12.150Z",
      "modifiedAt": "2026-03-21T10:12:03.700Z"
    }
  ]
}
```

## `recording:get`

Loads and parses a recording for replay.

Request payload:

```json
{ "fileName": "2026-03-21T10-11-12-123Z_optional_custom_name.csv" }
```

Ack success:

```json
{
  "ok": true,
  "recording": {
    "fileName": "2026-03-21T10-11-12-123Z_optional_custom_name.csv",
    "channelNames": [
      "Angle",
      "Pitch 1",
      "Pitch Rate 1",
      "Pitch 2",
      "Pitch Rate 2",
      "Input",
      "Prediction"
    ],
    "samples": [
      { "timestamp": 1710000000000, "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] }
    ]
  }
}
```

Ack failure:

```json
{ "ok": false, "error": "Recording not found" }
```

## CSV Format

Files are stored under `recordings/`.

Header:

```csv
timestamp,Angle,Pitch 1,Pitch Rate 1,Pitch 2,Pitch Rate 2,Input,Prediction
```

Rows:

- `timestamp` is UNIX epoch time in milliseconds
- all values are decimal floats as received from UDP

## Frontend Behavior Summary

- Two tabs:
  - `Live Data`: real-time charts + start/stop recording
  - `Recordings`: file list + replay controls + replay charts
- Replay uses original sample timestamps and supports:
  - play/pause/reset
  - speed (0.5x, 1x, 2x, 4x)
  - scrub slider
- Both live and replay charts support per-graph expand/contract and show current value.
