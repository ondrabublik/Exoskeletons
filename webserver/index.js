const express = require('express');
const dgram = require('dgram');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { Server } = require('socket.io');

const app = express();
app.use(express.static('public'));

// File download route
app.get('/api/recordings/:fileName', (req, res) => {
	const safeFileName = path.basename(req.params.fileName || '');
	if (!safeFileName.toLowerCase().endsWith('.csv')) {
		return res.status(400).json({ error: 'Only .csv files are allowed' });
	}
	const filePath = path.join(RECORDINGS_DIR, safeFileName);
	if (!fs.existsSync(filePath)) {
		return res.status(404).json({ error: 'Recording not found' });
	}
	res.download(filePath, safeFileName);
});

const RECORDINGS_DIR = path.join(__dirname, 'recordings');
fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

const udpServer = dgram.createSocket('udp4');
const httpServer = http.createServer(app);
const io = new Server(httpServer, {
	cors: { origin: "*" }
});

const CHANNEL_NAMES = [
	'Angle',
	'Pitch 1',
	'Pitch Rate 1',
	'Pitch 2',
	'Pitch Rate 2',
	'Input',
	'Prediction'
];

let activeRecording = null;
const RECORDING_FLUSH_INTERVAL_MS = 100;
const RECORDING_FLUSH_BYTES = 64 * 1024;

function sanitizeName(name) {
	if (!name || typeof name !== 'string') {
		return '';
	}
	return name
		.trim()
		.replace(/[^a-zA-Z0-9_-]+/g, '_')
		.replace(/_+/g, '_')
		.slice(0, 64);
}

function createRecordingFilename(customName) {
	const stamp = new Date().toISOString().replace(/[:.]/g, '-');
	const safeName = sanitizeName(customName);
	return safeName ? `${stamp}_${safeName}.csv` : `${stamp}.csv`;
}

function getRecordingStatus() {
	if (!activeRecording) {
		return { isRecording: false };
	}

	return {
		isRecording: true,
		fileName: activeRecording.fileName,
		startedAt: activeRecording.startedAt
	};
}

function listRecordings() {
	const files = fs.readdirSync(RECORDINGS_DIR)
		.filter((entry) => entry.toLowerCase().endsWith('.csv'));

	const recordings = files.map((fileName) => {
		const filePath = path.join(RECORDINGS_DIR, fileName);
		const stat = fs.statSync(filePath);
		return {
			fileName,
			sizeBytes: stat.size,
			createdAt: stat.birthtime.toISOString(),
			modifiedAt: stat.mtime.toISOString()
		};
	});

	recordings.sort((a, b) => new Date(b.modifiedAt) - new Date(a.modifiedAt));
	return recordings;
}

function loadRecording(fileName) {
	const safeFileName = path.basename(fileName || '');
	if (!safeFileName.toLowerCase().endsWith('.csv')) {
		throw new Error('Only .csv files are allowed');
	}

	const filePath = path.join(RECORDINGS_DIR, safeFileName);
	if (!fs.existsSync(filePath)) {
		throw new Error('Recording not found');
	}

	const csv = fs.readFileSync(filePath, 'utf8');
	const lines = csv.split(/\r?\n/).filter((line) => line.trim().length > 0);
	if (lines.length < 2) {
		return {
			fileName: safeFileName,
			channelNames: CHANNEL_NAMES,
			samples: []
		};
	}

	const samples = [];
	for (let i = 1; i < lines.length; i++) {
		const parts = lines[i].split(',');
		if (parts.length < 1 + CHANNEL_NAMES.length) {
			continue;
		}

		const timestamp = Number(parts[0]);
		const values = parts.slice(1, 1 + CHANNEL_NAMES.length).map((part) => Number(part));
		if (!Number.isFinite(timestamp) || values.some((value) => !Number.isFinite(value))) {
			continue;
		}

		samples.push({
			timestamp,
			values
		});
	}

	return {
		fileName: safeFileName,
		channelNames: CHANNEL_NAMES,
		samples
	};
}

function startRecording(customName) {
	if (activeRecording) {
		throw new Error('Recording is already active');
	}

	const fileName = createRecordingFilename(customName);
	const filePath = path.join(RECORDINGS_DIR, fileName);
	const stream = fs.createWriteStream(filePath, { flags: 'w' });

	stream.write(`timestamp,${CHANNEL_NAMES.join(',')}\n`);

	activeRecording = {
		fileName,
		filePath,
		stream,
		startedAt: new Date().toISOString(),
		sampleCount: 0,
		pendingCsv: '',
		flushTimer: null,
		startHrNs: process.hrtime.bigint()
	};

	activeRecording.flushTimer = setInterval(() => {
		if (!activeRecording || activeRecording.fileName !== fileName) {
			return;
		}
		if (activeRecording.pendingCsv.length > 0) {
			activeRecording.stream.write(activeRecording.pendingCsv);
			activeRecording.pendingCsv = '';
		}
	}, RECORDING_FLUSH_INTERVAL_MS);

	io.emit('recording:status', getRecordingStatus());
	return getRecordingStatus();
}

function stopRecording() {
	if (!activeRecording) {
		return getRecordingStatus();
	}

	const recording = activeRecording;
	activeRecording = null;

	if (recording.flushTimer) {
		clearInterval(recording.flushTimer);
		recording.flushTimer = null;
	}
	if (recording.pendingCsv.length > 0) {
		recording.stream.write(recording.pendingCsv);
		recording.pendingCsv = '';
	}
	recording.stream.end();
	io.emit('recording:status', getRecordingStatus());
	io.emit('recordings:list', listRecordings());

	return {
		isRecording: false,
		lastFileName: recording.fileName,
		sampleCount: recording.sampleCount
	};
}

// UDP receives Arduino data
const valueCount = 7;
udpServer.on('message', (msg) => {
	try {
		if (msg.length !== 4 * valueCount) {
			console.log('Unexpected packet size:', msg.length);
			return;
		}

		const values = [];
		for (let i = 0; i < valueCount; i++) {
			values.push(msg.readFloatLE(i * 4));
		}

		const packet = {
			timestamp: Date.now(),
			values
		};

		if (activeRecording) {
			const elapsedMs = Number(process.hrtime.bigint() - activeRecording.startHrNs) / 1e6;
			activeRecording.pendingCsv += `${elapsedMs.toFixed(3)},${values.join(',')}\n`;
			if (activeRecording.pendingCsv.length >= RECORDING_FLUSH_BYTES) {
				activeRecording.stream.write(activeRecording.pendingCsv);
				activeRecording.pendingCsv = '';
			}
			activeRecording.sampleCount += 1;
		}

		io.emit('sensor-data', packet);

	} catch (ex) {
		console.log(`Bad data: ${ex}`);
	}
});

udpServer.bind(9999);

// WebSocket server
io.on('connection', (socket) => {
	console.log(`Browser ${socket.handshake.headers.host} connected`);

	socket.emit('recording:status', getRecordingStatus());
	socket.emit('recordings:list', listRecordings());

	socket.on('recording:start', (payload = {}, callback) => {
		try {
			const status = startRecording(payload.name);
			if (typeof callback === 'function') {
				callback({ ok: true, status });
			}
		} catch (error) {
			if (typeof callback === 'function') {
				callback({ ok: false, error: error.message });
			}
		}
	});

	socket.on('disconnect', (reason) => {
        console.log(`Browser ${socket.handshake.headers.host} disconnected; Reason: ${reason}`);
    });

	socket.on('recording:stop', (_payload = {}, callback) => {
		try {
			const status = stopRecording();
			if (typeof callback === 'function') {
				callback({ ok: true, status });
			}
		} catch (error) {
			if (typeof callback === 'function') {
				callback({ ok: false, error: error.message });
			}
		}
	});

	socket.on('recordings:list', (_payload = {}, callback) => {
		try {
			const recordings = listRecordings();
			if (typeof callback === 'function') {
				callback({ ok: true, recordings });
			}
		} catch (error) {
			if (typeof callback === 'function') {
				callback({ ok: false, error: error.message });
			}
		}
	});

	socket.on('recording:get', (payload = {}, callback) => {
		try {
			const recording = loadRecording(payload.fileName);
			if (typeof callback === 'function') {
				callback({ ok: true, recording });
			}
		} catch (error) {
			if (typeof callback === 'function') {
				callback({ ok: false, error: error.message });
			}
		}
	});

	socket.on('recording:delete', (payload = {}, callback) => {
		try {
			const safeFileName = path.basename(payload.fileName || '');
			if (!safeFileName.toLowerCase().endsWith('.csv')) {
				throw new Error('Invalid file name');
			}
			if (activeRecording && activeRecording.fileName === safeFileName) {
				throw new Error('Cannot delete an active recording');
			}
			const filePath = path.join(RECORDINGS_DIR, safeFileName);
			if (!fs.existsSync(filePath)) {
				throw new Error('Recording not found');
			}
			fs.unlinkSync(filePath);
			io.emit('recordings:list', listRecordings());
			if (typeof callback === 'function') {
				callback({ ok: true });
			}
		} catch (error) {
			if (typeof callback === 'function') {
				callback({ ok: false, error: error.message });
			}
		}
	});

	socket.on('recording:deleteAll', (_payload = {}, callback) => {
		try {
			const activeFileName = activeRecording ? activeRecording.fileName : null;
			const files = fs.readdirSync(RECORDINGS_DIR)
				.filter((f) => f.toLowerCase().endsWith('.csv') && f !== activeFileName);
			files.forEach((f) => fs.unlinkSync(path.join(RECORDINGS_DIR, f)));
			io.emit('recordings:list', listRecordings());
			if (typeof callback === 'function') {
				callback({ ok: true, deleted: files.length });
			}
		} catch (error) {
			if (typeof callback === 'function') {
				callback({ ok: false, error: error.message });
			}
		}
	});
});

httpServer.listen(3000, () => {
	console.log('WebSocket server on port 3000');
});
