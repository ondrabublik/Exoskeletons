const express = require('express');
const dgram = require('dgram');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { Server } = require('socket.io');

const app = express();
app.use(express.static('public'));

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
		sampleCount: 0
	};

	io.emit('recording:status', getRecordingStatus());
	return getRecordingStatus();
}

function stopRecording() {
	if (!activeRecording) {
		return getRecordingStatus();
	}

	const recording = activeRecording;
	activeRecording = null;

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
			activeRecording.stream.write(`${packet.timestamp},${values.join(',')}\n`);
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
	console.log('Browser connected');

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
});

httpServer.listen(3000, () => {
	console.log('WebSocket server on port 3000');
});
