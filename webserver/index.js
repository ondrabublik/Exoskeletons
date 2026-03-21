// const dgram = require('dgram');

// const server = dgram.createSocket('udp4');

// server.on('message', (msg, rinfo) => {
// 	console.log(`Received: ${msg} from ${rinfo.address}:${rinfo.port}`);

// 	try {
// 		const data = JSON.parse(msg.toString());
// 		console.log('Parsed value:', data.value);
// 	} catch (e) {
// 		console.log('Invalid JSON');
// 	}
// });

// server.on('listening', () => {
// 	const address = server.address();
// 	console.log(`UDP server listening on ${address.address}:${address.port}`);
// });

// server.bind(9999); // same port as Arduino sends to

const express = require('express');
const dgram = require('dgram');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
app.use(express.static('public'));

const udpServer = dgram.createSocket('udp4');
const httpServer = http.createServer(app);
const io = new Server(httpServer, {
	cors: { origin: "*" }
});

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
		
		// console.log('Values:', values);
		io.emit('sensor-data', values);

	} catch (ex) {
		console.log(`Bad data: ${ex}`);
	}
});

udpServer.bind(9999);

// WebSocket server
io.on('connection', (socket) => {
	console.log('Browser connected');
});

httpServer.listen(3000, () => {
	console.log('WebSocket server on port 3000');
});
