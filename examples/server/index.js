const WebSocket = require('ws');

const wss = new WebSocket.Server({
    port: 9876
}, function(){
    console.log('WebSocket server is ready');
});

wss.on('connection', function(ws){
    console.log('A client connected');

    ws.on('message', function(data, isBinary){
        console.log('Received: ' + data);

        // Broadcast the received data to all connected clients
        wss.clients.forEach(function each(client) {
            if (client.readyState === WebSocket.OPEN) {
                client.send(data, { binary: isBinary });
            }
        });
    });

});
