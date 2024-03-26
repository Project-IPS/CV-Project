document.addEventListener('DOMContentLoaded', function () {
    const innCountElement = document.getElementById('inn-count');
    const outCountElement = document.getElementById('out-count');
    const totalCountElement = document.getElementById('total-count');
    const imageElement = document.getElementById('latest-image');
    const tableBody = document.getElementById('register-table').getElementsByTagName('tbody')[0];

    // Make sure this URL matches the WebSocket server address and port
    const ws = new WebSocket('ws://localhost:9876');

    ws.onopen = function () {
        console.log('Connected to the WebSocket server');
    };

    ws.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            // Update counts
            innCountElement.innerText = data.inn;
            outCountElement.innerText = data.out;
            totalCountElement.innerText = data.total;
            if (data.image) {
                imageElement.src = 'data:image/jpeg;base64,' + data.image;  // Adjust the MIME type if necessary
            }
            // Update table
            tableBody.innerHTML = ''; // Clear previous entries
            data.entries.forEach((entry, index) => {
                let newRow = tableBody.insertRow();
                newRow.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${entry.id}</td>
                    <td>${entry.time_in || 'N/A'}</td>
                    <td>${entry.time_out || 'N/A'}</td>
                    <td>${entry.duration || 'N/A'}</td> <!-- Populate duration -->
                `;
            });
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
    };

    ws.onclose = function () {
        console.log('Disconnected from the WebSocket server');
    };
});
