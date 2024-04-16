import json
import websocket
from datetime import datetime



# Load JSON config
with open('examples/config.json', 'r') as file:
    config = json.load(file)

class WebSocket:

    def __init__(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(config['websocket_settings']['host'])
        self.frame_count = 0
        self.inn = 0
        self.out = 0
        self.last_sent_inn = 0
        self.last_sent_out = 0
        self.entry_sent = set()
        self.crossing_time_in = None
        self.crossing_time_out = None
        self.pred_dst_pts = None

    def datetime_serializer(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError("Type not serializable")

    def format_datetime(self, dt):
        return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else 'N/A'

    def calculate_duration(self, time_in, time_out):
        """Calculate the duration between time_in and time_out."""
        # Check if both times are not None
        if time_in and time_out:
            # Calculate the duration
            duration = time_out - time_in
            # Format the duration in hours, minutes, and seconds
            return str(duration)
        return 'N/A'

    def Response(self):
        self.frame_count += 1
        data_to_send = {}

        if self.inn != self.last_sent_inn:
            data_to_send['inn'] = self.inn

        if self.out != self.last_sent_out:
            data_to_send['out'] = self.out
            data_to_send['total'] = self.inn - self.out
            self.last_sent_inn, self.last_sent_out = self.inn, self.out

        # Prepare and send entry data only once when an object enters
        entries = []
        for object_id in set(self.crossing_time_in) - self.entry_sent:  # Only send new entries
            entry_data = {
                'id': object_id,
                'time_in': self.format_datetime(self.crossing_time_in.get(object_id))
            }
            entries.append(entry_data)
            self.entry_sent.add(object_id)

        # Send location updates every 15 frames
        if self.frame_count % 15 == 0:
            for object_id in self.pred_dst_pts:
                location_data = {
                    'id': object_id,
                    'Location': self.pred_dst_pts.get(object_id)
                }
                entries.append(location_data)

        # Prepare and send exit data only once when an object leaves
        for object_id in set(self.crossing_time_out):
            if object_id in self.crossing_time_in:  # Ensure the object has a recorded entry
                exit_data = {
                    'id': object_id,
                    'time_out': self.format_datetime(self.crossing_time_out.get(object_id)),
                    'duration': self.calculate_duration(self.crossing_time_in.get(object_id),
                                                        self.crossing_time_out.get(object_id))
                }
                entries.append(exit_data)

                # Once the object has left, we can remove it from entry_sent and exit_sent
                self.entry_sent.discard(object_id)

        if entries:
            data_to_send['entries'] = entries

        # Convert to JSON string
        data_json = json.dumps(data_to_send)

        # Send over WebSocket
        self.ws.send(data_json)