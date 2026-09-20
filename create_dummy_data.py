
import os
import wave
import struct

os.makedirs('test_data', exist_ok=True)
for i in range(5):
    with wave.open(f'test_data/dummy_{i}.wav', 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        # 1 second of silence
        data = struct.pack('<h', 0) * 44100
        w.writeframesraw(data)
print('Dummy data created')

