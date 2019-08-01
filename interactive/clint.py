import requests
import pyaudio
import wave

framerate = 16000
NUM_SAMPLES = 5000
channels = 1
sampwidth = 2
TIME = 10

server = "http://localhost:5001/recognize"


def save_wave_file(filename, data):
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()


def record(f, time=5):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=framerate,
        input=True,
        frames_per_buffer=NUM_SAMPLES,
    )
    my_buf = []
    count = 0
    print(f"录音中({time}s)")
    while count < TIME * time:
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count += 1
        print(".", end="", flush=True)

    save_wave_file(f, my_buf)
    stream.close()



record("record.wav", time=5)  # modify time to how long you want

f = open("record.wav", "rb")

files = {"file": f}

r = requests.post(server, files=files)

print("")
print("识别结果:")
print(r.text)