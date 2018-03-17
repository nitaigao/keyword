import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320

def main():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    output = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        frames_per_buffer=CHUNK,
                        output=True)

    stream.start_stream()
    output.start_stream()

    silence = chr(0) * CHUNK * CHANNELS * 2

    while True:
        available = stream.get_read_available()
        if available > 0:
            data = stream.read(CHUNK)
            output.write(data)
        else:
            output.write(silence)

    stream.stop_stream()
    stream.close()

    audio.terminate()

main()
