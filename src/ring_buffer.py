class RingBuffer:
    def __init__(self, size):
        self.data = [None for i in range(size)]

    def append(self, data):
        for _ in range(len(data)):
            self.data.pop(0)
        self.data.extend(data)

    def get(self):
        return self.data

if __name__ == "__main__":
    rb = RingBuffer(30)

    while True:
        rb.append([1])
        print(rb.get())
