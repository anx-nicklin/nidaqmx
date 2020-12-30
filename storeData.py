import zmq
import json
from utils import *

storage = []

for i in range(num_of_ports):
    storage.append([])

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:%s" % port)
socket.setsockopt_string(zmq.SUBSCRIBE, topic)

while True:
    message = socket.recv()
    decodedMessage = message.decode("utf-8")
    data = json.loads(decodedMessage[4 :])

    for i in range(num_of_ports):
        data_list[i].extend(data["y"][i])

    print(storage)

