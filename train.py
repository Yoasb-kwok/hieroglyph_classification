from ultralytics import YOLO, checks, hub
checks()

hub.login('cd0305b43101b80789be164d6ee40b63128244e9af')

model = YOLO('https://hub.ultralytics.com/models/g0C0KQKkX3y9D6iunMIo')
results = model.train()