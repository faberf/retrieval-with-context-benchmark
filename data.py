import ujson as json


class Query:
    def __init__(self, text, correct_segments) -> None:
        self.text = text
        self.correct_segments = correct_segments

captions = None

asr_transcripts = None

image_embeddings = None

timestamps = None

queries= [
    Query(
        "A woman sorting through mail.",
        []
    )
]

def load(path):
    global captions, asr_transcripts, image_embeddings, timestamps
    with open(path, 'r') as f:
        data = json.load(f)
    captions = data["captions"]
    asr_transcripts = data["asr_transcripts"]
    image_embeddings = data["image_embeddings"]
    