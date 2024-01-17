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
        "A woman sorting mail into ashelf.",
        [68]
    ),
    Query(
        "Beat von Fischer",
        [15]
    ),
    Query(
        "Parcel sorting line",
        [66]
    ),
    Query(
        "A caveman holding a keyboard in front of a TV.",
        [117]
    ),
    Query(
        "Sprechhörer schreit von Turm herab",
        [127]
    ),
    Query(
        "Alexander Graham Bell",
        [143]
    ),
    Query(
        "Mail being loaded onto an airplane for shipping.",
        [75]
    ),
    Query(
        "Mann macht Rauchzeichen",
        [82]
    ),
    Query(
        "Samuel Morse",
        [87]
    ),
    Query(
        "Frau tippt Telegram in Computer",
        [101]
    ),
    Query(
        "Device used for closing a bag of air mail.",
        [70]
    )
]

def load(path):
    global captions, asr_transcripts, image_embeddings, timestamps
    with open(path, 'r') as f:
        data = json.load(f)
    captions = data["captions"]
    asr_transcripts = data["asr_transcripts"]
    image_embeddings = data["image_embeddings"]
    