import ujson as json
import os

class Query:
    def __init__(self, text, correct_segments, classification) -> None:
        self.text = text
        self.correct_segments = correct_segments
        self.classification = classification

captions = None

asr_transcripts = None

image_embeddings = None

timestamps = None

caption_embeddings = None

asr_embeddings = None

query_embeddings = None

query_embeddings_clip = None

queries= [
    Query(
        "A woman sorting mail into a shelf.",
        [68],
        "visual"
    ),
    Query(
        "Beat von Fischer",
        [15],
        "audio"
    ),
    Query(
        "Parcel sorting line",
        [66, 65],
        "context"
    ),
    Query(
        "A caveman holding a keyboard in front of a TV.",
        [117],
        "visual"
    ),
    Query(
        "Sprechhörer schreit von Turm herab",
        [127],
        "context"
    ),
    Query(
        "Alexander Graham Bell",
        [143],
        "audio"
    ),
    Query(
        "Mail being loaded onto an airplane for shipping.",
        [75],
        "context"
    ),
    Query(
        "Mann macht Rauchzeichen",
        [82],
        "context"
    ),
    Query(
        "Samuel Morse",
        [87],
        "audio"
    ),
    Query(
        "Frau tippt Telegram in Computer",
        [101,102],
        "context"
    ),
    Query(
        "Device used for closing a bag of air mail",
        [69,70,71,72,73,74],
        "context"
    ),
    Query(
        "Mail being transported in the 19th century",
        [23],
        "context"
    ),
    Query(
        "Man sorting mail on a train",
        [42],
        "context"
    ),
    Query(
        "Comedic acting",
        [2,3,4,5,6,7,8,9,10,11,12,13,14,77,82,84,117,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,180,181],
        "context"
    ),
    Query(
        "Man presenting in front of a screen",
        [114,115],
        "visual"
    ),
    Query(
        "Induction Coil",
        [146, 147],
        "audio"
    ),
    Query(
        "First telephone service in zurich",
        [152],
        "audio"
    ),
    Query(
        "Junge Frau liegt auf dem Bett mit telefon in der hand",
        [97,99],
        "visual"
    ),
    Query(
        "Historial portrait",
        [15,87,143],
        "visual"
    ),
    Query(
        "Man walking in tunnel full of cables",
        [159],
        "visual"
        )
]

def load(path):
    global captions, asr_transcripts, image_embeddings, timestamps, caption_embeddings, asr_embeddings, query_embeddings, query_embeddings_clip
    with open(path, 'rb') as f:
        data = json.load(f)
    captions = data["captions"]
    asr_transcripts = data["asr_transcripts"]
    image_embeddings = data["image_embeddings"]
    timestamps = data["segments"]
    
    if os.path.exists("caption_embeddings.json"):
        with open("caption_embeddings.json", "r") as f:
            caption_embeddings = json.load(f)
    
    if os.path.exists("asr_embeddings.json"):
        with open("asr_embeddings.json", "r") as f:
            asr_embeddings = json.load(f)
    
    if os.path.exists("query_embeddings.json"):
        with open("query_embeddings.json", "r") as f:
            query_embeddings = json.load(f)
    
    if os.path.exists("query_embeddings_clip.json"):
        with open("query_embeddings_clip.json", "r") as f:
            query_embeddings_clip = json.load(f)