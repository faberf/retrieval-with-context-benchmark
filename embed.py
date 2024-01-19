import ujson as json
from multilingual_e5_large import MultilingualE5Large
from clip import ClipVitLargePatch14
from data import load

model = MultilingualE5Large()
clip_model = ClipVitLargePatch14()
load("data.json")

from data import queries, image_embeddings, captions, asr_transcripts, timestamps


if __name__ == "__main__":
    
    task = "Given a search query, find the description of the video segment that best matches the query."
    
    model.load_model()
    # caption_embeddings = [model.document_embedding(caption) for caption in captions]
    # asr_embeddings = [model.document_embedding(asr) for asr in asr_transcripts]
    query_embeddings = [model.query_embedding(query.text) for query in queries]
    
    clip_model.load_model()
    query_embeddings_clip = [clip_model.text_embedding(query.text) for query in queries]
    
    
    # with open("caption_embeddings.json", "w") as f:
    #     json.dump(caption_embeddings, f)
    # with open("asr_embeddings.json", "w") as f:
    #     json.dump(asr_embeddings, f)
    with open("query_embeddings.json", "w") as f:
        json.dump(query_embeddings, f)
    
    with open("query_embeddings_clip.json", "w") as f:
        json.dump(query_embeddings_clip, f)
    
    print("done")