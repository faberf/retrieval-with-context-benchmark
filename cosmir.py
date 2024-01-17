import ujson as json
from clip import ClipVitLargePatch14
from data import load
from functools import lru_cache
import os

model = ClipVitLargePatch14()
model.load_model()
load("data.json")

from data import image_embeddings, captions, asr_transcripts, timestamps

video_description = """The video is a documentary exploring the evolution of communication and transportation technologies throughout history. It covers a range of topics from the early days of postal services and telegraphy to modern advancements in computer technology, satellite communications, and transportation methods like trains and automobiles. The narrative also weaves in elements of historical significance, such as notable figures and historical artifacts, to provide a comprehensive overview of how these technologies have shaped human civilization over time.
"""

def format_features(segment_number):
    return str({"asr_transcript": asr_transcripts[segment_number], "image_caption": captions[segment_number]})

@lru_cache(maxsize=1000)
def load_template(template_name):
    with open(f"prompts/{template_name}.txt", 'r') as f:
        return f.read()

def create_prompt(segment_number, window_size, template_name):
    segment_features = format_features(segment_number)

    # Fix for previous segments - limit to start of the list
    start_index = max(segment_number - window_size, 0)
    previous_segments = "\n".join([format_features(i) for i in range(start_index, segment_number)])

    # Fix for next segments - limit to end of the list
    end_index = min(segment_number + window_size, len(asr_transcripts) - 1)
    next_segments = "\n".join([format_features(i) for i in range(segment_number + 1, end_index + 1)])

    prompt_template = load_template(template_name)
    
    return prompt_template.format(segment_number=segment_number, segment_features=segment_features, previous_segments=previous_segments, next_segments=next_segments, video_description=video_description)


def extract_descriptions(base_path, window_size, template_name):
    from openai import OpenAI
    client = OpenAI()
    
    descriptions = []
    
    for i in range(len(asr_transcripts)):
        prompt = create_prompt(i, window_size, template_name)
        response = client.chat.completions.create(
            model="gpt-4",    
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        print(response.choices[0].message.content)
        descriptions.append(response.choices[0].message.content)
    
    output_file = f"{base_path}-ws_{window_size}-t_{template_name}.json"
    with open(output_file, 'w') as f:
        f.write("[\n"+ ',\n'.join(map( json.dumps, descriptions))+ "\n]")

def extract_embeddings(base_path, window_size, template_name):
    descriptions_file = f"{base_path}-ws_{window_size}-t_{template_name}.json"
    if not os.path.exists(descriptions_file):
        extract_descriptions(base_path, window_size, template_name)
    with open(descriptions_file, 'r') as f:
        descriptions = json.load(f)
    
    embeddings = model.text_embedding(descriptions)
    
    embeddings_file = f"{base_path}-ws_{window_size}-t_{template_name}-embeddings.json"
    with open(embeddings_file, 'w') as f:
        f.write("[\n"+ ',\n'.join(map( json.dumps, embeddings))+ "\n]")
    
if __name__ == "__main__":
    extract_embeddings("cosmir_descriptions", window_size=4, template_name="basic")