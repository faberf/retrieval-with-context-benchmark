import json
import requests


api_url = 'http://localhost:7070/api/ptt/query'

headers = {
    'Content-Type': 'application/json'
}

query_clip = {
        "operations": {
            "clip" : {"type": "RETRIEVER", "field": "clip", "input": "text"},
            "relations" : {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "clip"},
            "lookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "relations"},
            "aggregator" : {"type": "TRANSFORMER", "transformerName": "ScoreAggregator",  "input": "lookup"},
            "filelookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
        },
        "context": {
            "global": {
                "limit": "1000"
            },
            "local" : {
                "lookup":{"field": "time", "keys": "start, end"},
                "relations" :{"outgoing": "partOf"},				
                "filelookup": {"field": "file", "keys": "path"}
            }
        },
        "output": "filelookup"
}

query_asr = {
        "operations": {
            "feature" : {"type": "RETRIEVER", "field": "asr", "input": "text"},
            "relations" : {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "feature"},
            "lookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "relations"},
            "aggregator" : {"type": "TRANSFORMER", "transformerName": "ScoreAggregator",  "input": "lookup"},
            "filelookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
        },
        "context": {
            "global": {
                "limit": "1000"
            },
            "local" : {
                "lookup": {"field": "time", "keys": "start, end"},
                "relations" : {"outgoing": "partOf"},				
                "filelookup": {"field": "file", "keys": "path"}
            }
        },
        "output": "filelookup"
}

query_caption = {
        "operations": {
            "feature" : {"type": "RETRIEVER", "field": "caption", "input": "text"},
            "relations" : {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "feature"},
            "lookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "relations"},
            "aggregator" : {"type": "TRANSFORMER", "transformerName": "ScoreAggregator",  "input": "lookup"},
            "filelookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
        },
        "context": {
            "global": {
                "limit": "1000"
            },
            "local" : {
                "lookup": {"field": "time", "keys": "start, end"},
                "relations" : {"outgoing": "partOf"},				
                "filelookup": {"field": "file", "keys": "path"}
            }
        },
        "output": "filelookup"
}

query_ocr = {
        "operations": {
            "feature" : {"type": "RETRIEVER", "field": "ocr", "input": "text"},
            "relations" : {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "feature"},
            "lookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "relations"},
            "aggregator" : {"type": "TRANSFORMER", "transformerName": "ScoreAggregator",  "input": "lookup"},
            "filelookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
        },
        "context": {
            "global": {
                "limit": "1000"
            },
            "local" : {
                "lookup": {"field": "time", "keys": "start, end"},
                "relations" : {"outgoing": "partOf"},				
                "filelookup": {"field": "file", "keys": "path"}
            }
        },
        "output": "filelookup"
}

query_mix = {
        "operations": {
            "feature1" : {"type": "RETRIEVER", "field": "clip", "input": "clip"},
            "feature2" : {"type": "RETRIEVER", "field": "ocr", "input": "ocr"},
            "feature3" : {"type": "RETRIEVER", "field": "asr", "input": "asr"},
            "feature4" : {"type": "RETRIEVER", "field": "caption", "input": "caption"},
            "score" : {"type": "AGGREGATOR", "aggregatorName": "WeightedScoreFusion", "inputs": ["feature1", "feature2","feature3", "feature4"]},
            "relations" : {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "score"},
            "lookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "relations"},
            "aggregator" : {"type": "TRANSFORMER", "transformerName": "ScoreAggregator",  "input": "lookup"},
            "filelookup" : {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
        },
        "context": {
            "global": {
                "limit": "1000"
            },
            "local" : {
                "lookup": {"field": "time", "keys": "start, end"},
                "relations" : {"outgoing": "partOf"},				
                "filelookup": {"field": "file", "keys": "path"},
                "score": {"weights": "0.5,0.2,0.2,0.1"}
            }
        },
        "output": "filelookup"
}


def query_ve(query, query_template):
    response = requests.post(api_url, json=query_template, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        return {
            "query": query,
            "response": response_data
        }
    
    else:
        print(f"Request failed with status code {response.status_code}")
        print(f"Query text: {query['query']}")
        print("Response content:", response.content)
    

def single_query_ve(query, query_template):
    query_template["inputs"] = {"text": {"type": "TEXT", "data": query["query"]}}
    return query_ve(query, query_template)
        

def combined_query_ve(query, query_template):
    query_template["inputs"] = {
                "clip": {"type": "TEXT", "data": query["query"]},
                "ocr": {"type": "TEXT", "data": query["query"]},
                "asr": {"type": "TEXT", "data": query["query"]},
                "caption": {"type": "TEXT", "data": query["query"]}
        }
    return query_ve(query, query_template)


if __name__ == "__main__":
    benchmark_queries = 've-benchmarking/benchmark_queries.json'
    results_path = 've-benchmarking/query_results/'
    
    with open(benchmark_queries, 'r') as file:
        queries = json.load(file)

    results_clip = []
    results_asr = []
    results_caption = []
    results_ocr = []
    results_mix = []
    for query in queries:
        results_clip.append(single_query_ve(query, query_clip))
        results_asr.append(single_query_ve(query, query_asr))
        results_caption.append(single_query_ve(query, query_caption))
        results_ocr.append(single_query_ve(query, query_ocr))
        results_mix.append(combined_query_ve(query, query_mix))
        
    with open(results_path + 'clip.json', 'w') as out:
        json.dump(results_clip, out)
    with open(results_path + 'asr.json', 'w') as out:
        json.dump(results_asr, out)
    with open(results_path + 'caption.json', 'w') as out:
        json.dump(results_caption, out)
    with open(results_path + 'ocr.json', 'w') as out:
        json.dump(results_ocr, out)
    with open(results_path + 'mix.json', 'w') as out:
        json.dump(results_mix, out)
