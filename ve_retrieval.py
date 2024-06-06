import json
import requests
import os

api_url = 'http://localhost:7070/api/ptt/query'

headers = {
    'Content-Type': 'application/json'
}


def create_query_template(field):
    return {
        "operations": {
            "feature": {"type": "RETRIEVER", "field": field, "input": "text"},
            "relations": {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "feature"},
            "lookup": {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "relations"},
            "aggregator": {"type": "TRANSFORMER", "transformerName": "ScoreAggregator", "input": "lookup"},
            "filelookup": {"type": "TRANSFORMER", "transformerName": "FieldLookup", "input": "aggregator"}
        },
        "context": {
            "global": {"limit": "1000"},
            "local": {
                "lookup": {"field": "time", "keys": "start, end"},
                "relations": {"outgoing": "partOf"},
                "filelookup": {"field": "file", "keys": "path"}
            }
        },
        "output": "filelookup"
    }


query_clip = create_query_template("clip")
query_asr = create_query_template("asr")
query_caption = create_query_template("caption")
query_ocr = create_query_template("ocr")

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
        

def combined_query_ve(query, query_template, weights):
    query_template["inputs"] = {
                "clip": {"type": "TEXT", "data": query["query"]},
                "ocr": {"type": "TEXT", "data": query["query"]},
                "asr": {"type": "TEXT", "data": query["query"]},
                "caption": {"type": "TEXT", "data": query["query"]}
        }
    query_template["context"]["local"]["score"]["weights"] = weights
    return query_ve(query, query_template)


def save_results(results, results_path, filename):
    with open(os.path.join(results_path, filename), 'w') as out:
        json.dump(results, out)


if __name__ == "__main__":
    benchmark_queries = 've-benchmarking/benchmark_queries.json'
    results_path = 've-benchmarking/query_results/'
    
    with open(benchmark_queries, 'r') as file:
        queries = json.load(file)

    weight_clip = 0.5
    weight_ocr = 0.2
    weight_asr = 0.2
    weight_caption = 0.1
    weights = f"{weight_clip},{weight_ocr},{weight_asr},{weight_caption}"


    results = {
        "clip": [],
        "asr": [],
        "caption": [],
        "ocr": [],
        "mix": []
    }

    for query in queries:
        results["clip"].append(single_query_ve(query, query_clip))
        results["asr"].append(single_query_ve(query, query_asr))
        results["caption"].append(single_query_ve(query, query_caption))
        results["ocr"].append(single_query_ve(query, query_ocr))
        results["mix"].append(combined_query_ve(query, query_mix, weights))

    save_results(results["clip"], results_path, 'clip.json')
    save_results(results["asr"], results_path, 'asr.json')
    save_results(results["caption"], results_path, 'caption.json')
    save_results(results["ocr"], results_path, 'ocr.json')
    save_results(results["mix"], results_path, 'mix.json')
