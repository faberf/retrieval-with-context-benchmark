import json
import requests


api_url = 'http://localhost:7070/api/ptt/query'

headers = {
    'Content-Type': 'application/json'
}

query_clip = {
		"operations": {
			"clip" : {"type": "RETRIEVER", "field": "clip", "input": "mytext"},
			"relations" : {"type": "TRANSFORMER", "transformerName": "RelationExpander", "input": "clip"},
			"aggregator" : {"type": "TRANSFORMER", "transformerName": "ScoreAggregator",  "input": "relations"}
		},
		"context": {
			"global": {
				"limit": "10"
			},
			"local" : {
				"averagecolor": {
					"returnDescriptor": "false",
					"limit": "12"
				}				
			}
		},
		"output": "aggregator"
    }

query_asr = {}
query_caption = {}

payload = query_clip


def query_ve(query):
    payload["inputs"] = {"mytext": {"type": "TEXT", "data": query["query"]}}

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        print("Response data:", response_data)
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response content:", response.content)


if __name__ == "__main__":
    benchmark_queries = 've-benchmarking/benchmark_queries.json'
    with open(benchmark_queries, 'r') as file:
        queries = json.load(file)

    for query in queries:
        query_ve(query)
