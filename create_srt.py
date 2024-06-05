

if __name__ == "__main__":
    captions_path = "offline_benchmarking/cosmir_descriptions-ws_4-t_basic.json"
    data_path = "offline_benchmarking/data.json"
    
    import ujson as json
    
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        timestamps = data["segments"]
        starts = [segment["segmentstartabs"] for segment in timestamps]
        ends = [segment["segmentendabs"] for segment in timestamps]
    
    # generate srt captions
    import pysrt
    subs = pysrt.SubRipFile()
    for i in range(len(captions)):
        sub = pysrt.SubRipItem()
        sub.index = i
        sub.text = captions[i]
        sub.start.seconds = starts[i]
        sub.end.seconds = ends[i]
        subs.append(sub)
    
    # save srt captions
    subs.save("offline_benchmarking/cosmir_descriptions-ws_4-t_basic.srt")