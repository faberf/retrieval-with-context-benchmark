import json

# Input data
segments = [
    {"segmentstartabs": 0.0, "segmentendabs": 0.14},
    {"segmentstartabs": 0.16, "segmentendabs": 4.62},
    {"segmentstartabs": 4.64, "segmentendabs": 5.5},
    {"segmentstartabs": 5.52, "segmentendabs": 5.62},
    {"segmentstartabs": 5.64, "segmentendabs": 9.5},
    {"segmentstartabs": 9.52, "segmentendabs": 9.7},
    {"segmentstartabs": 9.72, "segmentendabs": 9.98},
    {"segmentstartabs": 10.0, "segmentendabs": 10.06},
    {"segmentstartabs": 10.08, "segmentendabs": 10.82},
    {"segmentstartabs": 10.84, "segmentendabs": 13.1},
    {"segmentstartabs": 13.12, "segmentendabs": 18.06},
    {"segmentstartabs": 18.08, "segmentendabs": 21.22},
    {"segmentstartabs": 21.24, "segmentendabs": 23.06},
    {"segmentstartabs": 23.08, "segmentendabs": 26.06},
    {"segmentstartabs": 26.08, "segmentendabs": 36.46},
    {"segmentstartabs": 36.48, "segmentendabs": 49.66},
    {"segmentstartabs": 49.68, "segmentendabs": 55.3},
    {"segmentstartabs": 55.32, "segmentendabs": 62.5},
    {"segmentstartabs": 62.52, "segmentendabs": 67.22},
    {"segmentstartabs": 67.24, "segmentendabs": 67.3},
    {"segmentstartabs": 67.32, "segmentendabs": 67.66},
    {"segmentstartabs": 67.68, "segmentendabs": 67.74},
    {"segmentstartabs": 67.76, "segmentendabs": 71.9},
    {"segmentstartabs": 71.92, "segmentendabs": 78.66},
    {"segmentstartabs": 78.68, "segmentendabs": 80.1},
    {"segmentstartabs": 80.12, "segmentendabs": 80.18},
    {"segmentstartabs": 80.2, "segmentendabs": 81.18},
    {"segmentstartabs": 81.2, "segmentendabs": 81.34},
    {"segmentstartabs": 81.36, "segmentendabs": 81.62},
    {"segmentstartabs": 81.64, "segmentendabs": 82.54},
    {"segmentstartabs": 82.56, "segmentendabs": 82.82},
    {"segmentstartabs": 82.84, "segmentendabs": 83.34},
    {"segmentstartabs": 83.36, "segmentendabs": 83.62},
    {"segmentstartabs": 83.64, "segmentendabs": 84.02},
    {"segmentstartabs": 84.04, "segmentendabs": 84.98},
    {"segmentstartabs": 85.0, "segmentendabs": 86.46},
    {"segmentstartabs": 86.48, "segmentendabs": 90.1},
    {"segmentstartabs": 90.12, "segmentendabs": 91.78},
    {"segmentstartabs": 91.8, "segmentendabs": 91.98},
    {"segmentstartabs": 92.0, "segmentendabs": 92.18},
    {"segmentstartabs": 92.2, "segmentendabs": 93.46},
    {"segmentstartabs": 93.48, "segmentendabs": 94.5},
    {"segmentstartabs": 94.52, "segmentendabs": 103.94},
    {"segmentstartabs": 103.96, "segmentendabs": 114.26},
    {"segmentstartabs": 114.28, "segmentendabs": 114.46},
    {"segmentstartabs": 114.48, "segmentendabs": 114.82},
    {"segmentstartabs": 114.84, "segmentendabs": 115.02},
    {"segmentstartabs": 115.04, "segmentendabs": 115.22},
    {"segmentstartabs": 115.24, "segmentendabs": 115.38},
    {"segmentstartabs": 115.4, "segmentendabs": 116.22},
    {"segmentstartabs": 116.24, "segmentendabs": 116.7},
    {"segmentstartabs": 116.72, "segmentendabs": 116.82},
    {"segmentstartabs": 116.84, "segmentendabs": 117.46},
    {"segmentstartabs": 117.48, "segmentendabs": 117.7},
    {"segmentstartabs": 117.72, "segmentendabs": 118.1},
    {"segmentstartabs": 118.12, "segmentendabs": 123.54},
    {"segmentstartabs": 123.56, "segmentendabs": 125.86},
    {"segmentstartabs": 125.88, "segmentendabs": 126.18},
    {"segmentstartabs": 126.2, "segmentendabs": 126.7},
    {"segmentstartabs": 126.72, "segmentendabs": 131.86},
    {"segmentstartabs": 131.88, "segmentendabs": 133.34},
    {"segmentstartabs": 133.36, "segmentendabs": 134.5},
    {"segmentstartabs": 134.52, "segmentendabs": 134.94},
    {"segmentstartabs": 134.96, "segmentendabs": 135.18},
    {"segmentstartabs": 135.2, "segmentendabs": 139.06},
    {"segmentstartabs": 139.08, "segmentendabs": 146.7},
    {"segmentstartabs": 146.72, "segmentendabs": 152.42},
    {"segmentstartabs": 152.44, "segmentendabs": 165.1},
    {"segmentstartabs": 165.12, "segmentendabs": 171.5},
    {"segmentstartabs": 171.52, "segmentendabs": 171.94},
    {"segmentstartabs": 171.96, "segmentendabs": 174.1},
    {"segmentstartabs": 174.12, "segmentendabs": 175.18},
    {"segmentstartabs": 175.2, "segmentendabs": 175.66},
    {"segmentstartabs": 175.68, "segmentendabs": 175.9},
    {"segmentstartabs": 175.92, "segmentendabs": 176.66},
    {"segmentstartabs": 176.68, "segmentendabs": 190.58},
    {"segmentstartabs": 190.6, "segmentendabs": 200.82},
    {"segmentstartabs": 200.84, "segmentendabs": 211.06},
    {"segmentstartabs": 211.08, "segmentendabs": 213.9},
    {"segmentstartabs": 213.92, "segmentendabs": 215.7},
    {"segmentstartabs": 215.72, "segmentendabs": 221.7},
    {"segmentstartabs": 221.72, "segmentendabs": 226.42},
    {"segmentstartabs": 226.44, "segmentendabs": 241.1},
    {"segmentstartabs": 241.12, "segmentendabs": 242.02},
    {"segmentstartabs": 242.04, "segmentendabs": 248.78},
    {"segmentstartabs": 248.8, "segmentendabs": 253.94},
    {"segmentstartabs": 253.96, "segmentendabs": 263.34},
    {"segmentstartabs": 263.36, "segmentendabs": 269.82},
    {"segmentstartabs": 269.84, "segmentendabs": 276.78},
    {"segmentstartabs": 276.8, "segmentendabs": 281.06},
    {"segmentstartabs": 281.08, "segmentendabs": 285.9},
    {"segmentstartabs": 285.92, "segmentendabs": 291.14},
    {"segmentstartabs": 291.16, "segmentendabs": 293.98},
    {"segmentstartabs": 294.0, "segmentendabs": 294.18},
    {"segmentstartabs": 294.2, "segmentendabs": 300.78},
    {"segmentstartabs": 300.8, "segmentendabs": 305.54},
    {"segmentstartabs": 305.56, "segmentendabs": 310.02},
    {"segmentstartabs": 310.04, "segmentendabs": 315.38},
    {"segmentstartabs": 315.4, "segmentendabs": 316.94},
    {"segmentstartabs": 316.96, "segmentendabs": 320.58},
    {"segmentstartabs": 320.6, "segmentendabs": 324.26},
    {"segmentstartabs": 324.28, "segmentendabs": 328.46},
    {"segmentstartabs": 328.48, "segmentendabs": 334.22},
    {"segmentstartabs": 334.24, "segmentendabs": 339.94},
    {"segmentstartabs": 339.96, "segmentendabs": 344.1},
    {"segmentstartabs": 344.12, "segmentendabs": 344.26},
    {"segmentstartabs": 344.28, "segmentendabs": 348.82},
    {"segmentstartabs": 348.84, "segmentendabs": 352.18},
    {"segmentstartabs": 352.2, "segmentendabs": 354.78},
    {"segmentstartabs": 354.8, "segmentendabs": 366.3},
    {"segmentstartabs": 366.32, "segmentendabs": 368.9},
    {"segmentstartabs": 368.92, "segmentendabs": 377.58},
    {"segmentstartabs": 377.6, "segmentendabs": 384.82},
    {"segmentstartabs": 384.84, "segmentendabs": 389.14},
    {"segmentstartabs": 389.16, "segmentendabs": 403.58},
    {"segmentstartabs": 403.6, "segmentendabs": 418.1},
    {"segmentstartabs": 418.12, "segmentendabs": 420.82},
    {"segmentstartabs": 420.84, "segmentendabs": 429.02},
    {"segmentstartabs": 429.04, "segmentendabs": 430.54},
    {"segmentstartabs": 430.56, "segmentendabs": 432.02},
    {"segmentstartabs": 432.04, "segmentendabs": 434.18},
    {"segmentstartabs": 434.2, "segmentendabs": 436.1},
    {"segmentstartabs": 436.12, "segmentendabs": 440.22},
    {"segmentstartabs": 440.24, "segmentendabs": 440.26},
    {"segmentstartabs": 440.28, "segmentendabs": 447.78},
    {"segmentstartabs": 447.8, "segmentendabs": 448.74},
    {"segmentstartabs": 448.76, "segmentendabs": 449.26},
    {"segmentstartabs": 449.28, "segmentendabs": 452.54},
    {"segmentstartabs": 452.56, "segmentendabs": 455.34},
    {"segmentstartabs": 455.36, "segmentendabs": 464.1},
    {"segmentstartabs": 464.12, "segmentendabs": 464.18},
    {"segmentstartabs": 464.2, "segmentendabs": 464.5},
    {"segmentstartabs": 464.52, "segmentendabs": 465.02},
    {"segmentstartabs": 465.04, "segmentendabs": 467.1},
    {"segmentstartabs": 467.12, "segmentendabs": 468.14},
    {"segmentstartabs": 468.16, "segmentendabs": 469.38},
    {"segmentstartabs": 469.4, "segmentendabs": 472.98},
    {"segmentstartabs": 473.0, "segmentendabs": 473.34},
    {"segmentstartabs": 473.36, "segmentendabs": 473.62},
    {"segmentstartabs": 473.64, "segmentendabs": 473.82},
    {"segmentstartabs": 473.84, "segmentendabs": 474.82},
    {"segmentstartabs": 474.84, "segmentendabs": 479.22},
    {"segmentstartabs": 479.24, "segmentendabs": 479.5},
    {"segmentstartabs": 479.52, "segmentendabs": 484.02},
    {"segmentstartabs": 484.04, "segmentendabs": 487.18},
    {"segmentstartabs": 487.2, "segmentendabs": 494.18},
    {"segmentstartabs": 494.2, "segmentendabs": 499.1},
    {"segmentstartabs": 499.12, "segmentendabs": 506.54},
    {"segmentstartabs": 506.56, "segmentendabs": 513.26},
    {"segmentstartabs": 513.28, "segmentendabs": 517.98},
    {"segmentstartabs": 518.0, "segmentendabs": 518.02},
    {"segmentstartabs": 518.04, "segmentendabs": 525.78},
    {"segmentstartabs": 525.8, "segmentendabs": 533.86},
    {"segmentstartabs": 533.88, "segmentendabs": 539.7},
    {"segmentstartabs": 539.72, "segmentendabs": 551.86},
    {"segmentstartabs": 551.88, "segmentendabs": 565.1},
    {"segmentstartabs": 565.12, "segmentendabs": 568.02},
    {"segmentstartabs": 568.04, "segmentendabs": 568.1},
    {"segmentstartabs": 568.12, "segmentendabs": 570.62},
    {"segmentstartabs": 570.64, "segmentendabs": 577.18},
    {"segmentstartabs": 577.2, "segmentendabs": 577.54},
    {"segmentstartabs": 577.56, "segmentendabs": 586.18},
    {"segmentstartabs": 586.2, "segmentendabs": 589.3},
    {"segmentstartabs": 589.32, "segmentendabs": 595.7},
    {"segmentstartabs": 595.72, "segmentendabs": 599.78},
    {"segmentstartabs": 599.8, "segmentendabs": 604.7},
    {"segmentstartabs": 604.72, "segmentendabs": 608.9},
    {"segmentstartabs": 608.92, "segmentendabs": 620.1},
    {"segmentstartabs": 620.12, "segmentendabs": 626.82},
    {"segmentstartabs": 626.84, "segmentendabs": 629.46},
    {"segmentstartabs": 629.48, "segmentendabs": 632.66},
    {"segmentstartabs": 632.68, "segmentendabs": 634.3},
    {"segmentstartabs": 634.32, "segmentendabs": 635.62},
    {"segmentstartabs": 635.64, "segmentendabs": 641.22},
    {"segmentstartabs": 641.24, "segmentendabs": 641.26},
    {"segmentstartabs": 641.28, "segmentendabs": 642.82},
    {"segmentstartabs": 642.84, "segmentendabs": 645.18},
    {"segmentstartabs": 645.2, "segmentendabs": 645.58},
    {"segmentstartabs": 645.6, "segmentendabs": 647.42},
    {"segmentstartabs": 647.44, "segmentendabs": 655.42},
    {"segmentstartabs": 655.44, "segmentendabs": 670.18},
    {"segmentstartabs": 670.2, "segmentendabs": 671.7},
    {"segmentstartabs": 671.72, "segmentendabs": 672.14},
    {"segmentstartabs": 672.16, "segmentendabs": 672.26},
    {"segmentstartabs": 672.28, "segmentendabs": 674.66},
    {"segmentstartabs": 674.68, "segmentendabs": 676.9}
]

queries = [
    {"text": "A caveman holding a keyboard in front of a TV.", "indices": [117], "tag": "visual"},
    {"text": "Sprechhörer schreit von Turm herab", "indices": [127], "tag": "context"},
    {"text": "Alexander Graham Bell", "indices": [143], "tag": "audio"},
    {"text": "Mail being loaded onto an airplane for shipping.", "indices": [75], "tag": "context"},
    {"text": "Mann macht Rauchzeichen", "indices": [82], "tag": "context"},
    {"text": "Samuel Morse", "indices": [87], "tag": "audio"},
    {"text": "Frau tippt Telegram in Computer", "indices": [101, 102], "tag": "context"},
    {"text": "Device used for closing a bag of air mail", "indices": [69, 70, 71, 72, 73, 74], "tag": "context"},
    {"text": "Mail being transported in the 19th century", "indices": [23], "tag": "context"},
    {"text": "Man sorting mail on a train", "indices": [42], "tag": "context"},
    {"text": "Comedic acting", "indices": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 77, 82, 84, 117, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 180, 181], "tag": "context"},
    {"text": "Man presenting in front of a screen", "indices": [114, 115], "tag": "visual"},
    {"text": "Induction Coil", "indices": [146, 147], "tag": "audio"},
    {"text": "First telephone service in zurich", "indices": [152], "tag": "audio"},
    {"text": "Junge Frau liegt auf dem Bett mit telefon in der hand", "indices": [97, 99], "tag": "visual"},
    {"text": "Historical portrait", "indices": [15, 87, 143], "tag": "visual"},
    {"text": "Man walking in tunnel full of cables", "indices": [159], "tag": "visual"}
]

# Function to merge contiguous segments
def merge_contiguous_segments(segment_indices, segments):
    if not segment_indices:
        return []
    
    merged_segments = []
    segment_indices = sorted(segment_indices)
    
    start = segments[segment_indices[0]]["segmentstartabs"]
    end = segments[segment_indices[0]]["segmentendabs"]
    
    for i in range(1, len(segment_indices)):
        current_start = segments[segment_indices[i]]["segmentstartabs"]
        current_end = segments[segment_indices[i]]["segmentendabs"]
        
        if current_start <= end + 0.1:
            end = max(end, current_end)
        else:
            merged_segments.append([start, end])
            start = current_start
            end = current_end
    
    merged_segments.append([start, end])
    return merged_segments

# Process the queries
processed_queries = []
for query in queries:
    segment_indices = query["indices"]
    merged_segments = merge_contiguous_segments(segment_indices, segments)
    processed_queries.append({
        "text": query["text"],
        "segments": merged_segments,
        "tag": query["tag"]
    })

# Output the result as JSON
output_json = json.dumps(processed_queries, indent=4)
print(output_json)