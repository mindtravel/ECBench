import json
import argparse

type_list = {
    'SB': [
        "Scene State", "Scene Function", "Attribute Recognition", "Comparison", "Counting", 
        "Object Recognition", "OCR", "Spatial Relationship", "Function Reasoning", "State Reasoning",
        "Spatial Reasoning", "Mathematical Reasoning"
    ],
    'RC': [
        "Trajectory Review", "Temporal Reasoning", "Azimuth Awareness", "Distance Awareness",
        "Spatial Imagery", "Movement Imagery", "Causal Imagery"
    ],
    'static_scene': [
        "Scene State", "Scene Function", "Attribute Recognition", "Comparison", "Counting", 
        "Object Recognition", "OCR", "Spatial Relationship", "Function Reasoning", "State Reasoning",
        "Spatial Reasoning", "Mathematical Reasoning","Trajectory Review", "Temporal Reasoning", "Azimuth Awareness", "Distance Awareness",
        "Spatial Imagery", "Movement Imagery", "Causal Imagery"
    ],
    'CS': [
        "Comparative Common Sense", "Co-occurrence Common Sense", "Object Attribute Common Sense", "Spatial Common Sense"
    ],
    'UI': [
        "Ambiguous Reference", "Erroneous Reference", "Missing Reference"
    ],
    'hallucination': [
        "Comparative Common Sense", "Co-occurrence Common Sense", "Object Attribute Common Sense", "Spatial Common Sense",
        "Ambiguous Reference", "Erroneous Reference", "Missing Reference"
    ],
    'dynamic': [
        "Information Change", "Quantity Change", "Spatial Relationship Change", "State Change"
    ],
    'ID': [
        "Information Change"
    ],
    'QD': [
        "Quantity Change"
    ],
    'SPD':[
        "Spatial Relationship Change"
    ],
    'STD':[
        "State Change"
    ]
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--answer', default='output.json')  
    args = parser.parse_args()  
    
    sum = {}
    num = {}
    score = {}
    data = json.load(open(args.answer))
    for d in data:
        # 只统计有 multi_granularity_score 的条目
        if d.get('multi_granularity_score') is None:
            continue
        flag = 0
        dimensions = d['dimensions'].replace('_', ' ').replace("['", '').replace("']", '')
        for key_ in type_list.keys():
            if dimensions in type_list[key_]:
                flag = 1
                if key_ not in sum:
                    sum[key_] = 0
                    num[key_] = 0

                sum[key_] += d['multi_granularity_score']
                num[key_] += 1
            
    nums = 0
    sums = 0
    for key_ in sum.keys():
        score[key_] = sum[key_]/num[key_]
        nums+=num[key_]
        sums+=sum[key_]
        
    print(f"{'Category':<20} {'Value':>8}")
    print("-" * 30)
    for category, value in score.items():
        print(f"{category:<20} {value*100:>7.2f}")
    print(f"{'Overall Mean':<20} {sums/nums*100:>7.2f}")
