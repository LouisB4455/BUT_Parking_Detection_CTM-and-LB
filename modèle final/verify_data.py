import json, re

with open('monitoring_final_simple.html', 'r', encoding='utf-8') as f:
    content = f.read()
    match = re.search(r'const DATA = (\{[\s\S]*?\});', content)
    if match:
        data_str = match.group(1)
        data = json.loads(data_str)
        print("Structure DATA:")
        print(f"  - recent: {len(data.get('recent', []))} rows")
        print(f"  - alltime: {len(data.get('alltime', []))} rows")
        print(f"  - stats: {data.get('stats', {})}")
        if data.get('recent'):
            print(f"  - First recent row sample: {data['recent'][0]}")
        if data.get('alltime'):
            print(f"  - First alltime row sample: {data['alltime'][0]}")
