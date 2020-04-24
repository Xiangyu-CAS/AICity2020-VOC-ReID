import os

src_path = '/home/zxy/data/ReID/vehicle/AIC20_ReID/train_track_id.txt'

results = []
with open(src_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        line = [ele.zfill(6) + '.jpg' for ele in line]
        results.append(line)

with open(os.path.dirname(src_path) + '/train_track.txt', 'w') as f:
    for result in results:
        f.write(' '.join(result) + '\n')