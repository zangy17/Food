import json
with open('cub_90.json','r') as fp:
    cub=json.load(fp)
full_cub = {}
for c in cub:
    n = c.lower().replace('-',' ')
    full_cub[n] = []
    for t in cub[c]:
        full_cub[n].append('A photo of a '+n+', which has '+t)
with open('full_concept_90.json','w') as fw:
    json.dump(full_cub,fw)
