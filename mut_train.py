import numpy as np
import json
import sys

num_classes = 200
num_concept = int(sys.argv[1])
mutual_info = np.load('mutual_info_train.npy')
print(mutual_info.shape)
ids = np.argsort(mutual_info)[::-1]
with open('all_att_90.json', 'r') as fp:
    concepts = json.load(fp)
print(len(concepts))
#print(concepts[176])
class_label=np.load('class_label_des_90.npy')
print(class_label.shape)
class_label = class_label.T

class_ct = {}
class_concept_id = {}
for c in range(num_classes):
    class_ct[c]=0
    class_concept_id[c]=[]
new_concepts_id=[]
for i in range(len(class_label)):
    c_id = ids[i]
    flag = 0
    for c in class_ct:
        if class_ct[c] < 1:
            flag = 1
    if flag == 0:
        print('f',i)
        break


    for c in range(num_classes):
        if class_label[c_id][c]==1:
            if class_ct[c]>=1:
                continue
            else:
                class_ct[c] += 1
                if c_id in new_concepts_id:
                    continue
                new_concepts_id.append(c_id)
                for c in range(num_classes):
                    if class_label[c_id][c] == 1:
                        class_concept_id[c].append(c_id)

for i in range(len(ids)):
    c_id = ids[i]
    if len(new_concepts_id)>=num_concept:
        print('f', i)
        break

    for c in range(num_classes):
        if class_label[c_id][c]==1:

            class_ct[c] += 1
            if c_id in new_concepts_id:
                continue
            new_concepts_id.append(c_id)
            for c in range(num_classes):
                if class_label[c_id][c] == 1:
                    class_concept_id[c].append(c_id)
            break
#print(class_ct)
print(class_concept_id)
print(len(ids))
print(len(new_concepts_id))
print(len(set(new_concepts_id)))



new_concepts=[]
new_class_label=class_label[new_concepts_id]
for i in new_concepts_id:
    new_concepts.append(concepts[i])

new_class_label = new_class_label.T
with open('select_concept_train_'+str(num_concept)+'.json','w') as fw:
    json.dump(new_concepts,fw)
np.save('class_label_des_train_'+str(num_concept)+'.npy',new_class_label)


with open('new_concept_all.json', 'r') as fp:
    new_cub = json.load(fp)
cub_fff = {}
full_cub_fff = {}
class_names = list(new_cub.keys())
for i in range(len(class_names)):
    n = class_names[i]
    n = n.replace('-',' ')
    n = n.lower()
    cub_fff[n] = []
    full_cub_fff[n] = []

    for c_id in class_concept_id[i]:
        c = concepts[c_id]
        cub_fff[n].append(c)
        full_cub_fff[n].append('A photo of a '+ n+', which has '+c)

#with open('full_concept_'+shot+'shot'+str(num_concept)+'.json','w') as fw:
    #json.dump(full_cub_fff,fw)

tl = new_class_label.T
ct=0
for i in range(len(new_concepts)):
    if np.sum(tl[i])>1:
        ct+=1
print(ct/len(new_concepts))
