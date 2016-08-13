import pandas as pd
import numpy as np
import re, io
from pprint import pprint
import reqtcscore

score_dist = reqtcscore.ReqTCScore()

sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)
reqs_mod_1 = [req[1] for req in requirements if int(float(req[0][:-1]))==1]

t_cases = pd.read_csv('../data/PO_TC.csv')
documents = list(t_cases.Description)

req_dist = score_dist.dist_score(reqs_mod_1, documents)

dist_sort = [sorted(d[1], key=lambda x: x[1], reverse=True)[:5] for d in req_dist]

tc_map = [[(i, documents[i], score) for (i, score) in tc] for tc in dist_sort]
tc_map_df = pd.DataFrame(tc_map)
tc_map_df['Requirement'] = pd.Series(reqs_mod_1, index=tc_map_df.index)
tc_map_df['Score'] = [np.mean(tc[1]) for tc in dist_sort]
# print(type(documents[1]))
tc_map_df.to_csv('../data/output_test.csv')
pprint(tc_map_df)
pprint(tc_map)

pprint(req_dist)
pprint(dist_sort)