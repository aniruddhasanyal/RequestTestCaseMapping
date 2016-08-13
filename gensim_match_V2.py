import io
import re
import pandas as pd
from pprint import pprint
from pos_cos_dist import PosDist

dist_calc = PosDist()

sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)
reqs = [req[1] for req in requirements]

t_cases = pd.read_csv('../data/PO_TC.csv')
documents = list(t_cases.Description)

req_tc_dist = dist_calc.score_mat(reqs, documents)

pprint(req_tc_dist)