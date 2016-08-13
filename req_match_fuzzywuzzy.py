import io
import re
import pandas as pd
import numpy as np
from pprint import pprint
from fuzzywuzzy import process
import pos_match

pos_matcher = pos_match.PosMatch()


sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)

req_ext_df = pd.DataFrame({'ModuleNo': [int(float(requirements[i][0][:-1])) for i in range(requirements.__len__())],
                           'ReqNo': [float(requirements[i][0][:-1]) for i in range(requirements.__len__())],
                           'Req_det': [requirements[i][1] for i in range(requirements.__len__())]})

t_cases = pd.read_csv('../data/PO_TC.csv')
t_cases = t_cases[['Test Case Title','Description']]
t_cases_array = np.array(t_cases.Description)

result = [[pos_matcher.pos_match(requirement[1], t_case) for t_case in t_cases] for requirement in requirements]

# req = '''1.2. Create the Purchase Order
# Navigate: From the main menu, select Ordering > Orders. The Order Search window
# opens
# 1. In the Action field, select New Order and click OK. The PO Header Maintenance
# window opens.
# 2. In the Order Type field, select the order type.
# 3. In the Import Country field, enter the code for the import country, or click the LOV
# button and select the import country.
# 4. In the Supplier field, enter the ID of the supplier, or click the LOV button and
# select the supplier. The supplier defaults for other fields are filled in automatically.
# 5. In the Department field, enter the ID of the Department, or click the LOV button
# and select the department.
# Note: The department may be required depending on how
# the system was set up.
# 6. In the Not Before Date and Not After Date fields, enter the dates, or click the calendar
# button and select the dates.
# 7. Enter or edit the enabled fields as necessary.
# 8. Add items to the purchase order.
# 9. Distribute the items on a purchase order by diff, location, or store grade.
# 10. Click OK to save your changes and close the window'''
#
# result = [[requirement.ReqNo, process.extract(requirement.Req_det, np.array(t_cases.Description), limit=5)] for requirement in requirements]
#
# results = process.extract(req, np.array(t_cases.Description), limit=5)
#
# pprint(results)
result = [[req_head, process.extract(req_head, np.array(t_cases.Description), limit=5)] for req_head in req_heads2]