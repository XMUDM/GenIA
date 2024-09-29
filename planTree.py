import time
import math
import json

NOTE_TYPE = [
    'SetOp',
    'Append',
    'Unique',
    'Subquery Scan',
    'WindowAgg',
    'Gather',
    'Gather Merge',
    'Result',
    'Limit',
    'Materialize',
    'Aggregate',
    'Seq Scan',
    'Sort',
    'Nested Loop',
    'Hash',
    'Hash Join',
    'Index Scan',
    'Index Only Scan',
    'Bitmap Heap Scan',
    'Bitmap Index Scan',
    'Merge Join',
    'CTE Scan',
    'Merge Append',
    'Group',
]
USEFUL_TYPE = [
    'Nested Loop',
    'Merge Join',
    'Hash Join',
    'Seq Scan',
    'Sort',
    'Aggregate',
    'CTE Scan',
    'Merge Append',
    'Group',
]


class PlanTreeNode:
    def __init__(self, node_type=None, cost=None, plan_rows=None, plan_width=None):
        self.node_type = node_type
        self.cost = cost
        self.plan_rows = plan_rows
        self.plan_width = plan_width
        self.attributes = []
        self.height = 0
        self.children = []

    def plan2tree(self, plan, vocab):
        root_node = None
        node_type = plan['Node Type']
        if node_type in NOTE_TYPE:
            cost = plan['Total Cost']
            plan_rows = plan['Plan Rows']
            plan_width = plan['Plan Width']
            root_node = PlanTreeNode(node_type, cost, plan_rows, plan_width)
            attributes = []
            if node_type in USEFUL_TYPE:
                if node_type == 'Nested Loop':
                    if 'Join Filter' in plan.keys():
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Join Filter'] and 'p_start_date_sk' in plan['Join Filter']:
                            if 'p_start_date_sk' not in plan['Join Filter'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Join Filter'] and 'p_end_date_sk' in plan['Join Filter']:
                            if 'p_end_date_sk' not in plan['Join Filter'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Join Filter'] and 'r_reason_sk' in plan['Join Filter']:
                            if 'r_reason_sk' not in plan['Join Filter'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Join Filter'] and 's_store_sk' in plan['Join Filter']:
                            if 's_store_sk' not in plan['Join Filter'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            if column.split('#')[1] in plan['Join Filter']:
                                attributes.append(column)
                if node_type == 'Merge Join':
                    vocab_copy = vocab.copy()
                    if 'cp_start_date_sk' in plan['Merge Cond'] and 'p_start_date_sk' in plan['Merge Cond']:
                        if 'p_start_date_sk' not in plan['Merge Cond'].replace('cp_start_date_sk', ''):
                            vocab_copy.remove('promotion#p_start_date_sk')
                    if 'cp_end_date_sk' in plan['Merge Cond'] and 'p_end_date_sk' in plan['Merge Cond']:
                        if 'p_end_date_sk' not in plan['Merge Cond'].replace('cp_end_date_sk', ''):
                            vocab_copy.remove('promotion#p_end_date_sk')
                    if 'sr_reason_sk' in plan['Merge Cond'] and 'r_reason_sk' in plan['Merge Cond']:
                        if 'r_reason_sk' not in plan['Merge Cond'].replace('sr_reason_sk', ''):
                            vocab_copy.remove('reason#r_reason_sk')
                    if 'ss_store_sk' in plan['Merge Cond'] and 's_store_sk' in plan['Merge Cond']:
                        if 's_store_sk' not in plan['Merge Cond'].replace('ss_store_sk', ''):
                            vocab_copy.remove('store#s_store_sk')
                    for column in vocab_copy:
                        if column.split('#')[1] in plan['Merge Cond']:
                            attributes.append(column)
                if node_type == 'Hash Join':
                    vocab_copy = vocab.copy()
                    if 'cp_start_date_sk' in plan['Hash Cond'] and 'p_start_date_sk' in plan['Hash Cond']:
                        if 'p_start_date_sk' not in plan['Hash Cond'].replace('cp_start_date_sk', ''):
                            vocab_copy.remove('promotion#p_start_date_sk')
                    if 'cp_end_date_sk' in plan['Hash Cond'] and 'p_end_date_sk' in plan['Hash Cond']:
                        if 'p_end_date_sk' not in plan['Hash Cond'].replace('cp_end_date_sk', ''):
                            vocab_copy.remove('promotion#p_end_date_sk')
                    if 'sr_reason_sk' in plan['Hash Cond'] and 'r_reason_sk' in plan['Hash Cond']:
                        if 'r_reason_sk' not in plan['Hash Cond'].replace('sr_reason_sk', ''):
                            vocab_copy.remove('reason#r_reason_sk')
                    if 'ss_store_sk' in plan['Hash Cond'] and 's_store_sk' in plan['Hash Cond']:
                        if 's_store_sk' not in plan['Hash Cond'].replace('ss_store_sk', ''):
                            vocab_copy.remove('store#s_store_sk')
                    for column in vocab_copy:
                        if column.split('#')[1] in plan['Hash Cond']:
                            attributes.append(column)
                if node_type == 'Seq Scan':
                    if 'Filter' in plan.keys():
                        column_set = {}
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Filter'] and 'p_start_date_sk' in plan['Filter']:
                            if 'p_start_date_sk' not in plan['Filter'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Filter'] and 'p_end_date_sk' in plan['Filter']:
                            if 'p_end_date_sk' not in plan['Filter'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Filter'] and 'r_reason_sk' in plan['Filter']:
                            if 'r_reason_sk' not in plan['Filter'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Filter'] and 's_store_sk' in plan['Filter']:
                            if 's_store_sk' not in plan['Filter'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            c = column.split("#")[1]
                            if plan['Filter'].find(c) != -1:
                                column_set[column] = plan['Filter'].find(c)
                        sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                        attributes = list(sorted_column_set.keys())
                if node_type == 'CTE Scan':
                    if 'Filter' in plan.keys():
                        column_set = {}
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Filter'] and 'p_start_date_sk' in plan['Filter']:
                            if 'p_start_date_sk' not in plan['Filter'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Filter'] and 'p_end_date_sk' in plan['Filter']:
                            if 'p_end_date_sk' not in plan['Filter'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Filter'] and 'r_reason_sk' in plan['Filter']:
                            if 'r_reason_sk' not in plan['Filter'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Filter'] and 's_store_sk' in plan['Filter']:
                            if 's_store_sk' not in plan['Filter'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            c = column.split("#")[1]
                            if plan['Filter'].find(c) != -1:
                                column_set[column] = plan['Filter'].find(c)
                        sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                        attributes = list(sorted_column_set.keys())
                if node_type == 'Sort':
                    column_set = {}
                    vocab_copy = vocab.copy()
                    if 'cp_start_date_sk' in plan['Sort Key'] and 'p_start_date_sk' in plan['Sort Key']:
                        if 'p_start_date_sk' not in plan['Sort Key'].replace('cp_start_date_sk', ''):
                            vocab_copy.remove('promotion#p_start_date_sk')
                    if 'cp_end_date_sk' in plan['Sort Key'] and 'p_end_date_sk' in plan['Sort Key']:
                        if 'p_end_date_sk' not in plan['Sort Key'].replace('cp_end_date_sk', ''):
                            vocab_copy.remove('promotion#p_end_date_sk')
                    if 'sr_reason_sk' in plan['Sort Key'] and 'r_reason_sk' in plan['Sort Key']:
                        if 'r_reason_sk' not in plan['Sort Key'].replace('sr_reason_sk', ''):
                            vocab_copy.remove('reason#r_reason_sk')
                    if 'ss_store_sk' in plan['Sort Key'] and 's_store_sk' in plan['Sort Key']:
                        if 's_store_sk' not in plan['Sort Key'].replace('ss_store_sk', ''):
                            vocab_copy.remove('store#s_store_sk')
                    for column in vocab_copy:
                        c = column.split("#")[1]
                        if ', '.join(plan['Sort Key']).find(c) != -1:
                            column_set[column] = ', '.join(plan['Sort Key']).find(c)
                    sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                    attributes = list(sorted_column_set.keys())
                if node_type == 'Merge Append':
                    column_set = {}
                    vocab_copy = vocab.copy()
                    if 'cp_start_date_sk' in plan['Sort Key'] and 'p_start_date_sk' in plan['Sort Key']:
                        if 'p_start_date_sk' not in plan['Sort Key'].replace('cp_start_date_sk', ''):
                            vocab_copy.remove('promotion#p_start_date_sk')
                    if 'cp_end_date_sk' in plan['Sort Key'] and 'p_end_date_sk' in plan['Sort Key']:
                        if 'p_end_date_sk' not in plan['Sort Key'].replace('cp_end_date_sk', ''):
                            vocab_copy.remove('promotion#p_end_date_sk')
                    if 'sr_reason_sk' in plan['Sort Key'] and 'r_reason_sk' in plan['Sort Key']:
                        if 'r_reason_sk' not in plan['Sort Key'].replace('sr_reason_sk', ''):
                            vocab_copy.remove('reason#r_reason_sk')
                    if 'ss_store_sk' in plan['Sort Key'] and 's_store_sk' in plan['Sort Key']:
                        if 's_store_sk' not in plan['Sort Key'].replace('ss_store_sk', ''):
                            vocab_copy.remove('store#s_store_sk')
                    for column in vocab_copy:
                        c = column.split("#")[1]
                        if ', '.join(plan['Sort Key']).find(c) != -1:
                            column_set[column] = ', '.join(plan['Sort Key']).find(c)
                    sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                    attributes = list(sorted_column_set.keys())
                if node_type == 'Group':
                    column_set = {}
                    vocab_copy = vocab.copy()
                    if 'cp_start_date_sk' in plan['Group Key'] and 'p_start_date_sk' in plan['Group Key']:
                        if 'p_start_date_sk' not in plan['Group Key'].replace('cp_start_date_sk', ''):
                            vocab_copy.remove('promotion#p_start_date_sk')
                    if 'cp_end_date_sk' in plan['Group Key'] and 'p_end_date_sk' in plan['Group Key']:
                        if 'p_end_date_sk' not in plan['Group Key'].replace('cp_end_date_sk', ''):
                            vocab_copy.remove('promotion#p_end_date_sk')
                    if 'sr_reason_sk' in plan['Group Key'] and 'r_reason_sk' in plan['Group Key']:
                        if 'r_reason_sk' not in plan['GroupKey'].replace('sr_reason_sk', ''):
                            vocab_copy.remove('reason#r_reason_sk')
                    if 'ss_store_sk' in plan['Group Key'] and 's_store_sk' in plan['Group Key']:
                        if 's_store_sk' not in plan['Group Key'].replace('ss_store_sk', ''):
                            vocab_copy.remove('store#s_store_sk')
                    for column in vocab_copy:
                        c = column.split("#")[1]
                        if ', '.join(plan['Group Key']).find(c) != -1:
                            column_set[column] = ', '.join(plan['Group Key']).find(c)
                    sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                    attributes = list(sorted_column_set.keys())
                if node_type == 'Aggregate':
                    if 'Group Key' in plan.keys():
                        column_set = {}
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Group Key'] and 'p_start_date_sk' in plan['Group Key']:
                            if 'p_start_date_sk' not in plan['Group Key'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Group Key'] and 'p_end_date_sk' in plan['Group Key']:
                            if 'p_end_date_sk' not in plan['Group Key'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Group Key'] and 'r_reason_sk' in plan['Group Key']:
                            if 'r_reason_sk' not in plan['Group Key'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Group Key'] and 's_store_sk' in plan['Group Key']:
                            if 's_store_sk' not in plan['Group Key'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            c = column.split("#")[1]
                            if ', '.join(plan['Group Key']).find(c) != -1:
                                column_set[column] = ', '.join(plan['Group Key']).find(c)
                        sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                        attributes = list(sorted_column_set.keys())
                root_node.attributes = attributes
            if 'Plans' in plan.keys():
                root_node.children += root_node.add_child(plan['Plans'], vocab)
        else:
            print(node_type)
            print(plan)
            time.sleep(10000)
        return root_node

    def add_child(self, plans, vocab):
        children = []
        for plan in plans:
            node = None
            node_type = plan['Node Type']
            if node_type in NOTE_TYPE:
                cost = plan['Total Cost']
                plan_rows = plan['Plan Rows']
                plan_width = plan['Plan Width']
                node = PlanTreeNode(node_type, cost, plan_rows, plan_width)
                attributes = []
                if node_type in USEFUL_TYPE:
                    if node_type == 'Nested Loop':
                        if 'Join Filter' in plan.keys():
                            vocab_copy = vocab.copy()
                            if 'cp_start_date_sk' in plan['Join Filter'] and 'p_start_date_sk' in plan['Join Filter']:
                                if 'p_start_date_sk' not in plan['Join Filter'].replace('cp_start_date_sk', ''):
                                    vocab_copy.remove('promotion#p_start_date_sk')
                            if 'cp_end_date_sk' in plan['Join Filter'] and 'p_end_date_sk' in plan['Join Filter']:
                                if 'p_end_date_sk' not in plan['Join Filter'].replace('cp_end_date_sk', ''):
                                    vocab_copy.remove('promotion#p_end_date_sk')
                            if 'sr_reason_sk' in plan['Join Filter'] and 'r_reason_sk' in plan['Join Filter']:
                                if 'r_reason_sk' not in plan['Join Filter'].replace('sr_reason_sk', ''):
                                    vocab_copy.remove('reason#r_reason_sk')
                            if 'ss_store_sk' in plan['Join Filter'] and 's_store_sk' in plan['Join Filter']:
                                if 's_store_sk' not in plan['Join Filter'].replace('ss_store_sk', ''):
                                    vocab_copy.remove('store#s_store_sk')
                            for column in vocab_copy:
                                if column.split('#')[1] in plan['Join Filter']:
                                    attributes.append(column)
                    if node_type == 'Merge Join':
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Merge Cond'] and 'p_start_date_sk' in plan['Merge Cond']:
                            if 'p_start_date_sk' not in plan['Merge Cond'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Merge Cond'] and 'p_end_date_sk' in plan['Merge Cond']:
                            if 'p_end_date_sk' not in plan['Merge Cond'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Merge Cond'] and 'r_reason_sk' in plan['Merge Cond']:
                            if 'r_reason_sk' not in plan['Merge Cond'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Merge Cond'] and 's_store_sk' in plan['Merge Cond']:
                            if 's_store_sk' not in plan['Merge Cond'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            if column.split('#')[1] in plan['Merge Cond']:
                                attributes.append(column)
                    if node_type == 'Hash Join':
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Hash Cond'] and 'p_start_date_sk' in plan['Hash Cond']:
                            if 'p_start_date_sk' not in plan['Hash Cond'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Hash Cond'] and 'p_end_date_sk' in plan['Hash Cond']:
                            if 'p_end_date_sk' not in plan['Hash Cond'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Hash Cond'] and 'r_reason_sk' in plan['Hash Cond']:
                            if 'r_reason_sk' not in plan['Hash Cond'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Hash Cond'] and 's_store_sk' in plan['Hash Cond']:
                            if 's_store_sk' not in plan['Hash Cond'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            if column.split('#')[1] in plan['Hash Cond']:
                                attributes.append(column)
                    if node_type == 'Seq Scan':
                        if 'Filter' in plan.keys():
                            column_set = {}
                            vocab_copy = vocab.copy()
                            if 'cp_start_date_sk' in plan['Filter'] and 'p_start_date_sk' in plan['Filter']:
                                if 'p_start_date_sk' not in plan['Filter'].replace('cp_start_date_sk', ''):
                                    vocab_copy.remove('promotion#p_start_date_sk')
                            if 'cp_end_date_sk' in plan['Filter'] and 'p_end_date_sk' in plan['Filter']:
                                if 'p_end_date_sk' not in plan['Filter'].replace('cp_end_date_sk', ''):
                                    vocab_copy.remove('promotion#p_end_date_sk')
                            if 'sr_reason_sk' in plan['Filter'] and 'r_reason_sk' in plan['Filter']:
                                if 'r_reason_sk' not in plan['Filter'].replace('sr_reason_sk', ''):
                                    vocab_copy.remove('reason#r_reason_sk')
                            if 'ss_store_sk' in plan['Filter'] and 's_store_sk' in plan['Filter']:
                                if 's_store_sk' not in plan['Filter'].replace('ss_store_sk', ''):
                                    vocab_copy.remove('store#s_store_sk')
                            for column in vocab_copy:
                                c = column.split("#")[1]
                                if plan['Filter'].find(c) != -1:
                                    column_set[column] = plan['Filter'].find(c)
                            sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                            attributes = list(sorted_column_set.keys())
                    if node_type == 'CTE Scan':
                        if 'Filter' in plan.keys():
                            column_set = {}
                            vocab_copy = vocab.copy()
                            if 'cp_start_date_sk' in plan['Filter'] and 'p_start_date_sk' in plan['Filter']:
                                if 'p_start_date_sk' not in plan['Filter'].replace('cp_start_date_sk', ''):
                                    vocab_copy.remove('promotion#p_start_date_sk')
                            if 'cp_end_date_sk' in plan['Filter'] and 'p_end_date_sk' in plan['Filter']:
                                if 'p_end_date_sk' not in plan['Filter'].replace('cp_end_date_sk', ''):
                                    vocab_copy.remove('promotion#p_end_date_sk')
                            if 'sr_reason_sk' in plan['Filter'] and 'r_reason_sk' in plan['Filter']:
                                if 'r_reason_sk' not in plan['Filter'].replace('sr_reason_sk', ''):
                                    vocab_copy.remove('reason#r_reason_sk')
                            if 'ss_store_sk' in plan['Filter'] and 's_store_sk' in plan['Filter']:
                                if 's_store_sk' not in plan['Filter'].replace('ss_store_sk', ''):
                                    vocab_copy.remove('store#s_store_sk')
                            for column in vocab_copy:
                                c = column.split("#")[1]
                                if plan['Filter'].find(c) != -1:
                                    column_set[column] = plan['Filter'].find(c)
                            sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                            attributes = list(sorted_column_set.keys())
                    if node_type == 'Sort':
                        column_set = {}
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Sort Key'] and 'p_start_date_sk' in plan['Sort Key']:
                            if 'p_start_date_sk' not in plan['Sort Key'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Sort Key'] and 'p_end_date_sk' in plan['Sort Key']:
                            if 'p_end_date_sk' not in plan['Sort Key'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Sort Key'] and 'r_reason_sk' in plan['Sort Key']:
                            if 'r_reason_sk' not in plan['Sort Key'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Sort Key'] and 's_store_sk' in plan['Sort Key']:
                            if 's_store_sk' not in plan['Sort Key'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            c = column.split("#")[1]
                            if ', '.join(plan['Sort Key']).find(c) != -1:
                                column_set[column] = ', '.join(plan['Sort Key']).find(c)
                        sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                        attributes = list(sorted_column_set.keys())
                    if node_type == 'Merge Append':
                        column_set = {}
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Sort Key'] and 'p_start_date_sk' in plan['Sort Key']:
                            if 'p_start_date_sk' not in plan['Sort Key'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Sort Key'] and 'p_end_date_sk' in plan['Sort Key']:
                            if 'p_end_date_sk' not in plan['Sort Key'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Sort Key'] and 'r_reason_sk' in plan['Sort Key']:
                            if 'r_reason_sk' not in plan['Sort Key'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Sort Key'] and 's_store_sk' in plan['Sort Key']:
                            if 's_store_sk' not in plan['Sort Key'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            c = column.split("#")[1]
                            if ', '.join(plan['Sort Key']).find(c) != -1:
                                column_set[column] = ', '.join(plan['Sort Key']).find(c)
                        sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                        attributes = list(sorted_column_set.keys())
                    if node_type == 'Group':
                        column_set = {}
                        vocab_copy = vocab.copy()
                        if 'cp_start_date_sk' in plan['Group Key'] and 'p_start_date_sk' in plan['Group Key']:
                            if 'p_start_date_sk' not in plan['Group Key'].replace('cp_start_date_sk', ''):
                                vocab_copy.remove('promotion#p_start_date_sk')
                        if 'cp_end_date_sk' in plan['Group Key'] and 'p_end_date_sk' in plan['Group Key']:
                            if 'p_end_date_sk' not in plan['Group Key'].replace('cp_end_date_sk', ''):
                                vocab_copy.remove('promotion#p_end_date_sk')
                        if 'sr_reason_sk' in plan['Group Key'] and 'r_reason_sk' in plan['Group Key']:
                            if 'r_reason_sk' not in plan['GroupKey'].replace('sr_reason_sk', ''):
                                vocab_copy.remove('reason#r_reason_sk')
                        if 'ss_store_sk' in plan['Group Key'] and 's_store_sk' in plan['Group Key']:
                            if 's_store_sk' not in plan['Group Key'].replace('ss_store_sk', ''):
                                vocab_copy.remove('store#s_store_sk')
                        for column in vocab_copy:
                            c = column.split("#")[1]
                            if ', '.join(plan['Group Key']).find(c) != -1:
                                column_set[column] = ', '.join(plan['Group Key']).find(c)
                        sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                        attributes = list(sorted_column_set.keys())
                    if node_type == 'Aggregate':
                        if 'Group Key' in plan.keys():
                            column_set = {}
                            vocab_copy = vocab.copy()
                            if 'cp_start_date_sk' in plan['Group Key'] and 'p_start_date_sk' in plan['Group Key']:
                                if 'p_start_date_sk' not in plan['Group Key'].replace('cp_start_date_sk', ''):
                                    vocab_copy.remove('promotion#p_start_date_sk')
                            if 'cp_end_date_sk' in plan['Group Key'] and 'p_end_date_sk' in plan['Group Key']:
                                if 'p_end_date_sk' not in plan['Group Key'].replace('cp_end_date_sk', ''):
                                    vocab_copy.remove('promotion#p_end_date_sk')
                            if 'sr_reason_sk' in plan['Group Key'] and 'r_reason_sk' in plan['Group Key']:
                                if 'r_reason_sk' not in plan['Group Key'].replace('sr_reason_sk', ''):
                                    vocab_copy.remove('reason#r_reason_sk')
                            if 'ss_store_sk' in plan['Group Key'] and 's_store_sk' in plan['Group Key']:
                                if 's_store_sk' not in plan['Group Key'].replace('ss_store_sk', ''):
                                    vocab_copy.remove('store#s_store_sk')
                            for column in vocab_copy:
                                c = column.split("#")[1]
                                if ', '.join(plan['Group Key']).find(c) != -1:
                                    column_set[column] = ', '.join(plan['Group Key']).find(c)
                            sorted_column_set = dict(sorted(column_set.items(), key=lambda item: item[1]))
                            attributes = list(sorted_column_set.keys())
                    node.attributes = attributes
                if 'Plans' in plan.keys():
                    node.children += node.add_child(plan['Plans'], vocab)
            else:
                print(plan)
                time.sleep(10000)
            children.append(node)
        return children

    def visit_children_tpch(self, nodes, column_output, frequency):
        for node in nodes:
            if len(node.attributes) > 0:
                # 'Nested Loop': dim-1
                if node.node_type == 'Nested Loop':
                    for attr in node.attributes:
                        column_output[attr][15] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Merge Join': dim-1
                if node.node_type == 'Merge Join':
                    for attr in node.attributes:
                        column_output[attr][16] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Hash Join': dim-1
                if node.node_type == 'Hash Join':
                    for attr in node.attributes:
                        column_output[attr][17] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Seq Scan': dim-3
                if node.node_type == 'Seq Scan':
                    for attr in node.attributes[0:3]:
                        column_output[attr][18 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Sort': dim-3
                if node.node_type == 'Sort':
                    for attr in node.attributes[0:3]:
                        column_output[attr][21 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Aggregate': dim-6
                if node.node_type == 'Aggregate':
                    for attr in node.attributes[0:3]:
                        column_output[attr][24 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
            if len(node.children) > 0:
                column_output = node.visit_children(node.children, column_output, frequency)
        return column_output

    def visit_children_tpcds(self, nodes, column_output, frequency):
        for node in nodes:
            if len(node.attributes) > 0:
                # 'Nested Loop': dim-1
                if node.node_type == 'Nested Loop':
                    for attr in node.attributes:
                        column_output[attr][14] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Merge Join': dim-1
                if node.node_type == 'Merge Join':
                    for attr in node.attributes:
                        column_output[attr][15] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Hash Join': dim-1
                if node.node_type == 'Hash Join':
                    for attr in node.attributes:
                        column_output[attr][16] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Seq Scan': dim-3
                if node.node_type == 'Seq Scan':
                    for attr in node.attributes[0:3]:
                        column_output[attr][17 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'CTE Scan': dim-3
                if node.node_type == 'CTE Scan':
                    for attr in node.attributes[0:3]:
                        column_output[attr][20 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Sort': dim-3
                if node.node_type == 'Sort':
                    for attr in node.attributes[0:3]:
                        column_output[attr][23 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Merge Append': dim-3
                if node.node_type == 'Merge Append':
                    for attr in node.attributes[0:3]:
                        column_output[attr][26 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Group': dim-3
                if node.node_type == 'Group':
                    for attr in node.attributes[0:3]:
                        column_output[attr][29 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
                # 'Aggregate': dim-6
                if node.node_type == 'Aggregate':
                    for attr in node.attributes[0:3]:
                        column_output[attr][32 + node.attributes.index(attr)] += math.log(frequency + 1e-8) * math.log(node.plan_rows + 1e-8)
            if len(node.children) > 0:
                column_output = node.visit_children(node.children, column_output, frequency)
        return column_output