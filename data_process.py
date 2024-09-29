import os
import re
import json
import random
import time
import copy
import math
import numpy as np
from tqdm import tqdm
from psql.PostgreSQL import PGHypo as PG
from planTree import PlanTreeNode


class DataProcessor:
    def __init__(self, config, resource_path, dataset):
        self.config = config
        self.dataset = dataset
        self.db_connector = PG(config)
        print("Gen Column Information...")
        # type
        path1 = f"{resource_path}/type_{dataset}.json"
        if os.path.exists(path1):
            self.type = json.load(open(path1))
        # selectivity
        path2 = f"{resource_path}/1gb_selectivity_{dataset}.json"
        if os.path.exists(path2):
            self.selectivity = json.load(open(path2))
        else:
            self.selectivity = self.db_connector.get_selectivity(self.db_connector.get_tables())
            self.selectivity = dict(sorted(self.selectivity.items(), key=lambda x: x[0]))
            json.dump(self.selectivity, open(path2, 'w'))
        # table rows
        path3 = f"{resource_path}/1gb_table_rows_{dataset}.json"
        if os.path.exists(path3):
            self.table_rows = json.load(open(path3))
        else:
            self.table_rows = {}
            for column in list(self.type.keys()):
                self.table_rows[column] = self.db_connector.get_table_rows(column.split("#")[0])
            json.dump(self.table_rows, open(path3, 'w'))
        # storage
        path4 = f"{resource_path}/1gb_storage_{dataset}.json"
        if os.path.exists(path4):
            self.storage = json.load(open(path4))
        else:
            self.storage = {}
            for column in list(self.type.keys()):
                self.storage[column] = self.db_connector.get_storage(column)
            json.dump(self.storage, open(path4, 'w'))
        self.vocab = list(self.type.keys())
        # column distribution
        path5 = f"{resource_path}/1gb_distribution_{dataset}_v1.json"
        path6 = f"{resource_path}/1gb_distribution_{dataset}_v2.json"
        if os.path.exists(path5) and os.path.exists(path6):
            self.column_distribution = json.load(open(path6))
        else:
            self.column_distribution_v1 = {}
            self.column_distribution_v2 = {}
            for column in self.vocab:
                self.column_distribution_v1[column], self.column_distribution_v2[column] = self.get_distribution(column, self.config["distribution_dim"])
            json.dump(self.column_distribution_v1, open(path5, 'w'))
            json.dump(self.column_distribution_v2, open(path6, 'w'))
            self.column_distribution = self.column_distribution_v1
        for column in self.vocab:
            for i in range(len(self.column_distribution[column])):
                if self.column_distribution[column][i] != 0:
                    self.column_distribution[column][i] = math.log(self.column_distribution[column][i] + 1e-8)
        print("Gen Column Information End")
        self.new_columns = []
        for column in list(self.selectivity.keys()):
            if self.selectivity[column] <= 0.5 and self.table_rows[column] >= 10000:
                self.new_columns.append(column)

    def workload2embedding(self, workload, budget):
        if self.dataset == "tpch" or self.dataset == "chbenchmark":
            # output column matrix
            column_output = {}
            for column in self.vocab:
                column_output[column] = [math.log(self.type[column] + 1e-8)]
                column_output[column] += [self.selectivity[column]]
                column_output[column] += [math.log(self.table_rows[column] + 1e-8)]
                column_output[column] += [math.log(self.storage[column] + 1e-8)]
                column_output[column] += [math.log(budget + 1e-8)]
                column_output[column] += self.column_distribution[column]
                # column_output[column] += [0] * (3 + 5 * 3 + 1)
                column_output[column] += [0] * (1 * 3 + 3 * 3 + 3)
            # plan embedding
            for w in list(workload.keys()):
                frequency = workload[w]
                if ';' in w:
                    ws = w.split(';')
                    for new_w in ws:
                        # 'update': dim-1
                        if "UPDATE" in new_w and "SELECT" not in new_w:
                            table_name = new_w.split("UPDATE ")[1].split(" SET")[0]
                            column_area = new_w.split("SET ")[1].split(" WHERE")[0]
                            columns = self.db_connector.get_columns(table_name)
                            for column in columns:
                                if str.upper(column.split("#")[1]) in column_area:
                                    column_output[column][26] += math.log(frequency + 1e-8)
                        # 'insert': dim-1
                        if "INSERT INTO" in new_w:
                            table_name = new_w.split("INSERT INTO ")[1].split(" (")[0]
                            columns = self.db_connector.get_columns(table_name)
                            for column in columns:
                                column_output[column][27] += math.log(frequency + 1e-8)
                        # 'delete': dim-1
                        if "DELETE FROM" in new_w:
                            table_name = new_w.split("DELETE FROM ")[1].split(" WHERE")[0]
                            columns = self.db_connector.get_columns(table_name)
                            for column in columns:
                                column_output[column][28] += math.log(frequency + 1e-8)
                        plan = self.db_connector.get_plan(new_w)
                        plan_tree = PlanTreeNode().plan2tree(plan, self.vocab)
                        if len(plan_tree.attributes) > 0:
                            # 'Nested Loop': dim-1
                            if plan_tree.node_type == 'Nested Loop':
                                for attr in plan_tree.attributes:
                                    column_output[attr][15] += math.log(frequency + 1e-8) * math.log(
                                        plan_tree.plan_rows + 1e-8)
                            # 'Merge Join': dim-1
                            if plan_tree.node_type == 'Merge Join':
                                for attr in plan_tree.attributes:
                                    column_output[attr][16] += math.log(frequency + 1e-8) * math.log(
                                        plan_tree.plan_rows + 1e-8)
                            # 'Hash Join': dim-1
                            if plan_tree.node_type == 'Hash Join':
                                for attr in plan_tree.attributes:
                                    column_output[attr][17] += math.log(frequency + 1e-8) * math.log(
                                        plan_tree.plan_rows + 1e-8)
                            # 'Seq Scan': dim-3
                            if plan_tree.node_type == 'Seq Scan':
                                for attr in plan_tree.attributes[0:3]:
                                    column_output[attr][18 + plan_tree.attributes.index(attr)] += math.log(
                                        frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                            # 'Sort': dim-3
                            if plan_tree.node_type == 'Sort':
                                for attr in plan_tree.attributes[0:3]:
                                    column_output[attr][21 + plan_tree.attributes.index(attr)] += math.log(
                                        frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                            # 'Aggregate': dim-3
                            if plan_tree.node_type == 'Aggregate':
                                for attr in plan_tree.attributes[0:3]:
                                    column_output[attr][24 + plan_tree.attributes.index(attr)] += math.log(
                                        frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                        if len(plan_tree.children) > 0:
                            column_output = plan_tree.visit_children_tpch(plan_tree.children, column_output, frequency)
                else:
                    plan = self.db_connector.get_plan(new_w)
                    plan_tree = PlanTreeNode().plan2tree(plan, self.vocab)
                    if len(plan_tree.attributes) > 0:
                        # 'Nested Loop': dim-1
                        if plan_tree.node_type == 'Nested Loop':
                            for attr in plan_tree.attributes:
                                column_output[attr][15] += math.log(frequency + 1e-8) * math.log(
                                    plan_tree.plan_rows + 1e-8)
                        # 'Merge Join': dim-1
                        if plan_tree.node_type == 'Merge Join':
                            for attr in plan_tree.attributes:
                                column_output[attr][16] += math.log(frequency + 1e-8) * math.log(
                                    plan_tree.plan_rows + 1e-8)
                        # 'Hash Join': dim-1
                        if plan_tree.node_type == 'Hash Join':
                            for attr in plan_tree.attributes:
                                column_output[attr][17] += math.log(frequency + 1e-8) * math.log(
                                    plan_tree.plan_rows + 1e-8)
                        # 'Seq Scan': dim-3
                        if plan_tree.node_type == 'Seq Scan':
                            for attr in plan_tree.attributes[0:3]:
                                column_output[attr][18 + plan_tree.attributes.index(attr)] += math.log(
                                    frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                        # 'Sort': dim-3
                        if plan_tree.node_type == 'Sort':
                            for attr in plan_tree.attributes[0:3]:
                                column_output[attr][21 + plan_tree.attributes.index(attr)] += math.log(
                                    frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                        # 'Aggregate': dim-3
                        if plan_tree.node_type == 'Aggregate':
                            for attr in plan_tree.attributes[0:3]:
                                column_output[attr][24 + plan_tree.attributes.index(attr)] += math.log(
                                    frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    if len(plan_tree.children) > 0:
                        column_output = plan_tree.visit_children_tpch(plan_tree.children, column_output, frequency)
            return column_output
        elif self.dataset == "tpcds":
            # 输出列矩阵
            column_output = {}
            for column in self.vocab:
                # column_output[column] = [math.log(self.type[column] + 1e-8)]
                column_output[column] = [self.selectivity[column]]
                column_output[column] += [math.log(self.table_rows[column] + 1e-8)]
                column_output[column] += [math.log(self.storage[column] + 1e-8)]
                column_output[column] += [math.log(budget + 1e-8)]
                column_output[column] += self.column_distribution[column]
                # column_output[column] += [0] * (3 + 5 * 3 + 1)
                column_output[column] += [0] * (1 * 3 + 3 * 6 + 3)
            # Plan处理
            for w in list(workload.keys()):
                plan = self.db_connector.get_plan(w)
                plan_tree = PlanTreeNode().plan2tree(plan, self.vocab)
                frequency = workload[w]
                if len(plan_tree.attributes) > 0:
                    # 'Nested Loop': dim-1
                    if plan_tree.node_type == 'Nested Loop':
                        for attr in plan_tree.attributes:
                            column_output[attr][14] += math.log(frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Merge Join': dim-1
                    if plan_tree.node_type == 'Merge Join':
                        for attr in plan_tree.attributes:
                            column_output[attr][15] += math.log(frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Hash Join': dim-1
                    if plan_tree.node_type == 'Hash Join':
                        for attr in plan_tree.attributes:
                            column_output[attr][16] += math.log(frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Seq Scan': dim-3
                    if plan_tree.node_type == 'Seq Scan':
                        for attr in plan_tree.attributes[0:3]:
                            column_output[attr][17 + plan_tree.attributes.index(attr)] += math.log(
                                frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'CTE Scan': dim-3
                    if plan_tree.node_type == 'CTE Scan':
                        for attr in plan_tree.attributes[0:3]:
                            column_output[attr][20 + plan_tree.attributes.index(attr)] += math.log(
                                frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Sort': dim-3
                    if plan_tree.node_type == 'Sort':
                        for attr in plan_tree.attributes[0:3]:
                            column_output[attr][23 + plan_tree.attributes.index(attr)] += math.log(
                                frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Merge Append': dim-3
                    if plan_tree.node_type == 'Merge Append':
                        for attr in plan_tree.attributes[0:3]:
                            column_output[attr][26 + plan_tree.attributes.index(attr)] += math.log(
                                frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Group': dim-3
                    if plan_tree.node_type == 'Group':
                        for attr in plan_tree.attributes[0:3]:
                            column_output[attr][29 + plan_tree.attributes.index(attr)] += math.log(
                                frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                    # 'Aggregate': dim-3
                    if plan_tree.node_type == 'Aggregate':
                        for attr in plan_tree.attributes[0:3]:
                            column_output[attr][32 + plan_tree.attributes.index(attr)] += math.log(
                                frequency + 1e-8) * math.log(plan_tree.plan_rows + 1e-8)
                if len(plan_tree.children) > 0:
                    column_output = plan_tree.visit_children_tpcds(plan_tree.children, column_output, frequency)
            return column_output

    def process_data(self, data):
        progress_bar = tqdm(range(len(data)))
        for d in data:
            d['columns'] = self.workload2embedding(d['workload'], d['budget'])
            progress_bar.update(1)
        return data

    def gen_data(self, data, flag_v1, flag_v2, flag_v3, flag_v4):
        data_v1 = data_v2 = data_v3 = data_v4 = list()
        known_columns = []
        for d in data:
            workload = d['workload']
            for w in list(workload.keys()):
                for column in self.new_columns:
                    if column.split('#')[1] in w and column not in known_columns and column in self.new_columns:
                        known_columns.append(column)
        for c in known_columns:
            if c in self.new_columns:
                self.new_columns.remove(c)
        print(self.new_columns)
        if flag_v1:
            print("Gen helpful data v1: multi frequencies")
            progress_bar = tqdm(range(len(data)))
            data_v1 = copy.deepcopy(data)
            for d in data_v1:
                ratio = random.uniform(0, 1)
                for w in list(d['workload'].keys()):
                    d['workload'][w] = round(d['workload'][w] * ratio)
                progress_bar.update(1)
            print("end")
        if flag_v2:
            print("Gen helpful data v2: hybird workload")
            progress_bar = tqdm(range(len(data)))
            data_v2 = copy.deepcopy(data)
            new_data_v2 = []
            for i in range(len(data_v2)):
                while True:
                    workloads = random.sample(data_v2, 2)
                    workload1 = workloads[0]
                    workload2 = workloads[1]
                    str_workload1 = dict(sorted(workloads[0]['workload'].items(), key=lambda x: x[0]))
                    str_workload2 = dict(sorted(workloads[1]['workload'].items(), key=lambda x: x[0]))
                    if workload1['index'] != workload2['index'] and list(str_workload1.values()) != list(
                            str_workload2.values()) and list(str_workload1.keys()) == list(str_workload2.keys()):
                        ratio1 = random.uniform(0, 1)
                        ratio2 = 1 - ratio1
                        new_workload = copy.deepcopy(workload1)
                        for w in list(workload1['workload'].keys()):
                            new_workload['workload'][w] = round(
                                workload1['workload'][w] * ratio1 + workload2['workload'][w] * ratio2)
                        self.db_connector.delete_indexes()
                        initial_cost = (np.array(
                            self.db_connector.get_queries_cost(list(new_workload['workload'].keys()))) * np.array(
                            list(new_workload['workload'].values()))).sum()
                        if ';' in workload1['index']:
                            indexes = workload1['index'].strip().split(';')
                        else:
                            indexes = [workload1['index']]
                        for index in indexes:
                            self.db_connector.execute_create_hypo_v2(index)
                        current_cost = (np.array(
                            self.db_connector.get_queries_cost(list(new_workload['workload'].keys()))) * np.array(
                            list(new_workload['workload'].values()))).sum()
                        self.db_connector.delete_indexes()
                        label_reward1 = 100 * (initial_cost - current_cost) / initial_cost
                        if ';' in workload2['index']:
                            indexes = workload2['index'].strip().split(';')
                        else:
                            indexes = [workload2['index']]
                        for index in indexes:
                            self.db_connector.execute_create_hypo_v2(index)
                        current_cost = (np.array(
                            self.db_connector.get_queries_cost(list(new_workload['workload'].keys()))) * np.array(
                            list(new_workload['workload'].values()))).sum()
                        self.db_connector.delete_indexes()
                        label_reward2 = 100 * (initial_cost - current_cost) / initial_cost
                        if label_reward1 >= label_reward2:
                            new_workload['index'] = workload1['index']
                        else:
                            new_workload['index'] = workload2['index']
                        break
                    else:
                        continue
                new_data_v2.append(new_workload)
                progress_bar.update(1)
            data_v2 = new_data_v2
            print("end")
        if flag_v3:
            print("Gen helpful data v3: exchange column")
            progress_bar = tqdm(range(int(len(data) / 2)))
            data_v3 = copy.deepcopy(data)
            new_data_v3 = list()
            while len(new_data_v3) < int(len(data) / 2):
                for d in data_v3:
                    flag_join = False
                    replace_index = []
                    for index in d['index'].split(";"):
                        if 'key' not in index:
                            if index not in replace_index:
                                replace_index.append(index)
                            flag_join = True
                    if flag_join:
                        if len(replace_index) == 1:
                            if ',' in index:
                                column = index.split(',')[0]
                            else:
                                column = index
                            suitable_colums = []
                            for c in self.new_columns:
                                pre_str = c.split("#")[1].split("_")[0] + '_'
                                flag_in = False
                                for w in list(d['workload'].keys()):
                                    if pre_str in w:
                                        flag_in = True
                                if flag_in:
                                    suitable_colums.append(c)
                            if len(suitable_colums) != 0:
                                new_column = random.sample(suitable_colums, 1)[0]
                                d['index'] = d['index'].replace(column, new_column)
                                new_workload = dict()
                                flag_right = False
                                for w in list(d['workload'].keys()):
                                    new_w = w.replace(column.split("#")[1], new_column.split("#")[1])
                                    try:
                                        self.db_connector.get_plan(new_w)
                                        flag_right = True
                                    except:
                                        self.db_connector.close()
                                        self.db_connector = PG(self.config)
                                        flag_right = False
                                        break
                                    new_workload[new_w] = d['workload'][w]
                                if flag_right:
                                    d['workload'] = new_workload
                                    new_data_v3.append(d)
                        else:
                            for index in replace_index:
                                if ',' in index:
                                    column = index.split(',')[0]
                                else:
                                    column = index
                                suitable_colums = []
                                for c in self.new_columns:
                                    pre_str = c.split("#")[1].split("_")[0] + '_'
                                    flag_in = False
                                    for w in list(d['workload'].keys()):
                                        if pre_str in w:
                                            flag_in = True
                                    if flag_in:
                                        suitable_colums.append(c)
                                if len(suitable_colums) != 0:
                                    new_column = random.sample(suitable_colums, 1)[0]
                                    d['index'] = d['index'].replace(column, new_column)
                                    new_workload = dict()
                                    flag_right = False
                                    for w in list(d['workload'].keys()):
                                        new_w = w.replace(column.split("#")[1], new_column.split("#")[1])
                                        try:
                                            self.db_connector.get_plan(new_w)
                                            flag_right = True
                                        except:
                                            self.db_connector.close()
                                            self.db_connector = PG(self.config)
                                            flag_right = False
                                            break
                                        new_workload[new_w] = d['workload'][w]
                                    d['workload'] = new_workload
                            if flag_right:
                                new_data_v3.append(d)
                        if flag_right:
                            progress_bar.update(1)
            data_v3 = new_data_v3
            print("end")
        if flag_v4:
            print("Gen helpful data v4: exchange column")
            progress_bar = tqdm(range(len(data)))
            data_v4 = copy.deepcopy(data)
            new_data_v4 = list()
            for d in data_v4:
                key_columns = list()
                for index in d['index'].split(";"):
                    if 'key' in index.split(',')[0]:
                        key_columns.append(index.split(',')[0])
                if len(key_columns) >= 2:
                    sorted_columns = dict()
                    for ukc in key_columns:
                        if ukc.split("_")[1] in list(sorted_columns.keys()):
                            sorted_columns[ukc.split("_")[1]] += 1
                        if ukc.split("_")[1] not in list(sorted_columns.keys()):
                            sorted_columns[ukc.split("_")[1]] = 1
                    flag_same = False
                    sorted_columns = dict(sorted(sorted_columns.items(), key=lambda x: x[1]))
                    if list(sorted_columns.values())[-1] >= 2:
                        later_str = list(sorted_columns.keys())[-1]
                        new_sorted_columns = list()
                        for nsc in key_columns:
                            if nsc.split("_")[1] == later_str:
                                new_sorted_columns.append(nsc)
                        used_key_columns = random.sample(new_sorted_columns, 2)
                        new_workload = dict()
                        for w in list(d['workload'].keys()):
                            if used_key_columns[0].split('#')[1].split("_")[0] in w and \
                                    used_key_columns[1].split('#')[1].split("_")[0]:
                                flag_right = False
                                try:
                                    self.db_connector.get_plan(
                                        self.swap_substrings(w, used_key_columns[0].split('#')[1],
                                                             used_key_columns[1].split('#')[1]))
                                    flag_right = True
                                except:
                                    self.db_connector.close()
                                    self.db_connector = PG(self.config)
                                    flag_right = False
                                if flag_right:
                                    new_workload[self.swap_substrings(w, used_key_columns[0].split('#')[1],
                                                                      used_key_columns[1].split('#')[1])] = \
                                        d['workload'][w]
                        d['workload'] = new_workload
                        new_data_v4.append(d)
                progress_bar.update(1)
            data_v4 = new_data_v4
            print("end")
        new_data = data + data_v1 + data_v2 + data_v3 + data_v4
        random.shuffle(new_data)
        return new_data

    def swap_substrings(self, s, sub1, sub2):
        pattern = re.compile(r'({}|{})'.format(re.escape(sub1), re.escape(sub2)))
        return pattern.sub(lambda m: sub2 if m.group(0) == sub1 else sub1, s)

    def data_add_tg(self, data, vocab, model_dim):
        progress_bar = tqdm(range(len(data)))
        new_data = []
        for d in data:
            if d['index'] == '':
                continue
            t = d['index']
            new_t = []
            new_ty = []
            ts = t.split(";")
            new_ts = ['<start>']
            for item in ts:
                new_ts += item.split(',')
                new_ts.append(';')
            new_ts.append('<end>')
            for item in new_ts:
                src_new = list(d['columns'].values())
                src_copy = []
                # <pad>
                src_copy.append([0] * model_dim)
                # <start>
                src_copy.append([1] * (model_dim // 2) + [0] * (model_dim // 2))
                # ;
                src_copy.append([1] * model_dim)
                # <end>
                src_copy.append([0] * (model_dim // 2) + [1] * (model_dim // 2))
                src_copy += copy.deepcopy(src_new)
                new_t.append(src_copy[vocab.index(item)])
                new_ty.append(vocab.index(item))
            d['tgt'] = new_t
            d['tgt_y'] = new_ty
            progress_bar.update(1)
            new_data.append(d)
        return new_data

    def process_workload(self, workload):
        return self.workload2embedding(workload)

    def rank_indexes(self, indexes, workload):
        self.db_connector.delete_indexes()
        if ';' in indexes:
            indexes = indexes.strip().split(';')
        else:
            indexes = [indexes]
        initial_cost = (np.array(self.db_connector.get_queries_cost(list(workload.keys()))) * np.array(
                        list(workload.values()))).sum()
        indexes_dict = {}
        for index in indexes:
            oid = self.db_connector.execute_create_hypo_v2(index)
            storage = self.db_connector.get_storage_cost(oid)
            current_cost = (np.array(self.db_connector.get_queries_cost(list(workload.keys()))) * np.array(
                        list(workload.values()))).sum()
            self.db_connector.execute_delete_hypo(oid)
            reward = (initial_cost - current_cost) / initial_cost / storage
            # reward = (initial_cost - current_cost) / initial_cost
            indexes_dict[index] = reward
        sorted_indexes = dict(sorted(indexes_dict.items(), key=lambda item: item[1], reverse=True))
        ranked_indexes = ';'.join(list(sorted_indexes.keys()))
        return ranked_indexes

    def data_rank_indexes(self, data):
        progress_bar = tqdm(range(len(data)))
        for d in data:
            d['index'] = self.rank_indexes(d['index'], d['workload'])
            progress_bar.update(1)
        return data

    def data_shuffle_indexes(self, data):
        progress_bar = tqdm(range(len(data)))
        for d in data:
            index = d['index'].split(';')
            random.shuffle(index)
            d['index'] = ';'.join(index)
            progress_bar.update(1)
        return data

    def get_distribution(self, column, num):
        table_name = column.split("#")[0]
        column_name = column.split("#")[1]
        value_interval_v1 = [0] * num
        value_interval_v2 = [0] * num
        distinct_number = self.db_connector.get_distinct_number(table_name, column_name)
        real_number = self.db_connector.get_table_rows(table_name)
        for i in range(num):
            nth1 = i * distinct_number // num
            nth2 = (i + 1) * distinct_number // num
            if (i == 0 and nth2 != 0) or (nth1 == 0 and nth2 != 0):
                nth = nth2
                nth_value = self.db_connector.get_nth_row(table_name, column_name, nth)
                nth_number = self.db_connector.get_interval_number_left(table_name, column_name, nth_value)
                value_interval_v1[i] = nth_number / real_number
                value_interval_v2[i] = nth_number
            elif i == num - 1 and nth1 != 0:
                nth = nth1
                nth_value = self.db_connector.get_nth_row(table_name, column_name, nth)
                nth_number = self.db_connector.get_interval_number_right(table_name, column_name, nth_value)
                value_interval_v1[i] = nth_number / real_number
                value_interval_v2[i] = nth_number
            elif nth1 > 0 and nth1 != nth2:
                nth_value1 = self.db_connector.get_nth_row(table_name, column_name, nth1)
                nth_value2 = self.db_connector.get_nth_row(table_name, column_name, nth2)
                nth_number = self.db_connector.get_interval_number_double(table_name, column_name, nth_value1, nth_value2)
                value_interval_v1[i] = nth_number / real_number
                value_interval_v2[i] = nth_number
        return value_interval_v1, value_interval_v2

    def rank_indexes_v2(self, indexes, workload):
        self.db_connector.delete_indexes()
        initial_cost = (np.array(self.db_connector.get_queries_cost(list(workload.keys()))) * np.array(
                        list(workload.values()))).sum()
        indexes_dict = {}
        for index in indexes:
            oid = self.db_connector.execute_create_hypo_v2(index)
            storage = self.db_connector.get_storage_cost(oid)
            current_cost = (np.array(self.db_connector.get_queries_cost(list(workload.keys()))) * np.array(
                        list(workload.values()))).sum()
            self.db_connector.execute_delete_hypo(oid)
            reward = (initial_cost - current_cost) / initial_cost / storage
            # reward = (initial_cost - current_cost) / initial_cost
            indexes_dict[index] = reward
        sorted_indexes = dict(sorted(indexes_dict.items(), key=lambda item: item[1], reverse=True))
        return list(sorted_indexes.keys())



