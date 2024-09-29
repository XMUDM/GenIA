import os
from configparser import ConfigParser
from typing import List
import psycopg2 as pg
import pandas as pd
import time
import sys
import json


class PGHypo:
    def __init__(self, config):
        defaults = config["psql_connect"]
        self.host = defaults['pg_ip']
        self.port = defaults['pg_port']
        self.user = defaults['pg_user']
        self.password = defaults['pg_password']
        self.database = config["dataset"]
        self.conn = pg.connect(database=self.database, user=self.user, password=self.password, host=self.host,
                               port=self.port)

    def close(self):
        self.conn.close()

    def execute_create_hypo(self, index):
        if index == '':
            return
        table = index.split(" ")[0].split("#")[0]
        columns = index.split(" ")
        columns_str = ''
        for column in columns:
            columns_str = columns_str + ',' + column.split("#")[1]
        columns_str = columns_str[1:len(columns_str)]
        sql = "SELECT indexrelid FROM hypopg_create_index('CREATE INDEX ON " + table + "(" + columns_str + ")') ;"
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
        except:
            self.conn.commit()
            return 0
        rows = cur.fetchall()
        return int(rows[0][0])

    def execute_create_hypo_v2(self, index):
        table = index.split("#")[0]
        if ',' in index:
            columns = index.split(",")
        else:
            columns = [index]
        columns_str = ''
        for column in columns:
            columns_str = columns_str + ',' + column.split("#")[1]
        columns_str = columns_str[1:len(columns_str)]
        sql = "SELECT indexrelid FROM hypopg_create_index('CREATE INDEX ON " + table + "(" + columns_str + ")') ;"
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
        except:
            self.conn.commit()
            return 0
        rows = cur.fetchall()
        return int(rows[0][0])

    def execute_delete_hypo(self, oid):
        sql = "select * from hypopg_drop_index(" + str(oid) + ");"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        flag = str(rows[0][0])
        if flag == "t":
            return True
        return False

    def get_queries_cost(self, query_list):
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            query = "explain " + query
            cur.execute(query)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_list.append(float(cost_info[cost_info.index("..") + 2:cost_info.index(" rows=")]))
        return cost_list

    def get_storage_cost(self, oid_list):
        costs = list()
        cur = self.conn.cursor()
        for i, oid in enumerate([oid_list, ]):
            if oid == 0:
                continue
            sql = "select * from hypopg_relation_size(" + str(oid) + ");"
            cur.execute(sql)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_long = int(cost_info)
            costs.append(cost_long)
            # print(cost_long)
        return costs

    def get_rel_cost(self, query_list):
        print("real")
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        cost_sum = 0
        for i, query in enumerate(query_list):
            print("Query :" + str(i))
            print(query)
            _start = time.time()
            cur.execute(query)
            _end = time.time()
            cost_list.append(_end - _start)
            cost_sum += (_end - _start)
        return cost_sum

    def execute_sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def delete_indexes(self):
        sql = 'select * from hypopg_reset();'
        self.execute_sql(sql)

    def get_sel(self, table_name, condition):
        cur = self.conn.cursor()
        totalQuery = "select * from " + table_name + ";"
        cur.execute("EXPLAIN " + totalQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from " + table_name + " Where " + condition + ";"
        # print(resQuery)
        cur.execute("EXPLAIN  " + resQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        return select_rows / total_rows

    def create_indexes(self, indexes):
        i = 0
        for index in indexes:
            schema = index.split("#")
            sql = 'CREATE INDEX START_X_IDx' + str(i) + ' ON ' + schema[0] + "(" + schema[1] + ');'
            print(sql)
            self.execute_sql(sql)
            i += 1

    def delete_t_indexes(self):
        sql = "SELECT relname from pg_class where relkind = 'i' and relname like 'start_x_idx%';"
        print(sql)
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        indexes = []
        for row in rows:
            indexes.append(row[0])
        print(indexes)
        for index in indexes:
            sql = 'drop index ' + index + ';'
            print(sql)
            self.execute_sql(sql)

    def get_tables(self):
        tables_sql = "select tablename from pg_tables where schemaname='public';"
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()
        table_names = list()
        for i, table_name in enumerate(rows):
            table_names.append(table_name[0])
        return table_names

    def get_table_rows(self, table):
        tables_sql = f"select count(*) from {table};"
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()[0][0]
        return rows

    def get_columns(self, table_name):
        attrs_sql = f"select column_name, data_type from information_schema.columns where table_schema='public' and table_name='{table_name}';"
        cur = self.conn.cursor()
        cur.execute(attrs_sql)
        rows = cur.fetchall()
        attrs = list()
        for i, attr in enumerate(rows):
            info = str(table_name) + "#" + str(attr[0])
            attrs.append(info)
        return attrs

    def get_selectivity(self, tables):
        sql1 = f"select relname, reltuples from pg_class where relname in {tuple(tables)}"
        sql2 = "select tablename,attname,n_distinct from pg_stats where schemaname = 'public'"
        cur = self.conn.cursor()
        cur.execute(sql1)
        result1 = cur.fetchall()
        cur.execute(sql2)
        result2 = cur.fetchall()
        index_selectivities = {}
        for i in result2:
            if i[2] < 0:
                index_selectivities[i[0] + "#" + i[1]] = -i[2]
            else:
                sum = 0
                for j in result1:
                    if j[0] == i[0]:
                        sum = j[1]
                        break
                index_selectivities[i[0] + "#" + i[1]] = i[2] / sum
        return index_selectivities

    def get_cardinal(self):
        sql = "select tablename,attname,n_distinct from pg_stats where schemaname = 'public'"
        cur = self.conn.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        index_cardinals = {}

        for i in result:
            if i[2] < 0:
                sql = f"select count(distinct({i[1]})) from {i[0]}"
                cur.execute(sql)
                result2 = cur.fetchall()
                index_cardinals[i[0] + "#" + i[1]] = int(result2[0][0]) / 1000
            else:
                index_cardinals[i[0] + "#" + i[1]] = int(i[2]) / 1000
        return index_cardinals

    def get_storage(self, column):
        oid = self.execute_create_hypo(column)
        cost = self.get_storage_cost(oid)[0]
        self.execute_delete_hypo(oid)
        return round(cost / 1024 / 1024, 2)

    def get_plan(self, query):
        statement = f"explain (format json) {query};"
        cur = self.conn.cursor()
        cur.execute(statement)
        result = cur.fetchall()
        query_plan = result[0][0][0]["Plan"]
        return query_plan

    def get_indexable_columns(self, tables, num):
        alpha = num  # 索引选择性筛选条件
        sql1 = f"select relname, reltuples from pg_class where relname in {tuple(tables)}"
        sql2 = "select tablename,attname,n_distinct from pg_stats where schemaname = 'public'"
        cur = self.conn.cursor()
        cur.execute(sql1)
        result1 = cur.fetchall()
        cur.execute(sql2)
        result2 = cur.fetchall()
        index_selectivities = {}
        for i in result2:
            if i[2] < 0:
                if -i[2] <= alpha:
                    index_selectivities[i[0] + "#" + i[1]] = -i[2]
            else:
                sum = 0
                for j in result1:
                    if j[0] == i[0]:
                        sum = j[1]
                        break
                if i[2] / sum <= alpha:
                    index_selectivities[i[0] + "#" + i[1]] = i[2] / sum
        return index_selectivities

    def get_rel_index(self):
        statement = "select tablename, indexdef from pg_indexes where schemaname = 'public';"
        cur = self.conn.cursor()
        cur.execute(statement)
        result = cur.fetchall()
        indexes = []
        for r in result:
            table = r[0]
            columns = r[1]
            columns = columns.split('btree (')[1].split(')')[0].split(', ')
            cs = []
            for column in columns:
                cs.append(table + '#' + column)
            indexes += [cs]
        return indexes

    def get_distinct_number(self, table_name, column_name):
        sql = f"select count(distinct({column_name})) from {table_name};"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()[0][0]
        return rows

    def get_nth_row(self, table_name, column_name, n):
        sql = f"select distinct({column_name}) from {table_name} order by {column_name} asc limit 1 offset {n-1};"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()[0][0]
        return rows

    def get_interval_number_double(self, table_name, column_name, value1, value2):
        sql = f"select count(*) from {table_name} where {column_name} > '{value1}' and {column_name} <= '{value2}';"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()[0][0]
        return rows

    def get_interval_number_left(self, table_name, column_name, value):
        sql = f"select count(*) from {table_name} where {column_name} <= '{value}';"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()[0][0]
        return rows

    def get_interval_number_right(self, table_name, column_name, value):
        sql = f"select count(*) from {table_name} where {column_name} > '{value}';"
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()[0][0]
        return rows
