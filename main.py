import os
import sys
import json
import time
import copy
import math
import torch
import random
import logging
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from data_process import DataProcessor
from psql.PostgreSQL import PGHypo as PG
from model import LabelSmoothing, make_model, data_gen_train, data_gen_test, SimpleLossCompute, subsequent_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    # training mode
    model.train()
    # model to cuda
    model.to(device)
    for i, batch in enumerate(data_iter):
        torch.cuda.empty_cache()
        batch.src = batch.src.to(device)
        batch.tgt = batch.tgt.to(device)
        batch.src_mask1 = batch.src_mask1.to(device)
        batch.src_mask2 = batch.src_mask2.to(device)
        batch.src_mask3 = batch.src_mask3.to(device)
        batch.tgt_mask = batch.tgt_mask.to(device)
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask1, batch.src_mask2, batch.src_mask3, batch.tgt_mask
        )
        torch.cuda.empty_cache()
        batch.tgt_y = batch.tgt_y.to(device)
        batch.ntokens = batch.ntokens.to(device)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 120 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens


# generate test data
def get_test_data(d_model, test_data):
    # calculate column_mask
    seq_len = V - 4
    block_indexes = [0, 8, 24, 28, 37, 46, 51, 54]
    block_sizes = [8, 16, 4, 9, 9, 5, 3, 7]
    # table_attention
    block_mask1 = torch.ones(seq_len, seq_len)
    for i, size in enumerate(block_sizes):
        start_idx = block_indexes[i]
        end_idx = block_indexes[i] + block_sizes[i]
        block_mask1[start_idx:end_idx, start_idx:end_idx] = 0
    block_mask2 = torch.zeros(seq_len, seq_len)
    # column_attention
    for i, size in enumerate(block_sizes):
        start_idx = block_indexes[i]
        end_idx = block_indexes[i] + block_sizes[i]
        block_mask2[start_idx:end_idx, start_idx:end_idx] = 1
    # data generator
    valid_data_iter = data_gen_test(vocab, test_data, d_model, 1, batch_valid_epoch, block_mask1, block_mask2)
    test_data = []
    for i, batch in enumerate(valid_data_iter):
        test_data.append(batch)
    return test_data


# test model
def test_model_storage_FSM(block_indexes, block_sizes, d_model, model, test_data, pth_path, config,
                           model_dim, vocab_len, max_len, start_symbol):
    # 加载模型参数
    print(f"Loading model from: {pth_path}/model_{dataset}_30p_final.pth")
    model.load_state_dict(torch.load(f"{pth_path}/model_{dataset}_30p_final.pth"))
    print(f"Load model End")
    # 模型切换到预测模型
    model.eval()
    # prepare workloads
    db_connector = PG(config)
    # 计算column_mask
    seq_len = V - 3
    # table_attention
    block_mask1 = torch.ones(seq_len, seq_len)
    for i, size in enumerate(block_sizes):
        start_idx = block_indexes[i]
        end_idx = block_indexes[i] + block_sizes[i]
        block_mask1[start_idx:end_idx, start_idx:end_idx] = 0
    block_mask2 = torch.zeros(seq_len, seq_len)
    # column_attention
    for i, size in enumerate(block_sizes):
        start_idx = block_indexes[i]
        end_idx = block_indexes[i] + block_sizes[i]
        block_mask2[start_idx:end_idx, start_idx:end_idx] = 1
    # 测试数据生成器
    valid_data_iter = data_gen_test(vocab, test_data, d_model, 1, batch_valid_epoch, block_mask1, block_mask2)
    reward_compare_sum = 0
    reward_compare_num = 0
    reward_label_sum = 0
    reward_gen_sum = 0
    print("Inference Begin")
    start_time = time.time()
    test_workload = []
    for i, batch in enumerate(valid_data_iter):
        new_data = {}
        # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(batch.src).to(device)
        new_ys = copy.deepcopy(ys).tolist()

        # 生成new_vocab
        new_vocab = dict()
        new_vocab['<pad>'] = [0.0] * model_dim
        new_vocab['<start>'] = [1.0] * model_dim
        new_vocab[';'] = ([1.0] * int((model_dim // 2))) + ([0.0] * int((model_dim // 2 + model_dim % 2)))
        new_src = batch.src.tolist()[0]
        for k in range(len(new_src)):
            new_vocab[vocab[k + 3]] = new_src[k]

        # 对new_ys启发式编码
        for k in range(len(new_ys[0])):
            new_ys[0][k] = list(new_vocab.values())[int(new_ys[0][k])]
        new_ys = torch.tensor(new_ys).to(device)
        max_tgt_length = max_len
        pe = torch.zeros(max_tgt_length, model_dim)
        position = torch.arange(0, max_tgt_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim // 2 * 2, 2) * -(math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        new_ys = new_ys + pe[:, : new_ys.size(1)].requires_grad_(False)
        # 对ys解码
        str_ys = copy.deepcopy(ys).tolist()[0]
        str_ys = str_ys[1:len(str_ys)]
        for k in range(len(str_ys)):
            str_ys[k] = list(new_vocab.keys())[int(str_ys[k])]
        str_ys = ' '.join(str_ys)
        next_word = -1
        model.to(device)  # 将模型移到设备上
        budget = math.exp(batch.src[0][0][3]) - 1e-8
        # budget = batch.src[0][0][3]
        batch.src = batch.src.to(device)
        batch.tgt = batch.tgt.to(device)
        batch.src_mask1 = batch.src_mask1.to(device)
        batch.src_mask2 = batch.src_mask2.to(device)
        batch.src_mask3 = batch.src_mask3.to(device)
        batch.tgt_mask = batch.tgt_mask.to(device)
        batch.tgt_y = batch.tgt_y.to(device)
        memory = model.encode(batch.src, batch.src_mask1, batch.src_mask2)
        created_indexes = []
        flag_init = True
        flag_index_first = False
        flag_index_inner = False
        flag_seq = False
        flag_done = False
        column_set = []
        label_indexes_lenth = len(batch.tgt_o[0].split(';'))
        index_lenth = 0
        while True:
            if index_lenth >= label_indexes_lenth:
                break
            # 掩码矩阵，1表示掩
            mask = [1] * vocab_len
            # 初始化
            if flag_init:
                flag_init = False
                flag_index_first = True
                flag_index_inner = False
                flag_seq = False
            # 上一步选了分隔符
            if flag_seq:
                flag_index_first = True
                flag_index_inner = False
                flag_seq = False
            # 索引第一个属性
            if flag_index_first:
                # 掩码workload无关属性
                for j in range(batch.src.shape[1]):
                    flag_use = False
                    for k in range(14, 26):
                        if batch.src[0, j, k] != 0.0:
                            flag_use = True
                    if flag_use:
                        mask[j + 3] = 0
                flag_seq = False
                flag_index_first = False
                flag_index_inner = True
            if flag_index_inner:
                upper = lowwer = 0
                for item in block_indexes:
                    if (next_word - 3) >= item:
                        lowwer = item
                    else:
                        break
                upper = lowwer + block_sizes[block_indexes.index(lowwer)]
                if len(column_set) == 1:
                    for i in range(lowwer, upper):
                        flag_use = False
                        for k in range(14, 26):
                            if batch.src[0, i, k] != 0.0:
                                flag_use = True
                        if i != (next_word - 3) and flag_use:
                            mask[i + 3] = 0
                    # 可结束
                    index_str = column_set[0]
                    if index_str not in created_indexes:
                        mask[2] = 0
                        if flag_done:
                            mask[0] = 0
                    flag_index_first = False
                    flag_index_inner = True
                    flag_seq = False
                if len(column_set) == 2:  # 第三属性
                    for i in range(lowwer, upper):
                        flag_same = False
                        column_set_copy = column_set.copy()
                        column_set_copy.append(vocab[3 + i])
                        index_str = ' '.join(column_set_copy)
                        if index_str in created_indexes:
                            flag_same = True
                        flag_use = False
                        for k in range(14, 26):
                            if batch.src[0, i, k] != 0.0:
                                flag_use = True
                        if i != (next_word - 3) and i != (
                                vocab.index(column_set[0]) - 3) and not flag_same and flag_use:
                            mask[i + 3] = 0
                        # 可结束
                        index_str = ' '.join(column_set)
                        if index_str not in created_indexes:
                            mask[2] = 0
                            if flag_done:
                                mask[0] = 0
                        flag_index_first = False
                        flag_index_inner = False
                        flag_seq = True
                if len(column_set) == 3:
                    # 可结束
                    index_str = ' '.join(column_set)
                    if index_str not in created_indexes:
                        mask[2] = 0
                        if flag_done:
                            mask[0] = 0
                        flag_index_first = False
                        flag_index_inner = False
                        flag_seq = True
                    else:
                        break
            if mask == [1] * vocab_len:
                break
            out = model.decode(memory, batch.src_mask3, new_ys, subsequent_mask(ys.size(1)).type_as(batch.src))
            prob = model.generator(out[:, -1]).to('cpu').detach().numpy()
            for j in range(prob.shape[1]):
                if mask[j] == 1:
                    prob[0][j] = -sys.maxsize
            _, next_word = torch.max(torch.tensor(prob), dim=1)
            next_word = next_word.item()
            if next_word == 0:
                break
            if next_word != 2:
                column_set.append(vocab[next_word])
            if next_word == 2:
                index_lenth += 1
                flag_done = True
                flag_seq = True
                flag_index_first = False
                flag_index_inner = False
                indexes_str = str_ys.replace(' ; ', ';').replace(' ;', ';').split(';')
                index_str = indexes_str[len(indexes_str) - 1]
                created_indexes.append(index_str)
                column_set = list()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(batch.src).fill_(next_word)], dim=1).to(device)
            new_ys = copy.deepcopy(ys).tolist()
            # 对new_ys启发式编码
            for k in range(len(new_ys[0])):
                new_ys[0][k] = list(new_vocab.values())[int(new_ys[0][k])]
            new_ys = torch.tensor(new_ys).to(device)
            max_tgt_length = max_len
            pe = torch.zeros(max_tgt_length, model_dim)
            position = torch.arange(0, max_tgt_length).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model_dim // 2 * 2, 2) * -(math.log(10000.0) / model_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).to(device)
            new_ys = new_ys + pe[:, : new_ys.size(1)].requires_grad_(False)
            if new_ys.size(1) > 50:
                break
            # 对ys解码
            str_ys = copy.deepcopy(ys).tolist()[0]
            str_ys = str_ys[1:len(str_ys)]
            for k in range(len(str_ys)):
                str_ys[k] = list(new_vocab.keys())[int(str_ys[k])]
            str_ys = ' '.join(str_ys)
            if next_word == 2:
                next_word = -1
        new_data['budget'] = budget
        new_data['workload'] = batch.workload[0]
        new_data['label_index'] = batch.tgt_o[0]
        new_data['gen_index'] = ';'.join(created_indexes)
        test_workload.append(new_data)
    print("Inference End")
    end_time = time.time()
    print("Evaluate Begin")
    for i in range(len(test_workload)):
        storage_cost = 0
        budget = test_workload[i]['budget']
        workload = test_workload[i]['workload']
        created_indexes = test_workload[i]['gen_index'].split(";")
        init_cost = (np.array(db_connector.get_queries_cost(list(workload.keys()))) * np.array(
            list(workload.values()))).sum()
        # created_indexes = data_processor.rank_indexes_v2(created_indexes, workload)
        db_connector.delete_indexes()
        for j in range(len(created_indexes)):
            oid = db_connector.execute_create_hypo(created_indexes[j])
            storage = db_connector.get_storage_cost(oid)[0] / 1024 / 1024
            storage_cost += storage
            if storage_cost > budget:
                storage_cost -= storage
                db_connector.execute_delete_hypo(oid)
                if len(created_indexes[j].split(" ")) > 1:
                    new_created_index = " ".join(
                        created_indexes[j].split(" ")[0:len(created_indexes[j].split(" ")) - 1])
                    created_indexes[j] = new_created_index
                    oid = db_connector.execute_create_hypo(created_indexes[j])
                    storage = db_connector.get_storage_cost(oid)[0] / 1024 / 1024
                    storage_cost += storage
                    if storage_cost > budget:
                        storage_cost -= storage
                        db_connector.execute_delete_hypo(oid)
        gen_cost = (np.array(db_connector.get_queries_cost(list(workload.keys()))) * np.array(
            list(workload.values()))).sum()
        gen_reward = 100 * (init_cost - gen_cost) / init_cost
        db_connector.delete_indexes()
        # 计算label index的reward
        label_indexes = test_workload[i]['label_index'].split(';')
        label_storage = 0
        for j in range(len(label_indexes)):
            oid = db_connector.execute_create_hypo(label_indexes[j].replace(',', ' '))
            storage = db_connector.get_storage_cost(oid)[0] / 1024 / 1024
            label_storage += storage
            if label_storage > budget:
                db_connector.execute_delete_hypo(oid)
        print(label_storage)
        print(budget)
        label_cost = (np.array(db_connector.get_queries_cost(list(workload.keys()))) * np.array(
            list(workload.values()))).sum()
        label_reward = 100 * (init_cost - label_cost) / init_cost
        db_connector.delete_indexes()
        print(f'Generate Index: {";".join(created_indexes)}')
        print(f'Label    Index: {test_workload[i]["label_index"]}')
        print(f'Reward Compare: {gen_reward} : {label_reward}')
        reward_gen_sum += gen_reward
        reward_label_sum += label_reward
        if label_reward > 1:
            if label_reward >= gen_reward:
                compare = 100 * (label_reward - gen_reward) / label_reward
            else:
                compare = -100 * (gen_reward - label_reward) / gen_reward
            reward_compare_sum += compare
            reward_compare_num += 1
            print(f'Reward    down: {compare}%')
    print("Evaluate End")
    print(f"Inference Time: {end_time - start_time}")
    print(f'Reward Compare Average v1: {reward_compare_sum / reward_compare_num}%')
    print(f'Reward Compare Average v2: {(reward_label_sum - reward_gen_sum) * 100 / reward_label_sum}%')
    return reward_compare_sum / reward_compare_num


if __name__ == '__main__':
    # load config file
    config = json.load(open("config.json"))
    dataset = config['dataset'].split('1')[0]
    if dataset == 'tpch':
        resource_path = 'resource/tpch'
        pth_path = 'pth/tpch'
        model_dim = 30
    elif dataset == 'tpcds':
        resource_path = 'resource/tpcds'
        pth_path = 'pth/tpcds'
        model_dim = 38
    elif dataset == 'chbenchmark':
        resource_path = 'resource/chbenchmark'
        pth_path = 'pth/chbenchmark'
        model_dim = 30
    # set cuda
    device = f"cuda:{config['device']}"
    # training parameters
    epoch_num = config['epoch_num']                                     # training epoch
    batch_size = config['batch_size']                                   # batch_size
    batch_train_epoch = config['batch_train_epoch']                     # batch in each epoch of training
    batch_valid_epoch = config['batch_valid_epoch']                     # batch in each epoch of testing
    vocab = json.load(open(f"{resource_path}/vocab_{dataset}.json"))    # load vocabulary
    V = len(vocab)                                                      # number of vocabulary
    # init model
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)    # loss
    model = make_model(V, N=config['layer_numer'], d_model=model_dim)   # model
    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, betas=(0.9, 0.999), eps=1e-9)
    # apply lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.99)

    # load data
    data = []
    template_workload = []
    # template_data = json.load(open(f"{resource_path}/new_tpch1gb_storage_f10000_random_30p_n5_ranked.json"))
    # template_data = json.load(open(f"{resource_path}/new_tpch1gb_storage_f10000_6_19_30p_n5_ranked.json"))
    template_data = json.load(open(f"{resource_path}/new_tpch1gb_storage_f10000_13_19_30p_n5_ranked.json"))
    random.shuffle(template_data)

    # filter unavailable data
    for d in template_data:
        if d["reward"] > 0 and len(d['index']) != 0 and d['index'] != '':
            template_workload.append(d)
    new_template_workload = []
    for workload in template_workload:
        flag_right = True
        for key in list(workload['workload'].keys()):
            if "(select" in key or "view" in key:
                flag_right = False
        if flag_right:
            new_template_workload.append(workload)
    template_workload = new_template_workload

    # process data
    print("Processing Data...")
    # template_workload = template_workload[0:5000]
    data_processor = DataProcessor(config, resource_path, dataset)
    # rank index
    # data = data_processor.data_rank_indexes(template_workload)
    # json.dump(data, open(f'{resource_path}/new_tpch1gb_storage_f10000_random_30p_n5_ranked.json', 'w'))
    # json.dump(data, open(f'{resource_path}/new_tpch1gb_storage_f10000_3_19_15p_n5_ranked.json', 'w'))
    # json.dump(data, open(f'{resource_path}/new_tpch1gb_storage_f10000_13_19_30p_n5_ranked.json', 'w'))
    # featurization
    # data = data_processor.process_data(data)
    # json.dump(data, open(f'{resource_path}/new_tpch1gb_storage_f10000_random_30p_n5_ranked.json', 'w'))
    # json.dump(data, open(f'{resource_path}/new_tpch1gb_storage_f10000_6_19_30p_n5_ranked.json', 'w'))
    # json.dump(data, open(f'{resource_path}/new_tpch1gb_storage_f10000_13_19_30p_n5_ranked.json', 'w'))
    print("Processing Data End")

    # perturbation strategy
    template_workload = data_processor.gen_data(template_workload, True, True, True, True)
    template_workload = data_processor.process_data(template_workload)
    random.shuffle(template_workload)

    # separate training set and testing set
    data = template_workload
    random.shuffle(data)
    print(f'Number of Data: {len(data)}')
    template_train_data = []
    template_test_data = []
    count = 0
    for i in template_workload:
        if i['index'] == '':
            continue
        if count % 10 != 9:
        # if count % 10 != 9 and count % 10 != 8:
            template_train_data.append(i)
        else:
            template_test_data.append(i)
        count += 1
    train_data = template_train_data
    random.shuffle(train_data)
    test_data = template_test_data
    random.shuffle(test_data)
    print(f'Number of Train_Set: {len(train_data)}')
    print(f'Number of Test_Set: {len(test_data)}')

    # set test data
    random_test_data = json.load(open(f"{resource_path}/new_tpch1gb_storage_f10000_random_30p_n5_ranked.json"))
    other_test_data = json.load(open(f"{resource_path}/new_tpch1gb_storage_f10000_6_19_30p_n5_ranked.json"))

    # training
    flag_train = config['flag_train']
    losses = []
    if flag_train == "True":
        print("Processing Train Data...")
        train_data = data_processor.data_add_tg(train_data, vocab, model_dim)
        print("Processing Train Data End")
        loss_batch = []
        template_reward_trace = []
        other_reward_trace = []
        old_loss_average = new_loss_average = 0
        old_template_test_reward = new_template_test_reward = 100
        old_other_test_reward = new_other_test_reward = 100
        template_test_data = get_test_data(model_dim, template_test_data)
        other_test_data = get_test_data(model_dim, other_test_data)
        for epoch in range(epoch_num):
            print(f"\nepoch {epoch}")
            print("Train...")
            torch.cuda.empty_cache()
            # column_mask
            seq_len = V - 4
            block_indexes = [0, 8, 24, 28, 37, 46, 51, 54]
            block_sizes = [8, 16, 4, 9, 9, 5, 3, 7]
            # table_attention
            block_mask1 = torch.ones(seq_len, seq_len)
            for i, size in enumerate(block_sizes):
                start_idx = block_indexes[i]
                end_idx = block_indexes[i] + block_sizes[i]
                block_mask1[start_idx:end_idx, start_idx:end_idx] = 0
            block_mask2 = torch.zeros(seq_len, seq_len)
            # column_attention
            for i, size in enumerate(block_sizes):
                start_idx = block_indexes[i]
                end_idx = block_indexes[i] + block_sizes[i]
                block_mask2[start_idx:end_idx, start_idx:end_idx] = 1
            # training generator
            data_iter = data_gen_train(train_data, model_dim, batch_size, batch_train_epoch, block_mask1, block_mask2)
            # loss computer
            loss_compute = SimpleLossCompute(model.generator, criterion)
            # train
            train_mean_loss = run_epoch(data_iter, model, loss_compute, optimizer, lr_scheduler)
            losses.append(train_mean_loss.to('cpu'))
            print(f"train loss: {train_mean_loss}")
            logging.info(f"epoch {epoch} train loss: {train_mean_loss}")
            if epoch % 50 == 0 and epoch != 0:
                loss_batch.append(train_mean_loss)
                loss_sum = 0
                for l in loss_batch:
                    loss_sum += l
                loss_sum /= 50
                old_loss_average = new_loss_average
                new_loss_average = loss_sum
                print(f"old_loss_average: {old_loss_average}")
                print(f"new_loss_average: {new_loss_average}")
                old_template_test_reward = new_template_test_reward
                old_other_test_reward = new_other_test_reward
                print('Training End')
                # drwa loss figure
                plt.figure(2)
                x = range(len(losses))
                y = losses
                plt.plot(x, y, marker='x')
                plt.savefig("loss.png", dpi=120)
                plt.clf()
                plt.close()
                # save model
                torch.save(model.state_dict(), f"{pth_path}/transformer_model_{dataset}_30p_c{epoch // 50}_noise_2_4.pth")
                loss_batch = []
                template_reward_trace.append(new_template_test_reward)
                other_reward_trace.append(new_other_test_reward)
                with open("reward_trace_template.txt", "w") as file:
                    for item in template_reward_trace:
                        file.write(str(item) + "\n")
                with open("reward_trace_other.txt", "w") as file:
                    for item in other_reward_trace:
                        file.write(str(item) + "\n")
                """if new_other_test_reward < 5 and old_other_test_reward < new_other_test_reward:
                    print(111)
                    break"""
                """if old_loss_average != 0 and old_loss_average - new_loss_average < 0.01:
                    print(222)
                    print(old_loss_average)
                    print(new_loss_average)
                    break
                else:
                    # save model
                    torch.save(model.state_dict(), f"{pth_path}/transformer_model_{dataset}_30p_c{epoch // 100}.pth")"""
            else:
                loss_batch.append(train_mean_loss)

    # testing
    flag_test = config['flag_test']
    if flag_test == "True":
        print('Testing...')
        block_indexes = [0, 8, 24, 28, 37, 46, 51, 54]
        block_sizes = [8, 16, 4, 9, 9, 5, 3, 7]
        test_model_storage_FSM(block_indexes, block_sizes, model_dim, model, other_test_data, pth_path, config, model_dim, V, max_len=600, start_symbol=1)
        print('Testing End')
