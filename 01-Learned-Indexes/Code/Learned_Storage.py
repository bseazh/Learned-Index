# Main file for storage optimization

import pandas as pd
import numpy as np
from Trained_NN import TrainedNN, ParameterPool, set_data_type, AbstractNN
from btree import BTree
from data.create_data import create_data_storage, Distribution
import time, json, math, getopt, sys, gc, csv

# setting
STORE_NUMBER = 100000
BLOCK_SIZE = 100

# data files for existing data
# “数据文件”    
storePath = {
    Distribution.RANDOM: "data/random_s.csv",
    Distribution.BINOMIAL: "data/binomial_s.csv",
    Distribution.POISSON: "data/poisson_s.csv",
    Distribution.EXPONENTIAL: "data/exponential_s.csv",
    Distribution.NORMAL: "data/normal_s.csv",
    Distribution.LOGNORMAL: "data/lognormal_s.csv"
}

# data files for data to be inserted
# “生成数据” 位置存放
toStorePath = {
    Distribution.RANDOM: "data/random_t.csv",
    Distribution.BINOMIAL: "data/binomial_t.csv",
    Distribution.POISSON: "data/poisson_t.csv",
    Distribution.EXPONENTIAL: "data/exponential_t.csv",
    Distribution.NORMAL: "data/normal_t.csv",
    Distribution.LOGNORMAL: "data/lognormal_t.csv"
}

# path for write 
# 规范化 字符串
pathString = {
    Distribution.RANDOM: "Random",
    Distribution.BINOMIAL: "Binomial",
    Distribution.POISSON: "Poisson",
    Distribution.EXPONENTIAL: "Exponential",
    Distribution.NORMAL: "Normal",
    Distribution.LOGNORMAL: "Lognormal"
}

# 阈值设定
thresholdPool = {
    Distribution.RANDOM: [1, 1],
    Distribution.EXPONENTIAL: [1, 10000]
}

useThresholdPool = {
    Distribution.RANDOM: [True, True],
    Distribution.EXPONENTIAL: [True, False],
}

# binary search for data segment
# 利用二分查找，找出相应的段(Segment)
def part_binary_search(data_list, pos_list, key):
    start = 0
    end = len(pos_list) - 1
    mid = 0
    while start <= end:
        mid = (start + end) / 2
        if data_list[pos_list[mid]] < key:
            start = mid + 1
        elif data_list[pos_list[mid]] > key:
            end = mid - 1
        else:
            return mid
    if data_list[pos_list[mid]] > key and mid != 0:
        return mid - 1
    else:
        return mid

# binary search for position in data segment
# segment -> position
def pos_binary_search(data_list, key):
    start = 0
    end = len(data_list) - 1
    mid = 0
    while start <= end:
        mid = (start + end) / 2
        if data_list[mid] == key or data_list[mid] == -1:
            return mid
        elif data_list[mid] < key:
            start = mid + 1
        else:
            end = mid - 1
    if data_list[mid] > key:
        return mid - 1
    else:
        return mid

# function for train index
def hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums,
                    learning_rate_nums, keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):

	#生成空壳的 input label 
    stage_length = len(stage_nums)
    col_num = stage_nums[1]
    tmp_inputs = [[[] for i in range(col_num)] for i in range(stage_length)]
    tmp_labels = [[[] for i in range(col_num)] for i in range(stage_length)]
    index = [[None for i in range(col_num)] for i in range(stage_length)]

    # 利用参数导入,放入刚才生成的空壳
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y

    # 测试集
    test_inputs = test_data_x

    # 从上往下 训练模型
    # 得到： index[i][j] <- test (x,y)
    # 通过 index -> train
    # tmp_inputs
    # tmp_label
    for i in range(0, stage_length):
        for j in range(0, stage_nums[i]):

        	# 防止空数据 
            if len(tmp_labels[i][j]) == 0:
                continue
            
            # train_data_x 
            inputs = tmp_inputs[i][j]

			# model 对应的labels
            labels = []
            # 测试标签(已知)
            test_labels = []

            # i == 0 , 根结点
            # 100 / 10 ，[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9]
            if i == 0:
            	# divisor = 10 作为分块的 除数 
                divisor = stage_nums[i + 1] * 1.0 / (STORE_NUMBER / BLOCK_SIZE)
                # 标签： 56 / 10 = 5 
                for k in tmp_labels[i][j]:
                    labels.append(int(k * divisor))
                # 测试标签(已知)：56 / 10 = 5 
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:
            # 如果非根节点，其实 i,j 
                labels = tmp_labels[i][j]
                test_labels = test_data_y

            # TrainedNN 来训练
            tmp_index = TrainedNN(threshold[i], use_threshold[i], core_nums[i], train_step_nums[i], batch_size_nums[i],
                                  learning_rate_nums[i],
                                  keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)
            tmp_index.train()

            # index[i][j] 充当 Model[i][j]
            index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i], tmp_index.cal_err())
            
            # 删除 回收 tmp_index
            del tmp_index
            gc.collect()

            # i 遍历层数
            if i < stage_length - 1:
            	# ind 遍历 input 
                for ind in range(len(tmp_inputs[i][j])):
                    # 预先 利用 index 模型来 对"ind" 预测出 "p" 分发到下一层 , j 
                    # predict 是真实位置 floor
                    p = index[i][j].predict(tmp_inputs[i][j][ind])
                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1
                    
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    # 在训练完模型 index[i][j] 后混合使用 BTree
    # 遍历最底层的叶子结点
    for j in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][j] is None:
            continue
        # 通过计算平均绝对误差
        mean_abs_err = index[stage_length - 1][j].

        # 判断是否大于这个阈值
        if mean_abs_err > threshold[stage_length - 1]:
            print("Using BTree")
            index[stage_length - 1][j] = BTree(2)
            # 利用 inputs , labels 重建BTree
            index[stage_length - 1][j].build(tmp_inputs[stage_length - 1][j], tmp_labels[stage_length - 1][j])
    return index

# calculate data distribution
# learn-density 实际就是设定参数 , 以及要训练的 数据分布
def learn_density(threshold, use_threshold, distribution, train_set_x, train_set_y, test_set_x, test_set_y):
    # 利用参数设定当前 “数据类型”
    set_data_type(distribution)
    if distribution == Distribution.RANDOM:
        parameter = ParameterPool.RANDOM.value
    elif distribution == Distribution.LOGNORMAL:
        parameter = ParameterPool.LOGNORMAL.value
    elif distribution == Distribution.EXPONENTIAL:
        parameter = ParameterPool.EXPONENTIAL.value
    elif distribution == Distribution.NORMAL:
        parameter = ParameterPool.NORMAL.value
    else:
        return

    # 参数设定
    stage_set = parameter.stage_set
    stage_set[1] = int(STORE_NUMBER / 10000)
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    print("*************start Learned NN************")
    print("Start Train")
    # train NN index
    start_time = time.time()
    trained_index = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set,
                                    learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, test_set_x, test_set_y)
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time %f " % learn_time)
    print("*************end Learned NN************")
    # 测出时间 以及 利用 混合模型 来训练出 trained_index
    return trained_index

# main function for storage optimization
def optimize_storage(do_compare, do_record, threshold, use_threshold, data_part_distance, learning_percent, distribution):
    # 输出的路径
    store_path = storePath[distribution]
    to_store_path = toStorePath[distribution]

    tmp_data = pd.read_csv(store_path, header=None)

    # 训练集 以及 测试集 的设定
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    global STORE_NUMBER
    STORE_NUMBER = tmp_data.shape[0]

    # 分支设定
    for i in range(STORE_NUMBER):
        # test_set_x.append(tmp_data.ix[i, 0])
        # test_set_y.append(tmp_data.ix[i, 1])
        train_set_x.append(tmp_data.ix[i, 0])
        train_set_y.append(tmp_data.ix[i, 1])

    # 需要存储的数据data , 已经排好序
    store_data = train_set_x[:]

    to_store_data = pd.read_csv(to_store_path, header=None)

    if do_compare == 1 or do_compare == 2:
        trained_index = learn_density(threshold, use_threshold, distribution, train_set_x, train_set_y, test_set_x,
                                      test_set_y)
        print("************Start Optimization**************")

        # 层数设定 10w -> 10,0000 -> 10个
        stage_size = int(STORE_NUMBER / 10000)
        # 最大、最小值
        min_value = train_set_x[0]
        max_value = train_set_x[-1]


        data_density = []
        data_density_pos = [0]

        # get data segment number according to distance
        # 手动设定的 distance -> 1e3 (random)/ 1e6 (exp)
        # 数据的分布情况来 设定相应的 多少部分 的 data_part 
        data_part_num = int(math.ceil((max_value - min_value) * 1.0 / data_part_distance))
        # 上次算出来的pre
        last_pre = 0
        # 总数 
        store_data_num = len(store_data)
        # calculate how many blocks have been occupied
        # 计算多少个 blocks 来存储 nums
        store_block_num = int(math.ceil(store_data_num * 1.0 / BLOCK_SIZE))
        start_time = time.time()

        # 分part 来预测第一个位置 
        for i in range(1, data_part_num):
            # calculate first data of every data segment
            # 利用 distance 来制作segment
            pre_data = min_value + i * data_part_distance
            # calculate position of data
            # 分发
            pre1 = trained_index[0][0].predict(pre_data)
            if pre1 > stage_size - 1:
                pre1 = stage_size - 1
            # 预测 position
            pre2 = trained_index[1][pre1].predict(pre_data)
            if pre2 > store_block_num:
                pre2 = store_block_num            
            if pre2 <= last_pre:
                continue
            if pre2 >= store_block_num - 1:
                break

            # calculate data ratio, using occupied blocks
            # 放入对应的 segment
            # 位置 position
            data_density_pos.append(pre2 * BLOCK_SIZE)
            # segment 相隔的 密度
            data_density.append(abs(pre2 - last_pre) * 1.0 / store_block_num)
            last_pre = pre2

        # ∵ 循环的part , 跳出来 , 封尾
        data_density_pos.append(store_data_num)
        data_density.append(abs(store_block_num - 1 - last_pre) * 1.0 / store_block_num)
        data_part_num = len(data_density)

        # enlarge storage according to estimation
        # 通常 learning_percent = 0.5 
        # 
        store_data = train_set_x[:]
        total_data_num = int(math.ceil(store_block_num * BLOCK_SIZE * (1.0 / learning_percent)))
        # -1 填充 剩余空间
        for i in range(total_data_num - store_data_num):
            store_data.append(-1)
        # block_pos = 0.95 * total_data_num 所在的位置 （19/20=0.95）
        block_pos = total_data_num - int(
            math.ceil(total_data_num * (1.0 / store_block_num) ) )

        data_optimization_pos = []
        data_free_pos = []

        # move data to new position
        # 循环 这个 data_part_num
        # 譬如 data_density = [ 0.1 * 10 ]
        for i in range(data_part_num, 0, -1):
            block_pos -= int(round(data_density[i - 1] * total_data_num))
            if block_pos <= 0 :                
                data_optimization_pos.insert(0, 0)                
                data_free_pos.insert(0, data_density_pos[i])
                break
            # record first data position in every data segment
            # 0.95 0.85 , 0.75 , 0.65 , 0.55 ... 
            data_optimization_pos.insert(0, block_pos)            
            # move data to new position
            store_data[block_pos: block_pos + data_density_pos[i] - data_density_pos[i - 1]] = \
                store_data[data_density_pos[i - 1]:data_density_pos[i]]
            # free space in old position
            # 出现交叉 则用-1填充
            if block_pos < data_density_pos[i]:
                store_data[data_density_pos[i - 1]:block_pos] = [-1] * (block_pos - data_density_pos[i - 1])
            else:
                store_data[data_density_pos[i - 1]:data_density_pos[i]] = [-1] * (
                        data_density_pos[i] - data_density_pos[i - 1])
            # record first free position (for insertion) in every data segment
            data_free_pos.insert(0, block_pos + data_density_pos[i] - data_density_pos[i - 1])
        end_time = time.time()
        average_optimize_time = (end_time - start_time) * 1.0 / to_store_data.shape[0]

        print("Average Optimize Time: %lf" % average_optimize_time)
        # 计算 其 均值 和 方差
        std_deviation = np.std(data_density)
        mean_density = np.mean(data_density)

        print("Density Standard Deviation: %f" % std_deviation)
        print("Mean Density: %f" % mean_density)

        if do_record:
            with open('optimization_result.csv', 'wb') as csvFile:
                csv_writer = csv.writer(csvFile)
                for i in store_data:
                    csv_writer.writerow([i])

        move_steps = len(train_set_x)
        # test optimization
        print("************With Optimization**************")
        start_time = time.time()        
        for i in range(to_store_data.shape[0]):
            pre_data = to_store_data.ix[i, 0]
            # calculate insertion position
            # find data segment for insertion
            part = part_binary_search(store_data, data_optimization_pos, pre_data)
            # find position in data segment for insertion
            pos = data_optimization_pos[part] + pos_binary_search(
                store_data[data_optimization_pos[part]: data_free_pos[part]], pre_data)
            ins_pos = data_free_pos[part]
            # insert data
            while store_data[ins_pos] != -1 and ins_pos < len(store_data) - 1:
                ins_pos += 1
            if ins_pos == len(store_data) - 1:
                store_data.append(-1)
            # 往后挪动
            store_data[pos + 2: ins_pos + 1] = store_data[pos + 1:ins_pos]
            data_free_pos[part] = ins_pos + 1
            store_data[pos + 1] = pre_data
            # 需要插入的位置 - 当前的pos 位置
            move_steps += ins_pos - pos
        end_time = time.time()
        # calculate moving steps and time
        average_move_steps = (move_steps * 1.0 / to_store_data.shape[0])
        average_move_time = (end_time - start_time) * 1.0 / to_store_data.shape[0]
        average_insert_time = average_move_time + average_optimize_time
        print("Average Move Steps: %f" % average_move_steps)
        print("Average Move Time: %f" % average_move_time)
        print("Average Insert Time: %f" % average_insert_time)
        result = [{"Average Moving Steps": average_move_steps, "Average Moving Time": average_move_time,
                   "Average Optimizing Time": average_optimize_time, "Average Insert Time": average_insert_time,
                   "Mean Density": mean_density, "Density Standard Deviation": std_deviation}]
        with open("store_performance/" + pathString[distribution] + "/optimization/" + str(
                data_part_distance) + "_" + str(learning_percent) + ".json", "wb") as jsonFile:
            json.dump(result, jsonFile)

        if do_record:
            with open('insert_result.csv', 'wb') as csvFile:
                csv_writer = csv.writer(csvFile)
                for i in store_data:
                    csv_writer.writerow([i])

    # 不增加新的空位进行 冗余
    if do_compare == 0 or do_compare == 2:
        # test no optimization
        print("************Without Optimization**************")
        store_data = train_set_x[:]
        move_steps = 0
        start_time = time.time()
        for i in range(to_store_data.shape[0]):
            pre_data = to_store_data.ix[i, 0]
            pos = pos_binary_search(store_data, pre_data)
            store_data.append(-1)
            store_data[pos + 2:len(store_data)] = store_data[pos + 1:len(store_data) - 1]
            store_data[pos + 1] = pre_data
            move_steps += len(store_data) - pos - 3
        end_time = time.time()
        average_move_steps = (move_steps * 1.0 / to_store_data.shape[0])
        average_move_time = (end_time - start_time) * 1.0 / to_store_data.shape[0]
        print("Average Move Steps: %f" % average_move_steps)
        print("Average Move Time: %f" % average_move_time)

        result = [{"Average Moving Steps": average_move_steps, "Average Moving Time": average_move_time}]

        with open("store_performance/" + pathString[distribution] + "/no_optimization/"
                  + str(learning_percent) + ".json", "wb") as jsonFile:
            json.dump(result, jsonFile)

# help message
def show_help_message(msg):
    help_message = {
        'command': 'python Learned_BTree.py -d <Distribution> [-p] [Percent] '
                   '[-s] [Distance] [-c] [Compare] [-n] [New data] [-r] [Record] [-h]',
        'distribution': 'Distribution: random, exponential',
        'percent': 'Percent: 0.1-1.0, default value = 0.5; train data size = 100,000',
        'distance': 'Distance:'
                    '[Random: 100-100,000, default = 1,000; '
                    'Exponential: 100,000-100,000,000, default = 1,000,000]',
        'compare': 'Compare: INTEGER, 2 for comparing, 1 for only optimization, 0 for only no optimization, default = 2',
        'new data': 'New data: INTEGER, 0 for no creating new data file, others for creating, default = 1',
        'record': 'Record: INTEGER, 0 for no printing out result, others for printing, default = 0',
        'noDistributionError': 'Please choose the distribution first.'}
    help_message_key = ['command', 'distribution', 'percent', 'distance', 'compare', 'new data', 'record']
    if msg == 'all':
        for k in help_message_key:
            print(help_message[k])

    else:
        print(help_message['command'])
        print('Error! ' + help_message[msg])

# command line
def main(argv):
    distribution = None
    per = 0.5
    num = 100000
    is_distribution = False
    distance = 1000
    do_compare = 2
    do_create = True
    do_record = False
    try:
        opts, [] = getopt.getopt(argv, "hd:s:p:c:n:r:")
    except getopt.GetoptError:
        show_help_message('command')
        sys.exit(2)
    for opt, arg in opts:
        arg = str(arg).lower()
        if opt == '-h':
            show_help_message('all')
            return
        elif opt == '-d':
            if arg == "random":
                distribution = Distribution.RANDOM
                is_distribution = True
                distance = 1000
            elif arg == "exponential":
                distribution = Distribution.EXPONENTIAL
                is_distribution = True
                distance = 1000000
            else:
                show_help_message('distribution')
                return
        elif opt == '-p':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            per = float(arg)
            if not 0.1 <= per <= 1.0:
                show_help_message('percent')
                return

        elif opt == '-s':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            distance = int(arg)
            if not 10 <= distance <= 100000000:
                show_help_message('distance')
                return

        elif opt == '-c':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            do_compare = int(arg)
            if not (do_compare == 0 or do_compare == 1 or do_compare == 2):
                return

        elif opt == '-n':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            do_create = not (int(arg) == 0)
        
        elif opt == '-r':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            do_record = not (int(arg) == 0)


        else:
            print("Unknown parameters, please use -h for instructions.")
            return

    if not is_distribution:
        show_help_message('noDistributionError')
        return
    if do_create:
        create_data_storage(distribution, per, num)
    optimize_storage(do_compare, do_record, thresholdPool[distribution], useThresholdPool[distribution], distance, per,
                     distribution)


if __name__ == "__main__":
    main(sys.argv[1:])
