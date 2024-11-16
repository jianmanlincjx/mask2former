def extract_lines(input_file, output_file, keyword, ignore_keywords):
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 如果行包含指定的关键词，但不包含忽略的关键词，则写入输出文件
                if keyword in line and not any(ignore_kw in line for ignore_kw in ignore_keywords):
                    outfile.write(line)

# 文件路径和关键词
input_file = '/data1/JM/code/mask2former/output_test/log.txt'  # 替换为你的输入txt文件路径
output_file = '/data1/JM/code/mask2former/output_test/log_split.txt'  # 替换为你希望保存输出的txt文件路径
keyword = 'd2.evaluation.testing INFO: copypaste: '  # 查找此关键词的行
ignore_keywords = ['Task', 'mIoU']  # 需要忽略的关键词列表

# 执行提取操作
extract_lines(input_file, output_file, keyword, ignore_keywords)
