def main():
    file_valid = '/data/Corpus/istella22/train.svm'
    wrote_valid = '/data/Corpus/istella22/train1.txt'

    with open(file_valid, 'r') as input_file, open(wrote_valid, 'w') as output_file:
        for line in input_file:
            # 假设文件格式是正确的，不进行错误处理
            parts = line.strip().split(" ")
            y = parts[0]  # label
            qid = parts[1].split(":")[1]  # query id
            features = parts[2:]  # features
            line_w = '{:.0f}'.format(float(y)) + " " + "qid:" + str(int(qid)) + " " + " ".join(features)
            output_file.write(line_w + '\n')

if __name__ == '__main__':
    main()
