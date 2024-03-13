from sklearn.datasets import load_svmlight_file
def main():
    file_valid = '/data/Corpus/istella22/valid.svm'
    wrote_valid = '/data/Corpus/istella22/valid.txt'
    offset = 0

    with open(file_valid, 'rb') as input_file, open(wrote_valid, 'w') as output_file:
        for line in input_file:
            X, Y, query_id = load_svmlight_file(input_file, n_features=221, offset=0, query_id=True,length=1)
            data_str = ' '.join([f'{i + 1}:{value}' for i, value in enumerate(X.data)])
            line_w = '{:.0f}'.format(Y[0]) + " " + "qid:" + str(int(query_id)) \
                   + " " + data_str
            output_file.write(line_w + '\n')
if __name__ == '__main__':
    main()