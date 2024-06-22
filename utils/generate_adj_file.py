import numpy as np
import pandas as pd
import pickle


class RelationCalculator:
    def __init__(self):
        self.classes = 5
        self.input_path = './multi_label/generate_adj/test_split_233472_20.csv'
        self.output_path = './utils/cavitation_test_adj/cavitation_test_split_233472_20_adj.pkl'
        self.adj_matrix = np.zeros(shape = (self.classes, self.classes))
        self.nums_matrix = np.zeros(shape = (self.classes))

    def load_label_data(self):
        try:
            self.label_dataset = pd.read_csv(self.input_path)
            self.label_dataset = self.label_dataset['label'].values.astype(int)
        except FileNotFoundError:
            print("Error: The label data file was not found")

    def get_nums_matrix(self):
        for data in self.label_dataset:
            self.nums_matrix[data] += 1
        return self.nums_matrix

    def calculate_relation(self, class1, class2):
        if class1 < 0 or class2 < 0 or class1 >= len(self.nums_matrix) or class2 >= len(self.nums_matrix):
            return None
        if class1 == class2:
            return 1.0
        return self.nums_matrix[class1] / (self.nums_matrix[class1] + self.nums_matrix[class2])

    def make_adj_file(self):
        for i in range(self.classes):
            for j in range(self.classes):
                self.adj_matrix[i][j] = self.calculate_relation(i,j)

        print('adj_matrix', '\n', self.adj_matrix)
        print('nums_matrix', '\n', self.nums_matrix)  

        adj = {'adj': self.adj_matrix, 'nums': self.nums_matrix}

        try:
            with open(self.output_path, 'wb') as file:
                pickle.dump(adj, file, protocol = 4)

        # pickle.dump(adj, open(self.output_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            print("Error: The directory './adj/xxxx/' does not exist.")

if __name__ == '__main__':
    calculator = RelationCalculator()
    calculator.load_label_data()
    calculator.get_nums_matrix()
    calculator.make_adj_file()












