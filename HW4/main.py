import numpy as np

class MSA:
    def __init__(self):
        self.seqs = []
        self.query = ""
        self.alphabet = set()
        self.get_input()
        self.dic_alphabet_to_index = {}
        for i in range(len(self.alphabet)):
            self.dic_alphabet_to_index[self.alphabet[i]] = i
        self.M = None
        self.psedudo_count = 2
        self.create_matrix_from_seqs()


    def get_input(self):
        n = int(input())
        for i in range(n):
            self.seqs.append(input())
            for j in range(len(self.seqs[0])):
                self.alphabet.add(self.seqs[i][j])
        self.query = input()
        self.alphabet = list(self.alphabet)
    
    def create_matrix_from_seqs(self):
        self.M = np.zeros((len(self.alphabet), len(self.seqs[0]))) + self.psedudo_count
        for i in range(len(self.seqs[0])):
            for j in range(len(self.seqs)):
                self.M[self.dic_alphabet_to_index[self.seqs[j][i]]][i] += 1
        for i in range(self.M.shape[1]):
            sum_col = np.sum(self.M[:, i])
            for j in range(self.M.shape[0]):
                self.M[j][i] /= sum_col
        for i in range(self.M.shape[0]):
            mean_row = np.mean(self.M[i, :])
            for j in range(self.M.shape[1]):
                self.M[i][j] = np.log2(self.M[i][j] / mean_row)
    
    def calculate_score_for_seq(self, seq):
        score = 0
        for i in range(len(seq)):
            score += self.M[self.dic_alphabet_to_index[seq[i]]][i]
        return score
    
    def create_all_solution_of_nonnegative_sum(self, tmp, number_of_unknown, sum_of_unknown, ans):
        sum_ = sum(tmp)
        if number_of_unknown == 1:
            ans.append(tmp + [sum_of_unknown - sum_])
        else:
            for i in range(sum_of_unknown - sum_ + 1):
                self.create_all_solution_of_nonnegative_sum(
                    tmp + [i],
                    number_of_unknown - 1,
                    sum_of_unknown,
                    ans
                )
        return
    
    def create_seq_from_answer(self, solution, seq):
        ans = ""
        for i in range(len(seq)):
            ans = ans + solution[i] * "-" + seq[i]
        return ans + "-" * solution[-1]
    
    def find_all_seqs_by_seq(self, seq):
        number_of_unknown = len(seq) + 1
        sum_of_unknown = len(self.seqs[0]) - len(seq)
        ans = []
        self.create_all_solution_of_nonnegative_sum([], number_of_unknown, sum_of_unknown, ans)
        best_seq = self.create_seq_from_answer(ans[0], seq)
        best_score = self.calculate_score_for_seq(best_seq)
        all_created_seqs = []
        for i in range(1, len(ans)):
            created_seq = self.create_seq_from_answer(ans[i], seq)
            tmp_score = self.calculate_score_for_seq(created_seq)
            if tmp_score > best_score:
                best_score = tmp_score
                best_seq = created_seq
        return best_seq, best_score
    
    def move_window_on_query(self, window_size):
        best_seq, best_score = self.find_all_seqs_by_seq(self.query[:window_size])
        for i in range(1, len(self.query) - window_size + 1):
            created_seq, created_score = self.find_all_seqs_by_seq(self.query[i:i+window_size])
            if created_score > best_score:
                best_score = created_score
                best_seq = created_seq
        return best_seq, best_score
    
    def find_all_seqs_by_query(self):
        best_seq, best_score = self.move_window_on_query(3)
        for w in range(4, len(self.seqs[0]) + 1):
            created_seq, created_score = self.move_window_on_query(w)
            if created_score > best_score:
                best_score = created_score
                best_seq = created_seq
        return best_seq
    
        
msa = MSA()
print(msa.find_all_seqs_by_query())