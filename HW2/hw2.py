"""
semi global alignment by dynamic programming
find all answers by backtracking
"""

import numpy as np

GAP = "-"
GAP_PENALTY = -9
PAM250 = {
'A': {'A':  2, 'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K': -1, 'L': -2, 'M': -1, 'N':  0, 'P':  1, 'Q':  0, 'R': -2, 'S':  1, 'T':  1, 'V':  0, 'W': -6, 'Y': -3},
'C': {'A': -2, 'C': 12, 'D': -5, 'E':-5, 'F': -4, 'G': -3, 'H': -3, 'I': -2, 'K': -5, 'L': -6, 'M': -5, 'N': -4, 'P': -3, 'Q': -5, 'R': -4, 'S':  0, 'T': -2, 'V': -2, 'W': -8, 'Y':  0},
'D': {'A':  0, 'C': -5, 'D':  4, 'E': 3, 'F': -6, 'G':  1, 'H':  1, 'I': -2, 'K':  0, 'L': -4, 'M': -3, 'N':  2, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
'E': {'A':  0, 'C': -5, 'D':  3, 'E': 4, 'F': -5, 'G':  0, 'H':  1, 'I': -2, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
'F': {'A': -3, 'C': -4, 'D': -6, 'E':-5, 'F':  9, 'G': -5, 'H': -2, 'I':  1, 'K': -5, 'L':  2, 'M':  0, 'N': -3, 'P': -5, 'Q': -5, 'R': -4, 'S': -3, 'T': -3, 'V': -1, 'W':  0, 'Y':  7},
'G': {'A':  1, 'C': -3, 'D':  1, 'E': 0, 'F': -5, 'G':  5, 'H': -2, 'I': -3, 'K': -2, 'L': -4, 'M': -3, 'N':  0, 'P':  0, 'Q': -1, 'R': -3, 'S':  1, 'T':  0, 'V': -1, 'W': -7, 'Y': -5},
'H': {'A': -1, 'C': -3, 'D':  1, 'E': 1, 'F': -2, 'G': -2, 'H':  6, 'I': -2, 'K':  0, 'L': -2, 'M': -2, 'N':  2, 'P':  0, 'Q':  3, 'R':  2, 'S': -1, 'T': -1, 'V': -2, 'W': -3, 'Y':  0},
'I': {'A': -1, 'C': -2, 'D': -2, 'E':-2, 'F':  1, 'G': -3, 'H': -2, 'I':  5, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -2, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -5, 'Y': -1},
'K': {'A': -1, 'C': -5, 'D':  0, 'E': 0, 'F': -5, 'G': -2, 'H':  0, 'I': -2, 'K':  5, 'L': -3, 'M':  0, 'N':  1, 'P': -1, 'Q':  1, 'R':  3, 'S':  0, 'T':  0, 'V': -2, 'W': -3, 'Y': -4},
'L': {'A': -2, 'C': -6, 'D': -4, 'E':-3, 'F':  2, 'G': -4, 'H': -2, 'I':  2, 'K': -3, 'L':  6, 'M':  4, 'N': -3, 'P': -3, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V':  2, 'W': -2, 'Y': -1},
'M': {'A': -1, 'C': -5, 'D': -3, 'E':-2, 'F':  0, 'G': -3, 'H': -2, 'I':  2, 'K':  0, 'L':  4, 'M':  6, 'N': -2, 'P': -2, 'Q': -1, 'R':  0, 'S': -2, 'T': -1, 'V':  2, 'W': -4, 'Y': -2},
'N': {'A':  0, 'C': -4, 'D':  2, 'E': 1, 'F': -3, 'G':  0, 'H':  2, 'I': -2, 'K':  1, 'L': -3, 'M': -2, 'N':  2, 'P':  0, 'Q':  1, 'R':  0, 'S':  1, 'T':  0, 'V': -2, 'W': -4, 'Y': -2},
'P': {'A':  1, 'C': -3, 'D': -1, 'E':-1, 'F': -5, 'G':  0, 'H':  0, 'I': -2, 'K': -1, 'L': -3, 'M': -2, 'N':  0, 'P':  6, 'Q':  0, 'R':  0, 'S':  1, 'T':  0, 'V': -1, 'W': -6, 'Y': -5},
'Q': {'A':  0, 'C': -5, 'D':  2, 'E': 2, 'F': -5, 'G': -1, 'H':  3, 'I': -2, 'K':  1, 'L': -2, 'M': -1, 'N':  1, 'P':  0, 'Q':  4, 'R':  1, 'S': -1, 'T': -1, 'V': -2, 'W': -5, 'Y': -4},
'R': {'A': -2, 'C': -4, 'D': -1, 'E':-1, 'F': -4, 'G': -3, 'H':  2, 'I': -2, 'K':  3, 'L': -3, 'M':  0, 'N':  0, 'P':  0, 'Q':  1, 'R':  6, 'S':  0, 'T': -1, 'V': -2, 'W':  2, 'Y': -4},
'S': {'A':  1, 'C':  0, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P':  1, 'Q': -1, 'R':  0, 'S':  2, 'T':  1, 'V': -1, 'W': -2, 'Y': -3},
'T': {'A':  1, 'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  0, 'H': -1, 'I':  0, 'K':  0, 'L': -2, 'M': -1, 'N':  0, 'P':  0, 'Q': -1, 'R': -1, 'S':  1, 'T':  3, 'V':  0, 'W': -5, 'Y': -3},
'V': {'A':  0, 'C': -2, 'D': -2, 'E':-2, 'F': -1, 'G': -1, 'H': -2, 'I':  4, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -1, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -6, 'Y': -2},
'W': {'A': -6, 'C': -8, 'D': -7, 'E':-7, 'F':  0, 'G': -7, 'H': -3, 'I': -5, 'K': -3, 'L': -2, 'M': -4, 'N': -4, 'P': -6, 'Q': -5, 'R':  2, 'S': -2, 'T': -5, 'V': -6, 'W': 17, 'Y':  0},
'Y': {'A': -3, 'C':  0, 'D': -4, 'E':-4, 'F':  7, 'G': -5, 'H':  0, 'I': -1, 'K': -4, 'L': -1, 'M': -2, 'N': -2, 'P': -5, 'Q': -4, 'R': -4, 'S': -3, 'T': -3, 'V': -2, 'W':  0, 'Y': 10}
}


class Sequence:
    def __init__(self, i_start, j_start):
        self.seqL = []
        self.seqT = []
        self.i_start = i_start
        self.j_start = j_start
        self.i_end = i_start
        self.j_end = j_start
    
    def add_to_end(self, w):
        self.seqL.insert(0, w[-2])
        self.seqT.insert(0, w[-1])
        self.i_end = w[-4]
        self.j_end = w[-3]
    
    def copy(self):
        s = Sequence(self.i_start, self.j_start)
        s.seqL = self.seqL[:]
        s.seqT = self.seqT[:]
        s.i_end = self.i_end
        s.j_end = self.j_end
        return s
    
    def padding(self, SL, ST):
        max_ = max(len(SL) - self.i_start, len(ST) - self.j_start)
        if max_ > 0:
            if max_ == len(SL) - self.i_start:
                for i in range(max_):
                    self.seqL.append(SL[self.i_start + i])
                    self.seqT.append(GAP)
            else:
                for i in range(max_):
                    self.seqL.append(GAP)
                    self.seqT.append(ST[self.j_start + i])
    
    def prepare(self, SL, ST):
        self.padding(SL, ST)
        return "".join(self.seqL), "".join(self.seqT)


class Aligner:
    def __init__(self, seqL, seqT):
        self.N = len(seqL)
        self.M = len(seqT)
        self.seqL = seqL
        self.seqT = seqT
        self.S = np.zeros((self.N+1, self.M+1, 4))
        self.S[0, :, 2] = 1
        self.S[:, 0, 1] = 1
        self.S[0, 0, 1], self.S[0, 0, 2], self.S[0, 0, 3] = 0, 0, 0
        self.max_ = None
        self.max_pos = []
        self.ANS = []
    
    def calculate_S(self):
        for i in range(1, self.N+1):
            for j in range(1, self.M+1):
                a1, a2, a3 = self.S[i - 1, j, 0] + GAP_PENALTY, self.S[i, j - 1, 0] + GAP_PENALTY, self.S[i - 1, j - 1, 0] + PAM250[self.seqL[i - 1]][self.seqT[j - 1]]
                maximum = max(a1, a2, a3)
                self.S[i, j, 0], self.S[i, j, 1], self.S[i, j, 2], self.S[i, j, 3] = maximum, maximum == a1, maximum == a2, maximum == a3
        self.max_ = self.S[0, self.M, 0]
        self.max_pos = [(0, self.M)]
        for i in range(1, self.N + 1):
            if self.max_ == self.S[i, self.M, 0]:
                self.max_pos.append((i, self.M))
            elif self.max_ < self.S[i, self.M, 0]:
                self.max_ = self.S[i, self.M, 0]
                self.max_pos = [(i, self.M)]
        for j in range(0, self.M):
            if self.max_ == self.S[self.N, j, 0]:
                self.max_pos.append((self.N, j))
            elif self.max_ < self.S[self.N, j, 0]:
                self.max_ = self.S[self.N, j, 0]
                self.max_pos = [(self.N, j)]
    
    def whats_back(self, i, j):
        w = []
        if self.S[i, j, 1]: w.append((i - 1, j, self.seqL[i - 1], GAP))
        if self.S[i, j, 2]: w.append((i, j - 1, GAP, self.seqT[j - 1]))
        if self.S[i, j, 3]: w.append((i - 1, j - 1, self.seqL[i - 1], self.seqT[j - 1]))
        return w
    
    def extend(self, Q):
        seq = Q.pop(0)
        W = self.whats_back(seq.i_end, seq.j_end)
        if len(W) == 0:
            self.ANS.append(seq)
        for w in W:
            tmp = seq.copy()
            tmp.add_to_end(w)
            Q.append(tmp)
    
    def traceback(self, i, j):
        Q = [Sequence(i, j)]
        self.extend(Q)
        while len(Q) > 0:
            self.extend(Q)
    
    def traceback_all(self):
        for m in self.max_pos:
            self.traceback(m[0], m[1])
    
    def print_S(self):
        print(self.S[:, :, 0])


seqT = input()
seqL = input()
aligner = Aligner(seqL, seqT)
aligner.calculate_S()
aligner.traceback_all()
aligner.print_S()
print(int(aligner.max_))
ANS = []
for seq in aligner.ANS:
    L, T = seq.prepare(seqL, seqT)
    ANS.append((T, L))
sortedSeq = [i[0]+i[1] for i in ANS]
sortedSeq.sort()
for i in sortedSeq:
    print(i[0:int(len(i)/2)])
    print(i[int(len(i)/2):])