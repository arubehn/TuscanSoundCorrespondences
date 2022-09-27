def nw_old(A, B, simMatrix, gap_index, gapPenalty=-1):
    # The Needleman-Wunsch algorithm

    # Stage 1: Create a zero matrix and fills it via algorithm
    n, m = len(A), len(B)
    mat = []
    for i in range(n + 1):
        mat.append([0] * (m + 1))
    for j in range(m + 1):
        mat[0][j] = gapPenalty * j
    for i in range(n + 1):
        mat[i][0] = gapPenalty * i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mat[i][j] = max(mat[i - 1][j - 1] + simMatrix[A[i - 1]][B[j - 1]],
                            mat[i][j - 1] + gapPenalty, mat[i - 1][j] + gapPenalty)

    # Stage 2: Computes the final alignment, by backtracking through matrix
    alignmentA = []
    alignmentB = []
    i, j = n, m
    while i and j:
        score, scoreDiag, scoreUp, scoreLeft = mat[i][j], mat[i - 1][j - 1], mat[i - 1][j], mat[i][j - 1]
        if score == scoreDiag + simMatrix[A[i - 1]][B[j - 1]]:
            alignmentA = [A[i - 1]] + alignmentA
            alignmentB = [B[j - 1]] + alignmentB
            i -= 1
            j -= 1
        elif score == scoreUp + gapPenalty:
            alignmentA = [A[i - 1]] + alignmentA
            alignmentB = [gap_index] + alignmentB
            i -= 1
        elif score == scoreLeft + gapPenalty:
            alignmentA = [gap_index] + alignmentA
            alignmentB = [B[j - 1]] + alignmentB
            j -= 1
    while i:
        alignmentA = [A[i - 1]] + alignmentA
        alignmentB = [gap_index] + alignmentB
        i -= 1
    while j:
        alignmentA = [gap_index] + alignmentA
        alignmentB = [B[j - 1]] + alignmentB
        j -= 1
    # Now return result in format: [1st alignment, 2nd alignment, similarity]
    return [alignmentA, alignmentB, mat[n][m]]


def nw(str1, str2, sim_matrix, gap_index):
    m = len(str1) + 1
    n = len(str2) + 1

    mtx = []
    a_subst = []
    b_subst = []
    for i in range(m):
        mtx.append([0] * n)
        a_subst.append([0] * n)
        b_subst.append([0] * n)

    for i in range(1, m):
        mtx[i][0] = mtx[i - 1][0] + sim_matrix[str1[i-1]][gap_index]
        a_subst[i][0] = str1[i-1]
        b_subst[i][0] = gap_index  # corresponds to gap symbol

    for j in range(1, n):
        mtx[0][j] = mtx[0][j-1] + sim_matrix[gap_index][str2[j-1]]
        a_subst[0][j] = gap_index  # corresponds to gap symbol
        b_subst[0][j] = str2[j-1]

    for i in range(1, m):
        for j in range(1, n):
            match_value = mtx[i-1][j-1] + sim_matrix[str1[i-1]][str2[j-1]]
            insertion_value = mtx[i][j-1] + sim_matrix[gap_index][str2[j-1]]
            deletion_value = mtx[i-1][j] + sim_matrix[str1[i-1]][gap_index]
            mtx[i][j] = max(match_value, insertion_value, deletion_value)

            if mtx[i][j] == match_value:
                a_subst[i][j] = str1[i - 1]
                b_subst[i][j] = str2[j - 1]
            elif mtx[i][j] == insertion_value:
                a_subst[i][j] = gap_index
                b_subst[i][j] = str2[j - 1]
            else:
                a_subst[i][j] = str1[i - 1]
                b_subst[i][j] = gap_index

    i = m-1
    j = n-1
    result1 = []
    result2 = []

    while i or j:
        a_part = a_subst[i][j]
        b_part = b_subst[i][j]
        result1.insert(0, a_part)
        result2.insert(0, b_part)
        if a_part != gap_index:
            i -= 1
        if b_part != gap_index:
            j -= 1
        if a_part == gap_index and b_part == gap_index:
            i -= 1
            j -= 1
        if i < 0 or j < 0:
            break

    similarity_score = mtx[m - 1][n - 1]
    str1_self_similarity = 0
    for segment in str1:
        str1_self_similarity += sim_matrix[segment][segment]
    str2_self_similarity = 0
    for segment in str2:
        str2_self_similarity += sim_matrix[segment][segment]

    similarity_score /= len(result1)
    str1_self_similarity /= m-1
    str2_self_similarity /= n-1

    normalized_distance_score = 1 - (2 * similarity_score) / (str1_self_similarity + str2_self_similarity)

    return [result1, result2, normalized_distance_score]
