def nw(seq1, seq2, sim_matrix, gap_index):
    """
    An implementation of the Needleman-Wunsch algorithm for pairwise alignment.
    This method is a "translation" of a Java implementation of NW for linguistic purposes written by Johannes Dellert:
    https://github.com/jdellert/iwsa/blob/master/src/main/java/de/jdellert/iwsa/align/NeedlemanWunschAlgorithm.java

    :param seq1: the first sequence to be aligned
    :param seq2: the second sequence to be aligned
    :param sim_matrix: a similarity matrix over all relevant sounds
    :param gap_index: the index in the matrix corresponding to the gap symbol
    :return:
    """
    m = len(seq1) + 1
    n = len(seq2) + 1

    mtx = []
    a_subst = []
    b_subst = []
    for i in range(m):
        mtx.append([0] * n)
        a_subst.append([0] * n)
        b_subst.append([0] * n)

    for i in range(1, m):
        mtx[i][0] = mtx[i - 1][0] + sim_matrix[seq1[i - 1]][gap_index]
        a_subst[i][0] = seq1[i - 1]
        b_subst[i][0] = gap_index  # corresponds to gap symbol

    for j in range(1, n):
        mtx[0][j] = mtx[0][j-1] + sim_matrix[gap_index][seq2[j - 1]]
        a_subst[0][j] = gap_index  # corresponds to gap symbol
        b_subst[0][j] = seq2[j - 1]

    for i in range(1, m):
        for j in range(1, n):
            match_value = mtx[i-1][j-1] + sim_matrix[seq1[i - 1]][seq2[j - 1]]
            insertion_value = mtx[i][j-1] + sim_matrix[gap_index][seq2[j - 1]]
            deletion_value = mtx[i-1][j] + sim_matrix[seq1[i - 1]][gap_index]
            mtx[i][j] = max(match_value, insertion_value, deletion_value)

            if mtx[i][j] == match_value:
                a_subst[i][j] = seq1[i - 1]
                b_subst[i][j] = seq2[j - 1]
            elif mtx[i][j] == insertion_value:
                a_subst[i][j] = gap_index
                b_subst[i][j] = seq2[j - 1]
            else:
                a_subst[i][j] = seq1[i - 1]
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
    for segment in seq1:
        str1_self_similarity += sim_matrix[segment][segment]
    str2_self_similarity = 0
    for segment in seq2:
        str2_self_similarity += sim_matrix[segment][segment]

    similarity_score /= len(result1)
    str1_self_similarity /= m-1
    str2_self_similarity /= n-1

    normalized_distance_score = 1 - (2 * similarity_score) / (str1_self_similarity + str2_self_similarity)

    return [result1, result2, normalized_distance_score]
