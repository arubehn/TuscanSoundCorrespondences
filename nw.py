def nw(A, B, simMatrix, gap_index, gapPenalty=-1):
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
