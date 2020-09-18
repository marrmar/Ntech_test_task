def findMaxSubArray(a):
    d = 0 #максимальная сумма подмассива с концом в этот момент времени, сумма не может быть отрицательной, так как сумма пустого массива равна 0
    begin_ind = 0
    end_ind = -1
    max_sum = 0
    begin_current = 0 #начало нынешнего отрезка суммы
    for i in range(0, len(a)):
        if d >= 0:
            d += a[i]
        else:
            d = a[i]
            begin_current = i
        if max_sum < d:
            max_sum = d
            begin_ind = begin_current
            end_ind = i
    return a[begin_ind:end_ind+1]


a = [*map(int, input().split())]
print(findMaxSubArray(a))
