# 使用循环和列表推导式创建数组，数组内部包含多个空列表
num_lists = 5
array_of_empty_lists = [[] for _ in range(num_lists)]

array_of_empty_lists[0].append(111)

# 打印数组内容
print(array_of_empty_lists)