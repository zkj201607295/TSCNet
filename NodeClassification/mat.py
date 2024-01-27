import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建一个折线图
plt.figure(figsize=(8, 6))  # 创建一个8x6大小的图像
plt.plot(x, y, label='sin(x)')
plt.title('Sine Function')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# 创建一个散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='data points', color='red', marker='o')
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 创建一个柱状图
categories = ['A', 'B', 'C', 'D']
values = [15, 30, 10, 25]
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='blue')
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.grid(axis='y')
plt.show()

# 创建一个饼图
labels = ['Apples', 'Bananas', 'Cherries', 'Dates']
sizes = [30, 25, 15, 10]
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart')
plt.show()
