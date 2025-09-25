import random

while True:
    try:
        low_number = int(input("请输入范围的最低限度（整数）："))
        high_number = int(input("请输入范围的最高限度（整数）："))
    
        if low_number >= high_number :
            print("最小的值怎么可能比最大值大呢？？？")
            continue  #必须要输，才能让while继续循环下去
        
        break  # 如果输入有效，跳出循环
    
    except ValueError :
        print("输入错误，请输入整数")  #如果上面的式子不成立则说明输入的为非整数(可以为任意其他的字符)


guess_number = random.randint (low_number,high_number)
middle_number = int((low_number+high_number)/2)

print(f"（调试用）随机生成的数字是: {guess_number}")  # 调试用，正式运行可删除

new_high_number = high_number
new_low_number = low_number

count = 1
while guess_number != middle_number:
    print (f"猜了{count}次，数字为{middle_number}")
    if middle_number > guess_number:
        new_high_number = middle_number
        middle_number = int((new_low_number + new_high_number)/2)
       
    elif middle_number < guess_number:
        new_low_number = middle_number
        middle_number = int((new_high_number + new_low_number)/2)
    count += 1

print (f"猜了{count}次，数字为{middle_number}")

