from time import sleep
number_input=(input("请输入你所需要的间隔时间（秒）"))
try:
    print("开始加载")
    number = float(number_input)
    for i in range (1,101):
        print(f"{i*"□"}{"■"*(100-i)}{i}%",end="\r")
        sleep(number)

except:
    print("请输入数字，不要输入其他无关信息")
None   #结束符
print()
print("加载完毕！")
"""
即为print("\n加载完毕")
"""