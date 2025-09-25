from random import randint

right_number = randint (1,10)
user_guess = None
count = 0 #输入的次数

while user_guess != right_number :
    print(f"欢迎来到猜数字游戏，这是你第{count + 1}次玩此游戏!")
    count +=1
    #用户输入一个1-10的数字
    user_guess = int(input("请输入一个1-10的数字："))
    if user_guess < 1 or user_guess >10 :
        print("输入错误，请输入1-10之间的数字！！")
    if user_guess < right_number and 1 <= user_guess <=10:
        print("你输入的小了")
    if user_guess > right_number and 1 <= user_guess <=10:
        print("你输入的大了")
else:
    print("输入正确")
