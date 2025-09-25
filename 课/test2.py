input_money = int (input("请输入你的充值金额（元）："))
result_money = input_money
people = str(input("您是否第一次来到本店（是或否）：")) 
if 1000 <= input_money < 2000:
    if people == ("是"):
        result_money = input_money * 1.15 + input_money * 0.15 * 0.1
    else:
        result_money = input_money * 1.15
elif 2000 <= input_money < 5000:
    if people == ("是"):
        result_money = input_money * 1.18 + input_money * 0.18* 0.1
    else:
        result_money = input_money * 1.18
elif 5000 <= input_money < 10000:
    if people == ("是"):
        result_money = input_money * 1.20 + (input_money  * 0.20 + 500) * 0.1
    else:
        result_money = input_money * 1.20 + 500
elif 10000 <= input_money :
    if people == ("是"):
        result_money = input_money + 11000
    else:
        result_money = input_money + 10000
else:
    result_money = input_money 
print(f"充值金额为{input_money}元，到账金额为{result_money}" )

