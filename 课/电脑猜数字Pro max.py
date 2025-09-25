low_number = 0
high_number = 1000000
import random 
answer = random.randint(low_number,high_number)



middle_number = int((high_number + low_number)/2)

new_high_number = high_number
new_low_number = low_number

guess_count = 0
while middle_number != answer:
    if middle_number > answer:
        new_high_number = middle_number
        middle_number = int((new_high_number+new_low_number)/2)
    elif middle_number < answer:
        new_low_number = middle_number
        middle_number = int((new_high_number+new_low_number)/2)

    guess_count += 1
    
    print (f"已经猜了{guess_count}次，猜的数字为{middle_number}")